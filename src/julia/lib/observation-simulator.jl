include("../../../config.jl");
include("wolfpack-types.jl");
include(joinpath(LIB_PATH, "simulation-types.jl"));
include(joinpath(LIB_PATH,"overlapping.jl"));

using Distributions
using Random

"""
The function initialises the number of packs given an OverlapMetric.
The output of the function is a Vector{WolfpackLifetime{N}} with N = 1 or 2.

Arguments :
`θ`: The parameters of the model.
`N_init`: the assumed number of wolf packs initially.
`T`: the end time.
`loc_dist`: The distribution of the location of the packs.
`om` : Overlapping metric.
"""
function initialise(θ::WolftrackParameters{N},
                    N_init::Integer,
                    T::Real,
                    loc_dist::MultivariateDistribution,
                    om::OverlapMetric = NoOverlap()) where N

    # Add the N desired packs.
    pack_lifetimes = WolfpackLifetime{N}[];
    for i in 1:N_init
        loc = sample_nonoverlapping(θ, pack_lifetimes, loc_dist, om);
        push!(pack_lifetimes, WolfpackLifetime(0.0, T, loc));
    end
    pack_lifetimes;
end

"""
The function rejection samples a new location vector respecting the
overlapping metric `om` and the existing pack locations in `pack_lifetimes`.
"""
function sample_nonoverlapping(θ::WolftrackParameters{N},
                               pack_lifetimes::Vector{WolfpackLifetime{N}},
                               loc_dist::MultivariateDistribution,
                               om::OverlapMetric = NoOverlap()) where N

    loc_prop = zeros(MVector{N, Float64});
    success = false; # Have we found a suitable candidate?
    while !success
        # Sample a proposed location for the new pack.
        rand!(loc_dist, loc_prop);

        # Compute acceptance probabilities of the location against
        # each existing pack.
        acceptance_probs = map(pack_lifetimes) do ws
            acceptance_probability(loc_prop, θ.Σ_obs,
                                   ws.location, θ.Σ_obs, om);
        end
        # Get the most offending acceptance probability and check if we are
        # successful.
        acceptance_prob = length(pack_lifetimes) == 0 ? 1.0 : minimum(acceptance_probs);
        rand() <= acceptance_prob && (success = true);
    end
    loc_prop;
end

"""
Deprecate?

Generate a history of wolfpacks according to a simplified model.
This function does not allow time-varying intensities.
The birth and death events follow a birth-death process, which is simulated
with a modified version of the Gillespie algorithm.
In this version, the packs do not move, and their initial location is drawn
from some distribution.

Arguments:
`θ`: The parameters of the model.
`N_init`: The number of initial packs.
`T`: The end time of the simulation.
`loc_dist`: The location distribution.

Optional arguments:
`om`: Overlapping metric.
"""
function sim_latent(θ::WolftrackParameters, N_init::Integer, T::Real,
                    loc_dist::MultivariateDistribution;
                    om::OverlapMetric = NoOverlap())
    @assert N_init >= 0 "`N_init` must be >= 0.";
    @assert T > 0.0 "`T` must be > 0.0";

    # Unpack parameters.
    λ_birth = θ.λ_birth;
    λ_birth0 = θ.λ_birth0;
    λ_death = θ.λ_death;

    t = 0.0; # Start time.
    time = Float64[]; # Vector keeping time points.
    N_evolution = Int[]; # Vector keeping the amount of packs at each time point.
    alive = Set(1:N_init); # The set of packs that are alive.
    n_packs = N_init; # The number of current packs.

    # Initialise the vector of wolfpack lifetimes.
    # The initial locations of initial packs respect the (possible) overlapping metric.
    pack_lifetimes = initialise(θ, n_packs, T, loc_dist, om);
    while t < T
        # Update the time and the number of packs at this time point.
        push!(time, t); push!(N_evolution, n_packs);

        # Update the birth and death rates. (can be a function later)
        tilde_birth = λ_birth + λ_birth0;
        tilde_death = λ_death;

        # Compute the overall rate and decide when the next event occurs.
        lambda_overall = tilde_birth + n_packs * tilde_death;
        t = t + rand(Exponential(1.0 / lambda_overall));
        t > T && break; # Break out if over T.

        # Decide whether a pack is born or dies.
        p_newborn = tilde_birth / lambda_overall;
        isborn = rand() <= p_newborn;
        if isborn # A new pack is born.
            # Get the packs alive at time t and sample a new pack location
            # respecting the overlap metric.
            alive_packs = packs_alive(pack_lifetimes, t);
            loc = sample_nonoverlapping(θ, alive_packs, loc_dist, om)

            # Add the pack to the list of WolfpackLifetime objects and the
            # set of packs alive.
            push!(pack_lifetimes, WolfpackLifetime(t, T, loc));
            push!(alive, length(pack_lifetimes));

        else # An existing pack dies.
            # Sample the pack number that dies.
            died = rand(alive);
            # Remove pack from the set of alive packs.
            pop!(alive, died);
            # Update the death time of the pack to the pack lifetimes.
            pack_lifetimes[died].death_time = t;
        end
        # Update the number of packs.
        n_packs = n_packs + 2 * isborn - 1;
    end
    # Build a WolfpackHistory object from the lifetimes.
    WolfpackHistory(0.0, T, pack_lifetimes);
end

"""
Draw event times from a homogeneous Poisson process with intensity λ between
`t0` and `T`.

Arguments:
- `times`: a vector of times where the drawn event times are stored.
- `λ`: the intensity of the process on the interval [t0, T).
- `t0`: the start time.
- `T`: the end time. `T` must be > `t0` (not checked).
"""
function sim_hpp!(times::Vector{<: AFloat}, λ::Real, t0::Real, T::Real)
    t = t0;
    incr = rand(Exponential(inv(λ)));
    while t + incr < T
        t = t + incr;
        push!(times, t);
        incr = rand(Exponential(1.0 / λ));
    end
    nothing;
end

"""
Simulate an N-D inhomogeneous Poisson process by thinning an N-D Poisson process.
Arguments:

* `bounds`: The bounds for the hypercube where to simulate.

* `λ_max`: An upper bound for the intensity of the IPP.

* `λ`: A function object taking an N-D object and returning the intensity at
that point. The function must take values less than or equal to `λ_max`.
"""
function sim_hpp(bounds::NTuple{N, Tuple{Float64, Float64}}, λ_max::Real, λ) where N
    @assert length(bounds) > 0 "`bounds` must have length greater than 0.";
    p = 1.0;
    for b in bounds
        p *= (b[2] - b[1]);
    end
    tot_λ = λ_max * p;

    M = rand(Poisson(tot_λ));
    x = zeros(Float64, N);
    out = Vector{NTuple{N, Float64}}(undef, 0);
    dist = product_distribution([map(b -> Uniform(b[1], b[2]), bounds)...]);
    for i in 1:M
        rand!(dist, x);
        λx = λ(x);
        prob = λx / λ_max;
        if rand() <= prob
            push!(out, NTuple{N, Float64}(x));
        end
    end
    out;
end


"""
Simulate a non-homogeneous Poisson process with a piecewise constant intensity
function. This amounts to simulating a homogeneous Poisson process on the
constant intervals.
"""
function sim_nhpp(pc::PiecewiseConstant)
    times = Float64[];
    for i in 2:length(pc.t)
        sim_hpp!(times, pc.y[i - 1], pc.t[i - 1], pc.t[i]);
    end
    times;
end

function draw_event(λ::Real, start::Real)
    incr = rand(Exponential(inv(λ)));
    start + incr;
end

"""
Simulate the latent state and (discretised) observations in one function call.

Arguments:
- `θ`: The parameters of the model.
- `N_init`: The number of packs initially.
- `T`: The end time of the time interval.

Optional:
- `delta`: value used in discretising step.
- `om` : an optional overlapping metric.
"""
function sim_latent_and_obs(θ::WolftrackParameters, N_init::Integer, T::Real,
                            loc_dist::MultivariateDistribution;
                            delta::AFloat = 1.0,
                            om::OverlapMetric = NoOverlap())

    wph = sim_latent(θ, N_init, T, loc_dist; om = om);
    times, locs, types = sim_wp_obs(θ, wph, loc_dist);
    wph, discretiser(times, locs, T; delta = delta);
end

"""
The function discretizes a set of observations in the following steps:
1. The observations are binned to bins of length `delta`.
2. For each bin:
    If there are any observations, the observations are shuffled and
    placed equidistantly within the bin.

Arguments:

* `times` : a vector containing the time points of the observations.

* `locations` : a vector containing the observations' locations.

* `T`: the end time of the time horizon.

Optional arguments:

* `multiplicities`: `nothing` or a vector of integers of length `length(times)`.
If not `nothing`, it is assumed that the corresponding observations have a multiplicity,
i.e in the processed output, there will be multiple observations at the same time
point, with time intervals between them set to zero.

* `delta`: the maximum discretisation interval.

* `shuffle`: Should the observations be shuffled per bin? Default is true.
"""
function discretiser(times::AVec{<: Real}, locations::AVec{<: AVec{<: Real}},
                     T::Real; multiplicities::Union{Vector{Int}, Nothing} = nothing,
                     delta::Real = 1.0, shuffle::Bool = true)
    @assert delta > 0.0 "`delta` must be positive.";
    @assert T > 0.0 "`T` must be positive.";
    @assert issorted(times) "the input times must be increasing.";
    if multiplicities != nothing
        msg = string("`multiplicities` must be of length `length(times)`.");
        @assert length(multiplicities) == length(times) msg;
    end

    # Create the output vector.
    disc_obs = Observation{length(locations[1])}[];
    sizehint!(disc_obs, length(locations));

    # Compute the number of bins.
    num_bins = convert(Int, ceil(T / delta));

    # Create a vector bin data.
    # Each element correspond to a bin. Each bin datum contains the vector of
    # observations in that bin, and their corresponding multiplicities.
    # The lower bound (in units of time) is also recorded.
    bins = map(1:num_bins) do i
        lower = (i - 1) * delta;
        upper = i * delta > T ? T : i * delta;
        indices = findall(t -> lower < t <= upper, times);
        if multiplicities == nothing
            zipped = collect(zip(locations[indices], ones(Int, length(indices))));
        else
            zipped = collect(zip(locations[indices], multiplicities[indices]));
        end
        zipped, lower;
    end

    # Process each bin.
    for b in bins
        locs = b[1];
        lower = b[2];

        # Get the length of the locations vector. If 0, then there are no observations
        # in the current bin.
        l = length(locs);
        if l == 0
            push!(disc_obs, WolfpackMisObs(lower + delta, delta));
        else # There is at least one observation.
            # Shuffle and place observations equidistantly.
            delta_t = delta / l;
            shuffle && shuffle!(locs);
            for i in 1:length(locs)
                o = locs[i][1];
                mult = locs[i][2];
                push!(disc_obs, WolfpackObs(o, lower + i * delta_t, delta_t));

                # Handle multiplicities.
                for j in 1:(mult - 1)
                    push!(disc_obs, WolfpackObs(o, lower + i * delta_t, 0.0));
                end
            end
        end
    end
    disc_obs;
end

"""
Generate a history of wolfpacks according to the model.
In this version, the packs do not move, and their initial location is drawn
from some distribution.

The total birth intensity at time t is given by: λ_b0 + λ_birth * bfun(t) * n_t,
where n_t is the amount of wolfpacks at time t. The "per pack" intensity is
therefore λ_birth * bfun(t).

Similarly, the "per pack" death intensity at time t is given by:
λ_death * dfun(t) * n_t, with the restriction that a death cannot occur
if there are no packs.

Arguments:
`θ`: The parameters of the model.
`N_init`: The number of initial packs.
`T`: The end time of the simulation.
`bfun`: A piecewise constant covariate function appearing in the total birth rate.
`dfun`: A piecewise constant covariate function appearing in the total death rate.
`loc_dist`: The location distribution for new packs.

Optional arguments:
`om`: Overlapping metric.
"""
function sim_latent_dynamic_nhpp(θ, N_init::Integer,
                                 bfun::PiecewiseConstant,
                                 dfun::PiecewiseConstant,
                                 loc_dist::MultivariateDistribution;
                                 om::OverlapMetric = NoOverlap())
    @assert N_init >= 0 "`N_init` must be >= 0.";
    @assert bfun.t[begin] == 0.0 "the first interval of `bfun` should begin at 0.0.";
    @assert domain(bfun) == domain(dfun) "domains of the covariates must match";

    # The vector `t_knots` gives the intervals at which both `bfun` and `dfun`
    # are constant.
    t_knots = constant_intervals(bfun, dfun);

    # Unpack parameters.
    λ_birth = θ.λ_birth;
    λ_birth0 = θ.λ_birth0;
    λ_death = θ.λ_death;

    t = 0.0; # Start time.
    T = domain(dfun)[2]; #t_knots[end]; # The end time.
    time = Float64[]; # Vector keeping sampled time points.
    N_evolution = Int[]; # Vector keeping the amount of packs at each time point.
    alive = Set(1:N_init); # The set of packs that are currently alive.
    next_i_knot = 2; # Index of next "knot" of piecewise constant function.

    # Initialise the vector of wolfpack lifetimes.
    # The initial locations of initial packs respect the (possible) overlapping metric.
    pack_lifetimes = initialise(θ, length(alive), T, loc_dist, om);

    while t < T
        next_t_knot = t_knots[next_i_knot];

        # Compute total λ's. Note that these will vary with n and have
        # to be recomputed at every step.
        λ_birth_tot = λ_birth0 + length(alive) * λ_birth * bfun(t);
        λ_death_tot = length(alive) * λ_death * dfun(t);
        λ_tot = λ_birth_tot + λ_death_tot;

        # Time increment.
        t_incr = rand(Exponential(inv(λ_tot)));
        tnew = t + t_incr; # Something possibly happens at this time.

        if tnew < next_t_knot # A new time was sampled before next change in covariate.

            # Decide whether a pack is born or dies.
            isborn = rand(Bernoulli(λ_birth_tot / λ_tot));

            # Handle birth and death.
            if isborn # A new pack is born.

                # Get the packs alive before the increment and sample a new pack location
                # respecting the overlap metric (if any).
                alive_packs = packs_alive(pack_lifetimes, t);
                loc = sample_nonoverlapping(θ, alive_packs, loc_dist, om)

                # Add the pack to the list of WolfpackLifetime objects and the
                # set of packs alive.
                push!(pack_lifetimes, WolfpackLifetime(tnew, T, loc));
                push!(alive, length(pack_lifetimes));

            else # An existing pack dies (if there are any)
                # This could be removed since n = 0 implies no death possible..
                if length(alive) > 0

                    # Sample the pack number that dies.
                    died = rand(alive);

                    # Remove the sampled pack from the set of alive packs.
                    pop!(alive, died);

                    # Update the death time of the pack to the pack lifetimes.
                    pack_lifetimes[died].death_time = tnew;
                end
            end
            # Increase t and save the time of the change and
            # the updated number of packs.
            t = tnew;
            push!(time, t); push!(N_evolution, length(alive));

        else # t + incr is more than `next_t_knot`.
            # This means we "jumped" over a time period where the total rate is constant,
            # i.e no more event times occured at that period.
            t = next_t_knot;
            next_i_knot += 1;
        end
    end
    # Build a WolfpackHistory object from the lifetimes.
    WolfpackHistory(0.0, T, pack_lifetimes);
end

function sim_N(nsim::Integer, θ::WolftrackParameters, N_init_dist,
               bfun::PiecewiseConstant, dfun::PiecewiseConstant,
               loc_dist::MultivariateDistribution;
               om::OverlapMetric = NoOverlap())
    map(1:nsim) do n
        N_init = rand(N_init_dist);
            sim = sim_latent_dynamic_nhpp(θ, N_init, bfun, dfun, loc_dist; om = om);
            evolution_of_N(sim);
    end
end

function sim_latent_dynamic_nhpp(θ::WolftrackParameters, N_init::Integer,
                                 covariates::IntensityCovariates{PWC, PWC, <: Any},
                                 loc_dist::MultivariateDistribution;
                                 om::OverlapMetric = NoOverlap())
    sim_latent_dynamic_nhpp(θ, N_init, covariates.birth, covariates.death,
                            loc_dist; om = om);
end

"""
The function simulates observations given a history of wolf packs.
The history of several wolfpacks is given as an argument and observations given
the lifetimes of the wolfpacks are returned.

The total observation intensity at time t is given by:
λ_clutter + λ_obs * ofun(t) * n_t,
where n_t is the amount of wolfpacks at time t. The "per pack" observation
intensity is therefore λ_obs * ofun(t).

Arguments:
- `θ`: the parameters.
- `wph`: an object of type WolfpackHistory containing the information
   about the packs (location, birth and death times).
- `ofun`: A piecewise constant function associated with the observation rate.
- `clutter_dist`: The distribution for clutter observations.
"""
function sim_obs_nhpp(θ::WolftrackParameters, wph::WolfpackHistory{N},
                      ofun::PiecewiseConstant,
                      clutter_dist::MultivariateDistribution) where N

    λ_obs = θ.λ_obs;
    λ_clutter = θ.λ_clutter;

    # `nevol_pc`: the number of packs as a piecewise function.
    # `λ_packs_tot_pc`: total obs rate from packs as a piecewise function.
    # `λ_tot_pc`: total obs rate as a piecewise function.
    nevol_pc = evolution_of_N(wph);
    λ_packs_tot_pc = λ_obs * ofun * nevol_pc;
    λ_tot_pc = λ_packs_tot_pc + λ_clutter;

    # Simulate the NHPP with the piecewise constant observation rate λ_tot_pc.
    event_times = sim_nhpp(λ_tot_pc);

    # Sample the type of each observation (clutter or real).
    obstype = Vector{Symbol}(undef, length(event_times));
    for (i, t) in enumerate(event_times)
        p_clutter = λ_clutter / λ_tot_pc(t);
        obstype[i] = rand() <= p_clutter ? :clutter : :real;
    end

    # Assign each observation to a pack or as clutter.
    obs_locs = [zeros(Float64, N) for i in 1:length(obstype)];
    for (i, o) in enumerate(obstype)
        if o == :real # Observation is a true observation.
            # Get packs that are alive, choose one randomly and sample.
            alive = packs_alive(wph.lifetimes, event_times[i]);
            winner = rand(alive);
            rand!(MvNormal(winner.location, Matrix(θ.Σ_obs)), obs_locs[i]);
        else # Observation is clutter.
            rand!(clutter_dist, obs_locs[i]);
        end
    end
    event_times, obs_locs, obstype;
end

function λ_obs_tot(t::Real, y1::Real, y2::Real, wph, θ, covariates)
    y = (y1, y2);
    λ_obs = θ.λ_obs;
    σobs = sqrt(θ.Σ_obs[1, 1]);
    λ_obs_t = covariates.obs_t(t);
    λ_obs_x = covariates.obs_x(y);
    λ_obs_x <= 0.0 && (return 0.0;);
    λ_obs_t <= 0.0 && (return 0.0;);
    tot = θ.λ_clutter * λ_obs_t * θ.inv_area * λ_obs_x;
    tot += λ_obs * λ_obs_x * λ_obs_t * total_pack_intensity(wph, t, y, σobs);
    tot;
end

function λ_obs_tot(x, wph, θ, covariates)
    λ_obs_tot(x[1], x[2], x[3], wph, θ, covariates);
end

function total_pack_intensity(wph::WolfpackHistory{2}, u::Real,
                              y, σobs::Real)
    tot = 0.0;
    for pack in wph.lifetimes
        if pack.birth_time <= u < pack.death_time
            tot += pdf(Normal(pack.location[1], σobs), y[1]) *
                   pdf(Normal(pack.location[2], σobs), y[2]);
        end
    end
    tot;
end

function log_total_pack_intensity(wph::WolfpackHistory{2}, u::Real,
                                  y, σobs::Real)
    tmp = zeros(length(wph.lifetimes));
    tmp .= -Inf;
    for (i, pack) in enumerate(wph.lifetimes)
        if pack.birth_time <= u < pack.death_time
            tmp[i] = logpdf(Normal(pack.location[1], σobs), y[1]) +
                   logpdf(Normal(pack.location[2], σobs), y[2]);
        end
    end
    logsumexp(tmp);
end

function log_λ_obs_tot(t::Real, y1::Real, y2::Real, wph, θ, covariates)
    tmp = zeros(2);
    y = (y1, y2);
    λ_obs = θ.λ_obs;
    σobs = sqrt(θ.Σ_obs[1, 1]);
    λ_obs_t = covariates.obs_t(t);
    λ_obs_x = covariates.obs_x(y);
    λ_obs_x <= 0.0 && (return 0.0;);
    λ_obs_t <= 0.0 && (return 0.0;);
    tmp[1] = log(θ.λ_clutter) + log(λ_obs_t) + log(θ.inv_area) + log(λ_obs_x);
    tmp[2] = log(λ_obs) + log(λ_obs_x) + log(λ_obs_t) + log_total_pack_intensity(wph, t, y, σobs);
    logsumexp(tmp);
end

"""
Return an upper bound of the intensity function of the model.
"""
function λ_obs_tot_max(wph, θ, covariates)
    Nmax = maximum(evolution_of_N(wph).y);
    λ_obs_x_max = maximum(covariates.obs_x.r);
    λ_obs_t_max = maximum(covariates.obs_t.y);
    pdf_bound = inv(sqrt(4.0 * pi * pi * det(θ.Σ_obs)));
    λ_obs_x_max * λ_obs_t_max * (θ.λ_obs * Nmax * pdf_bound + θ.λ_clutter * θ.inv_area);
end


"""
An uniform distribution with a domain defined by a set of pixels with
midpoints given in `midpoints` and size of pixels given by `pixel_size`.
"""
struct UniformGridDistr <: MultivariateDistribution{Continuous}
    midpoints::Vector{NTuple{2, Float64}}
    pixel_size::Float64
    function UniformGridDistr(midpoints::AVec{NTuple{2, Float64}}, pixel_size::Real)
        @assert pixel_size > 0 "`pixel_size` must be > 0.";
        @assert length(midpoints) > 0 "no midpoints given";
        new(midpoints, pixel_size);
    end
end
function rand!(ugd::UniformGridDistr, x::AVec{Float64})
    midpoint = ugd.midpoints[rand(1:length(ugd.midpoints))];
    half_pixel_size = ugd.pixel_size / 2.0;
    l₁ = midpoint[1] - half_pixel_size;
    u₁ = midpoint[1] + half_pixel_size;
    l₂ = midpoint[2] - half_pixel_size;
    u₂ = midpoint[2] + half_pixel_size;

    x[1] = rand(Uniform(l₁, u₁));
    x[2] = rand(Uniform(l₂, u₂));
    nothing;
end

function obs_given_packs(wph, covariates, θ, clutter_dist = nothing)
    @assert isconstdiag(θ.Σ_obs) "the covariance in θ should be constant diagonal";
    σobs = sqrt(θ.Σ_obs[1, 1]);

    out = Vector{NTuple{3, Float64}}(); # Container for all observations.
    λ_obs = θ.λ_obs;
    λ_clutter = θ.λ_clutter;
    has_clutter = λ_clutter > 0.0;

    # All points where the time covariate should be re-evaluated, since intensity
    # might change because of time covariate or number of territories alive changes.
    changepoints = constant_intervals(covariates.obs_t, evolution_of_N(wph));
    λ_obs_x_max = maximum(covariates.obs_x.r); # Maximal spatial covariate function value.
    x = zeros(2); # Temporary for drawing from normal distributions.

    for i in 1:(length(changepoints) - 1)
        cp = changepoints[i]; next_cp = changepoints[i + 1];
        time_interval_len = next_cp - cp;
        λ_obs_t = covariates.obs_t(cp); # Current value of time covariate.
        λ_tot_pack = λ_obs * λ_obs_t * λ_obs_x_max * time_interval_len; # Upper bound intensity.
        λ_tot_clutter = λ_clutter * λ_obs_t * λ_obs_x_max * time_interval_len;
        packs = packs_alive(wph, cp); # Packs that are alive.

        # Simulate observations from packs.
        for pack in packs
            N = rand(Poisson(λ_tot_pack));
            for j in 1:N
                x[1] = rand(Normal(pack.location[1], σobs));
                x[2] = rand(Normal(pack.location[2], σobs));
                if rand() <= (covariates.obs_x(x) / λ_obs_x_max)
                    tup = (rand(Uniform(cp, next_cp)), x[1], x[2]);
                    push!(out, tup);
                end
            end
        end

        # Simulate clutter observations.
        if has_clutter
            N = rand(Poisson(λ_tot_clutter));
            for j in 1:N
                rand!(clutter_dist, x);
                if rand() <= (covariates.obs_x(x) / λ_obs_x_max)
                    tup = (rand(Uniform(cp, next_cp)), x[1], x[2]);
                    push!(out, tup);
                end
            end
        end
    end
    sort!(out); # Sort out (wrt 1 dimension = time)
    out;
end

"""
Simulate an N trajectory from the discrete birth-death model adapted for the
wolf data.
"""
function sim_wolf_birth_model_discrete(θ, steps, Δ, N_init)
    N = zeros(Int, steps);
    n = N_init;
    λ_birth = θ.λ_birth;
    λ_death = θ.λ_death;
    λ_birth0 = θ.λ_birth0;
    p = zeros(3);

    for t in 1:steps
        ## Compute probabilities for outcomes.
        p[1] = (1.0 - exp(-Δ * (λ_birth * n + λ_birth0))) * exp(-Δ * n * λ_death); # Birth probability.
        p[2] = exp(-Δ * (λ_birth * n + λ_birth0)) * (1.0 - exp(-Δ * n * λ_death)); # Total death probability.
        p[3] = 1.0 - p[1] - p[2]; # No event probability.
        dist = DiscreteNonParametric([1, -1, 0], p); # Distribution.

        # Increment n.
        N[t] = n = n + rand(dist);
    end
    N;
end

"""
Simulate an N trajectory from the continuous birth & death process adapted for
wolf data.
"""
function sim_wolf_birth_model_continuous(θ, T, Δ, N_init)
    covariates = IntensityCovariates(birth = PiecewiseConstant((0.0, nextfloat(T))),
                                     death = PiecewiseConstant((0.0, nextfloat(T))));
    loc_dist = product_distribution([Uniform(-1, 1), Uniform(-1, 1)]); # Doesn't matter.
    wph = sim_latent_dynamic_nhpp(θ, N_init, covariates.birth, covariates.death, loc_dist)
    evolution_of_N(wph).(1.0:Δ:T);
end
