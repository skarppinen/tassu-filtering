include("../../../config.jl");
include("resampling.jl"); # Resampling algorithms.
include("wolfpack-types.jl");
include("gating.jl");
include("covariates.jl");
include("Gaussians.jl");
using LinearAlgebra

"""
Assuming Σ_obs = σ_obs^2 * I, return σ_obs^2 such that `A` is the area of the
circle containing `α` of the probability mass of the distribution N(.; 0, Σ_obs).
"""
function sigmaobssq(α::Real, A::Real)
    -inv(2.0 * pi / A * log(1.0 - α));
end

"""
Assuming Σ_obs = σ_obs^2 * I, return σ_obs^2 such that `d` is the diameter of the
circle containing `α` of the probability mass of the distribution N(.; 0, Σ_obs).
"""
function sigmaobssq_diameter(α::Real, d::Real)
    v = d / (2.0 * sqrt(quantile(Chisq(2), α)));
    v * v;
end

function _check_initial_distr(N_init_dist::DiscreteDistribution, npar::Integer,
                              max_npacks::Integer)

    msg = string("the maximal value in the support of `N_init_dist` must be smaller ",
                 "or equal to `max_npacks`.");
    @assert maximum(support(N_init_dist)) <= max_npacks msg;
    nothing;
end

"""
Function returns the initial vector of pack counts and their weights.
It is assumed that `N_init_dist` has a support with a size less than or equal
to `npar`.
"""
function get_initial_n_and_weights(N_init_dist::DiscreteFiniteSupport,
                                   npar::Integer)
    msg = string("the length of the support of `N_init_dist` must not exceed ",
                 "`npar`.");
    supp = support(N_init_dist);
    @assert length(supp) <= npar msg;
    N = zeros(typeof(npar), npar);
    W = zeros(Float64, npar);

    for (i, s) in enumerate(supp)
        N[i] = s;
        W[i] = pdf(N_init_dist, s);
    end
    normalize!(view(W, 1:length(supp)), 1);
    N, W;
end

"""
A function to determine which wolfpack type is appropriate for the given
dimension and the type of the Σ_obs matrix. If this matrix is of dimension 2 and
constant diagonal (a typical case), an optimised type for the wolfpacks
can be used internally.
"""
function determine_wolfpacktype(dimension::Integer, constdiag::Bool)
    if dimension == 2 && constdiag
        WPType = WolfpackConstDiag{2, 3};
    else
        WPType = Wolfpack{dimension, dimension * dimension};
    end
    WPType;
end

function flip(x::T, fst::T, snd::T) where T
    x == fst ? snd : fst;
end

function get_resampling(resampling::Type{<: Resampling}, npar::Integer, max_npacks::Integer)
    @assert npar > 0 && max_npacks > 0 "inappropriate value for `npar` or `max_npacks`. these must be positive";
    resampling <: StandardResampling && return resampling();
    resampling(npar, max_npacks);
end

"""
Initialise filtering state when initial distribution is Uniform for pack locations, but
a prior is used for the number of packs.
"""
function initialise_filter_state(N_init_dist::DiscreteFiniteSupport, npar::Integer,
                                 θ; n_threads::Integer = 1, max_npacks::Integer = 100)
    N_init_vec, W_init_vec = get_initial_n_and_weights(N_init_dist, npar);
    WolfpackFilterState(N_init_vec, W_init_vec, θ; n_threads = n_threads,
                        max_npacks = max_npacks);
end

"""
Initialise filtering state when there is initial knowledge of the location of the packs.
"""
function initialise_filter_state(locs::AVec{<: AVec{<: AFloat}}, npar::Integer,
                                 θ; n_threads::Integer = 1, max_npacks::Integer = 100)
    Nvec = zeros(Int, npar);
    Wvec = zeros(Float64, npar);
    Nvec[1] = length(locs);
    Wvec[1] = 1.0;
    σobssq = θ.Σ_obs[1, 1];
    fs = WolfpackFilterState(Nvec, Wvec, θ; n_threads = n_threads, max_npacks = max_npacks);
    fi = fs.fi[];
    particle = fs.X[1, fi];
    for i in 1:particle.N[]
        pack = particle.packs[i];
        pack.data[1] = locs[i][1];
        pack.data[2] = locs[i][2];
        pack.data[3] = σobssq;
        pack.newborn[] = false;
    end
    fs;
end

"""
Run the wolfpack filter and save filtering distributions at timepoints in `save`.
The function constructs all the necessary storage objects and runs the filter once.
The output of the function is an object of type `WolfpackFilterOutput`, which
contains the filtered distributions and the ancestrial lineages of the particles.

Mandatory arguments:
`data`: A vector of wolfpack observations. Each observation is either of type <: WolfpackObs,
i.e they consist of a time, delta and a location, or of type WolfpackMisObs,
i.e they only contain a time and a delta value.

`θ`: The parameter values controlling how the model functions. See the struct
`WolftrackParameters` for further details.

`init`: The initial distribution of the amount of wolf packs or a vector of pack locations.
If a distribution, the distribution can be any discrete distribution defined by the
`Distributions.jl` package, with finite support. Note that the highest value in the support of
the distribution must be less than or equal to `max_npacks` (see argument `max_npacks`) and
`npar` must be greater or equal to the size of the distributions's support.
These conditions are checked when the function is invoked.
In this case the locations of each pack are initially uniformly distributed and the
initial weights are computed based on the pdf values of the initial distribution.
If `init` is a vector of locations, the initial state of the model consist of a
single particle with weight 1.0 (and the rest with weight 0.0). The particle with
weight 1.0 gets the locations `init` as the initial pack locations and the pack size
parameter determines the initial uncertainty of each pack.

`save`: A non-empty vector of positive floating point values that specify time instants
at which the filtering distribution should be saved and returned.

Optional keyword arguments:
`covariates`: An object of type `IntensityCovariates` that can be used to specify
time-varying covariate functions for the birth, death and observation intensity.
By default, no covariate functions are used.

`gating`: An object of type <: Gating specifying which gating criterion should be
used to reduce computation and speed up the filter. Possibilites are an object
of type `NoGating` (default) or and object of type `ChiSquareGating`, which uses
the Mahalanobis distances between the observations and the packs to reduce
the number of likelihood computations.

`npar`: The number of particles to use in the filter. The default is 256.
More particles mean more accurate results, but increased computation time.
The number of particles needed depends greatly on the type of resampling used.

`resampling`: A subtype of `Resampling`, i.e the possibilites are: `OptimalResampling`,
`MultinomialResampling`, `KillingResampling`, `StratifiedResampling` or `SystematicResampling`.
The default and suggested is `OptimalResampling`, which is the best performing resampling algorithm
for the model. The other resamplings require less computation, but need a substantially
higher number of particles to work well. The optimal resampling guarantees distinctness
of the particles during filtering.

`n_threads`: An integer specifying how many threads should be used internally in
the filter computations. The default is 1. Numbers <= 1 specify no threading.
For values greater or equal to 2, the number of threads is set to
min(`n_threads`, *threads available on machine*). More threads
mean more speed, but the speed gains per core depend on the number
of particles used, number of packs simulated during filtering, and the type of
resampling used.

`max_npacks`: An integer specifying the maximal number of packs per particle.
The filter will terminate if the number of packs exceeds this number in any of
the particles during filtering. The default, (and minimum) value is 100 packs,
which should be enough for the current application. Note that this value will also
interact with the argument `init`. Increasing this number will increase
the memory consumption of the filter, especially when `resampling = OptimalResampling`.
"""
function run_wolfpack_filter(data::AVec{<: Observation{N}},
                             θ::WolftrackParameters{N},
                             init,
                             save::AVec{<: AFloat};
                             covariates::IntensityCovariates = IntensityCovariates(),
                             gating::Gating = NoGating(),
                             npar::Int = 256,
                             resampling::Type{<: Resampling} = OptimalResampling,
                             n_threads::Int = 1,
                             max_npacks::Int = 100) where N
    @assert !isempty(save) "the vector of saved timepoints, `save`, must not be empty.";
    @assert npar > 0 "the number of particles, `npar`, must be positive";
    @assert max_npacks >= 100 "the maximum number of packs, `max_npacks` should be >= 100";

    if init isa DiscreteFiniteSupport
        _check_initial_distr(init, npar, max_npacks);
    else
        @assert length(init) > 0 "`init` must not be an empty vector.";
    end

    # Make sure data has all the necessary timepoints. (in `save`).
    data = add_timepoints(data, save);

    # Construct resampling object.
    resampling = get_resampling(resampling, npar, max_npacks);

    # Construct state and output objects.
    fs = initialise_filter_state(init, npar, θ; n_threads = n_threads, max_npacks = max_npacks);
    fo = WolfpackFilterOutput{wolfpacktype(fs)}(npar, save;
                                                n_packs = ceil(Int, max_npacks / 2));

    # Do filtering.
    _collecting_filter!(fo, fs, data, θ, covariates, gating, resampling);
    fo;
end

function _collecting_filter!(fo::WolfpackFilterOutput, fs::WolfpackFilterState{WPType, N},
                             data::AVec{<: Observation{N}}, θ::WolftrackParameters{N},
                             covariates::IntensityCovariates, gating::Gating,
                             resampling::Resampling) where {WPType, N}
    npar = get_npar(fs);
    a = collect(1:npar); # Vector giving ancestors between saved timepoints.
    atmp = copy(a); # Helper vector for keeping ancestors.
    _init!(fs); time_i = 1;

    for o in data
        # Filter the next observation.
        @inbounds _wolfpack_filter_step!(fs, o, covariates, gating, θ, resampling);

        # Update ancestors.
        for i in 1:npar
            @inbounds a[i] = atmp[fs.A[i]];
        end
        atmp .= a;

        # Save time point (if requested).
        time_i = _save_time_point!(fo, fs, o, time_i, (a, atmp));
    end
    nothing;
end

function _save_time_point!(fo::WolfpackFilterOutput, fs::WolfpackFilterState,
                           o::Observation, time_i::Integer, atmps)
    @inbounds next_time = fo.time[time_i];
    otime = timeof(o)::Float64;
    if isapprox(otime, next_time)
        _save_filtering_distribution!(fo, fs, time_i);

        # Only save ancestry after we have recorded the first filtering
        # distribution of interest.
        if time_i > 1
            _save_ancestry!(fo, atmps[1], time_i);
        end

        # Reset ancestor vectors (since we just saved a distribution)
        # and increment time_i.
        atmps[1] .= atmps[2] .= 1:get_npar(fs); time_i += 1;
    end
    time_i;
end

function _save_filtering_distribution!(fo::WolfpackFilterOutput,
                                       fs::WolfpackFilterState,
                                       j::Integer)
    fi = fs.fi[];
    for i in 1:get_npar(fs)
        @inbounds copy!(fo.X[i, j], fs.X[i, fi]);
        @inbounds fo.W[i, j] = fs.W[i];
    end
    nothing;
end

function _save_filtering_distribution!(fo::WolfpackFilterOutput,
                                       fs::WolfpackFilterState{WPType, N, NT},
                                       j::Integer) where {WPType, N, NT}
    fi = fs.fi[];
    Threads.@threads for i in 1:get_npar(fs)
        @inbounds copy!(fo.X[i, j], fs.X[i, fi]);
        @inbounds fo.W[i, j] = fs.W[i];
    end
    nothing;
end
function _save_ancestry!(fo::WolfpackFilterOutput, a::AVec{<: Integer},
                         j::Integer)
    for i in 1:get_npar(fo)
        @inbounds fo.A[i, j - 1] = a[i];
    end
    nothing;
end

function _init!(fs::WolfpackFilterState)
  fs.V .= zero(Float64);
  for i in 1:length(fs.A)
      @inbounds fs.A[i] = i;
  end
  nothing;
end

"""
Filter one observation with a standard resampling algorithm without multithreading.
"""
function _wolfpack_filter_step!(fs::WolfpackFilterState{WPType, N, 1},
                                o::Observation{N},
                                covariates::IntensityCovariates,
                                gating::Gating,
                                θ::WolftrackParameters{N},
                                resampling::StandardResampling) where {WPType, N}

    # Resample.
    resample!(fs.A, fs.W, resampling);
    fi = fs.fi[]; ri = flip(fi, 1, 2);
    for (i, k) in enumerate(fs.A)
        @inbounds copy!(fs.X[i, ri], fs.X[k, fi]);
    end

    # Compute weights and update particles.
    @inbounds pds = fs.pds[1];
    for j in 1:get_npar(fs)
        # Reference to the current particle.
        @inbounds particle = fs.X[j, ri];

        # Build the proposal distribution table and sample one outcome of
        # births, deaths and data associations. Compute also the particle weight.
        w, bdc = sample_latent!(o, particle, pds, covariates, gating, θ);
        @inbounds fs.W[j] = w; # Assign the weight to the weight vector.

        # Modify the particle based on the sampled birth, death and
        # association. (b, d and c)
        update!(particle, bdc, o, θ);
    end
    fs.fi[] = ri; # Swap column of current filtering distribution.

    # Normalise weights and catch ingredients to the loglik approximation.
    fs.V .= fs.V .+ normalise_logweights!(fs.W);
    nothing;
end

"""
Filter one observation with a standard resampling algorithm and multithreading.
"""
function _wolfpack_filter_step!(fs::WolfpackFilterState{WPType, N, NT},
                                o::Observation{N},
                                covariates::IntensityCovariates,
                                gating::Gating,
                                θ::WolftrackParameters{N},
                                resampling::StandardResampling) where {WPType, N, NT}

    # Resample.
    resample!(fs.A, fs.W, resampling);
    fi = fs.fi[]; ri = flip(fi, 1, 2);
    Threads.@threads for i in 1:length(fs.A)
        @inbounds copy!(fs.X[i, ri], fs.X[fs.A[i], fi]);
    end

    npar = get_npar(fs);
    Threads.@threads for j in 1:npar
        pds = fs.pds[Threads.threadid()];
        rng = Random.THREAD_RNGs[Threads.threadid()];

        # Reference to the current particle.
        @inbounds particle = fs.X[j, ri];

        # Build the proposal distribution table and sample one outcome of
        # births, deaths and data associations. Compute also the particle weight.
        w, bdc = _rng_sample_latent!(rng, o, particle, pds, covariates, gating, θ);
        @inbounds fs.W[j] = w; # Assign the weight to the weight vector.

        # Modify the existing state based on the births and deaths and
        # associations that occured. (b, d and c)
        update!(particle, bdc, o, θ);
    end
    fs.fi[] = ri; # Swap reference to current filtering distribution.

    # Normalise weights and catch ingredients to the loglik approximation.
    fs.V .= fs.V .+ normalise_logweights!(fs.W);
    nothing;
end

@inline function writeto!(X::AVec{T}, start_i::Integer, v::AArr{T}) where T
    copyto!(X, start_i, v, 1, length(v));
end
@inline function prob_table_elems(n::Integer, ::Type{WolfpackMisObs})
    n + 2;
end
@inline function prob_table_elems(n::Integer, ::Type{<: WolfpackObs})
    a = prob_table_elems(n, WolfpackMisObs);
    a * a;
end
@inline function prob_table_elems(particle::WolfpackParticle, ::Type{WolfpackMisObs})
    prob_table_elems(particle.N[], WolfpackMisObs);
end
@inline function prob_table_elems(particle::WolfpackParticle, ::Type{<: WolfpackObs})
    prob_table_elems(particle.N[], WolfpackObs{2});
end
@inline function prob_table_size(particle::WolfpackParticle, ::Type{WolfpackMisObs})
    (prob_table_elems(particle, WolfpackMisObs),);
end
@inline function prob_table_size(particle::WolfpackParticle, ::Type{<: WolfpackObs})
    a = prob_table_elems(particle, WolfpackMisObs);
    a, a;
end
@inline function prob_table_shape(particle::WolfpackParticle, ::Type{WolfpackMisObs})
    (prob_table_elems(particle, WolfpackMisObs), 1);
end
@inline function prob_table_shape(particle::WolfpackParticle, obstype::Type{<: WolfpackObs})
    prob_table_size(particle, obstype);
end

"""
Filter one observation with the optimal resampling, no multithreading.
"""
function _wolfpack_filter_step!(fs::WolfpackFilterState{WPType, N, 1},
                                o::Observation,
                                covariates::IntensityCovariates,
                                gating::Gating,
                                θ::WolftrackParameters,
                                resampling::OptimalResampling) where {WPType, N}
    obstype = typeof(o);
    fi = fs.fi[]; ri = flip(fi, 1, 2);
    pds = @inbounds fs.pds[1];
    nparticles = get_npar(fs);

    # Compute write ranges based on the observation type.
    # These specify the range to which each particle can write in the
    # probability vector `resampling.q_max`. Obtain views given the current pack sizes.
    set_ranges!(resampling.wr, fs, obstype);
    q = @inbounds view(resampling.q_max, 1:resampling.wr.hi[end]);
    qA = view(resampling.qA_max, 1:length(q)); qA .= 1:length(q);

    # Populate the probability table for each particle and write each probability
    # table to the vector `q` to the indices specified by `wr`.
    for i in 1:nparticles
        @inbounds particle = fs.X[i, fi]; # Reference to current particle.
        npacks = particle.N[];
        check_packsize(npacks, pds);

        # Compute the probability table for particle i.
        compute_association_likelihoods!(pds, particle, o, gating, θ);
        populate_probability_table!(pds, particle, o, covariates, θ);

        # Write the probability table of particle i to the appropriate indices in the
        # vector `q`.
        pt_shape = prob_table_shape(particle, obstype);
        v_pt = view(pds.prob_table, 1:pt_shape[1], 1:pt_shape[2]);
        v_pt .+= log(fs.W[i]);

        @inbounds writeto!(q, resampling.wr.lo[i], v_pt);
    end

    # Find 1/c and resample.
    normalise_logweights!(q);
    inv_c, L = find_inv_c!(q, qA, nparticles);
    _optimal_resample!(fs.W, fs.A, q, qA, inv_c, L);

    # For each resampling outcome, find the particle index and outcome that happened.
    for i in 1:nparticles
        @inbounds r, elem = get_range_and_elem_index(resampling.wr, fs.A[i]); # Particle index and outcome.
        @inbounds fs.A[i] = r; # Record ancestor particle index.
        @inbounds dest = fs.X[i, ri];
        @inbounds src = fs.X[r, fi]; # The particle we need to read from.
        copy!(dest, src);
        bdc = get_sampling_outcome(src, obstype, elem);
        update!(dest, bdc, o, θ);
    end
    fs.fi[] = ri; # Swap reference to current filtering distribution.
    nothing;
end


"""
Filter one observation with the optimal resampling and multithreading.
"""
function _wolfpack_filter_step!(fs::WolfpackFilterState{WPType, N, NT},
                                o::Observation{N},
                                covariates::IntensityCovariates,
                                gating::Gating,
                                θ::WolftrackParameters,
                                resampling::OptimalResampling) where {WPType, N, NT}
    obstype = typeof(o);
    fi = fs.fi[]; ri = flip(fi, 1, 2);
    nparticles = get_npar(fs);

    # Compute write ranges based on the observation type.
    # These specify the range to which each particle can write in the
    # probability vector `resampling.q_max`. Obtain views given the current pack sizes.
    set_ranges!(resampling.wr, fs, obstype);
    q = @inbounds view(resampling.q_max, 1:resampling.wr.hi[end]);
    qA = view(resampling.qA_max, 1:length(q)); qA .= 1:length(q);

    # Populate the probability table for each particle and write each probability
    # table to the vector `q` to the indices specified by `wr`.
    Threads.@threads for i in 1:nparticles
        pds = fs.pds[Threads.threadid()];
        @inbounds particle = fs.X[i, fi]; # Reference to current particle.

        # Compute the probability table for particle i.
        compute_association_likelihoods!(pds, particle, o, gating, θ);
        populate_probability_table!(pds, particle, o, covariates, θ);

        # Write the probability table of particle i to the appropriate indices in the
        # vector `q`.
        pt_shape = prob_table_shape(particle, obstype);
        v_pt = view(pds.prob_table, 1:pt_shape[1], 1:pt_shape[2]);
        v_pt .+= log(fs.W[i]);
        @inbounds writeto!(q, resampling.wr.lo[i], v_pt);
    end

    # Find 1/c and resample.
    normalise_logweights!(q);
    inv_c, L = find_inv_c!(q, qA, nparticles);
    _optimal_resample!(fs.W, fs.A, q, qA, inv_c, L);

    # For each resampling outcome, find the particle index and outcome what happened.
    Threads.@threads for i in 1:nparticles
        @inbounds r, elem = get_range_and_elem_index(resampling.wr, fs.A[i]); # Particle index and outcome.
        @inbounds fs.A[i] = r; # Record ancestor particle index.
        @inbounds dest = fs.X[i, ri];
        @inbounds src = fs.X[r, fi]; # The particle we need to read from.
        copy!(dest, src);
        bdc = get_sampling_outcome(src, obstype, elem);
        update!(dest, bdc, o, θ);
    end
    fs.fi[] = ri; # Swap reference to current filtering distribution.
    nothing;
end

@inline function sample_latent!(o::Observation, particle::WolfpackParticle,
                                pds::ProposalDistStorage, covariates::IntensityCovariates,
                                gating::Gating, θ::WolftrackParameters)
    _rng_sample_latent!(Random.GLOBAL_RNG, o, particle, pds, covariates, gating, θ);
end



"""
Function populates the preallocated loglikelihood vector `pds.lltmp` with the loglikelihoods
resulting when each of the packs in the particle `particle` is updated with the observation
`o`. The Kalman updated means are also computed alongside, since they are required in
the approximations related to the spatial covariate. The loglikelihood of the association
with a newborn pack and clutter are also computed. In case of a newborn pack,
the Kalman mean is the observation. The object `gating` can influence the computation of the association
loglikelihoods of the packs. If gating kicks in, the temporary vector of Kalman mean
for the corresponding pack does not change, since association with that pack will anyway
be impossible at this iteration.
"""
function compute_association_likelihoods!(pds::ProposalDistStorage, particle::WolfpackParticle,
                                          o::WolfpackObs, gating::Gating,
                                          θ::WolftrackParameters)
    npacks = particle.N[];
    lltmp = pds.lltmp;
    meantmp = pds.meantmp;
    for k in 1:npacks
        @inbounds pack = particle.packs[k];
        if pack.newborn[]
            # This is probably rare but might happen (i.e there is a pack among
            # already alive packs that has never been associated).
            @inbounds lltmp[k] = θ._log_inv_area;
            @inbounds copy!(meantmp[k], o.location);
        else
            # Here we return -Inf if gating kicks in. The consequence is that
            # many elements in the probability table will also be -Inf =>
            # possibilities will be cut.
            @inbounds lltmp[k] = gated_loglik!(meantmp[k], pack, o, gating, θ);
        end
    end
    # Loglikelihood for associating with a new pack and clutter. +
    # Kalman updated mean for new pack.
    @inbounds lltmp[npacks + 1] = θ._log_inv_area;
    @inbounds copy!(meantmp[npacks + 1], o.location);
    @inbounds lltmp[npacks + 2] = θ._log_inv_area;
    nothing;
end

"""
In case of a missing observation, no association likelihoods are required.
(this method is required with optimal resampling)
"""
@inline function compute_association_likelihoods!(pds::ProposalDistStorage, particle::WolfpackParticle,
                                                  o::WolfpackMisObs, gating::Gating,
                                                  θ::WolftrackParameters)
    nothing;
end

function check_packsize(n_packs::Integer, pds::ProposalDistStorage)
    if n_packs > (pds.size - 2)
        msg = string("maximal number of packs reached in one of the particles, increase the size ",
                     "of `max_n_packs` argument or adjust parameter values.");
        throw(ArgumentError(msg));
    end
    nothing;
end

"""
Sample birth, death and association in the case of no location observation.

Arguments:
`rng`: The RNG object to use.
`o`: An observation of type WolfpackMisObs.
`particle`: The particle the sampling is concerned with.
`pds`: Storage object keeping the probability table and related quantities.
`g`: Gating object. Ignored in this method.
`θ`: The parameters of the model.
"""
function _rng_sample_latent!(rng::AbstractRNG, o::WolfpackMisObs,
                             particle::WolfpackParticle, pds::ProposalDistStorage,
                             covariates::IntensityCovariates, gating::Gating,
                             θ::WolftrackParameters)
    # Populate table with log probabilities.
    n_packs = particle.N[];
    check_packsize(n_packs, pds);
    populate_probability_table!(pds, particle, o, covariates, θ);

    # Compute the log particle weight.
    # Sample birth, death and association.
    pt_elems = prob_table_elems(particle, typeof(o));
    table = view(pds.prob_table, 1:pt_elems);
    w = logsumexp(table); # Compute logweight.
    bdc = _rng_sample_bdc!(rng, table);
    w, bdc;
end

"""
Sample birth, death and association in the case of a location observation.

Arguments:
`o`: An observation of type <: WolfpackObs.
`particle`: The particle the sampling is concerned with.
`pds`: Storage object keeping the probability table and related quantities.
`covariates`: A covariate object keeping timevarying covariate functions for observation,
birth and death rates.
`gating`: A gating object determining how gating should be done.
`θ`: The parameters of the model.
"""
function _rng_sample_latent!(rng::AbstractRNG, o::WolfpackObs, particle::WolfpackParticle,
                             pds::ProposalDistStorage, covariates::IntensityCovariates,
                             gating::Gating, θ::WolftrackParameters)
    n_packs = particle.N[];
    check_packsize(n_packs, pds);

    # Precompute likelihoods for associating o.
    compute_association_likelihoods!(pds, particle, o, gating, θ);

    # Populate table with log probabilities.
    populate_probability_table!(pds, particle, o, covariates, θ);

    # Compute the log particle weight.
    # Sample birth, death and association.
    pt_shape = prob_table_shape(particle, typeof(o));
    @inbounds table = view(pds.prob_table, 1:pt_shape[1], 1:pt_shape[2]);
    w = logsumexp(table); # Compute logweight. This is one of the costly operations.
    bdc = _rng_sample_bdc!(rng, table);
    w, bdc;
end

"""
Populate the probability table in the object `pds`, for birth, death and association in
the case of a location observation.
The table is organised such that the rows 1, .., n of the table
correspond to associating the observation with the pack that has the corresponding
indice. The row index n + 1 stands for a newborn pack and row index n + 2 stands
for a clutter association.
The first column stands for the case that no births and deaths occur. (under different
associations)
The second column stands for the case that a birth occurs and the remaining columns
stand for the case that one of the packs dies. (under different associations,
respectively)

Arguments:
`pds`: Storage object keeping the probability table and related quantities.
`particle`: The particle.
`o`: An observation of type <: WolfpackObs.
`covariates`: A covariate object keeping timevarying covariate functions for observation,
birth and death rates.
`θ`: Parameters of the model.
"""
function populate_probability_table!(pds::ProposalDistStorage, particle::WolfpackParticle,
                                     o::WolfpackObs, covariates::IntensityCovariates,
                                     θ::WolftrackParameters)
    n_prev = particle.N[];
    table = pds.prob_table;
    dt = o.dt;

    # Birth and death probability related values.
    tbd = o.time - dt; # The time to compute birth rate and death rate.
    tot_λ_birth = n_prev * θ.λ_birth * covariates.birth(tbd) + θ.λ_birth0;
    tot_λ_death = n_prev * θ.λ_death * covariates.death(tbd);
    log_birth = log_birth_prob(tot_λ_birth, tot_λ_death, dt);
    log_death = log_death_prob(tot_λ_birth, tot_λ_death, dt, n_prev);
    log_nobirth_nodeath = log(1.0 - n_prev * exp(log_death) - exp(log_birth));

    # Observation related values.
    λ_obs = θ.λ_obs;
    log_λ_obs = log(λ_obs);
    λ_clutter = θ.λ_clutter;
    log_λ_clutter = log(λ_clutter);
    tildeC = θ._tildeC;
    λ_obs_t = covariates.obs_t(o.time);
    log_λ_obs_t = log(λ_obs_t);
    λ_obs_x_yk = covariates.obs_x(o.location);
    log_λ_obs_x_yk = log(λ_obs_x_yk);
    premult = dt * λ_obs_t;
    premult_times_λ_obs = premult * λ_obs;

    # `log_tot_λ_pre` gives the approximate value for the product of integrals
    # in the case that no pack is associated. The association cases can be worked
    # out from it by subtracting and adding certain (single) terms.
    # These "correction terms" are stored here to `pds.λ_obs_x_assoc_tmp`.
    # The "pre-mean" log_λ_obs_x's are also stored since they are required
    # when computing probabilities for the death cases.
    log_tot_λ_pre = -premult * λ_clutter * tildeC;
    for i in 1:n_prev
       @inbounds pack = particle.packs[i];
       if pack.newborn[]
            @inbounds pds.λ_obs_x_pre_tmp[i] = tildeC;
            @inbounds pds.λ_obs_x_assoc_tmp[i] = premult_times_λ_obs * (tildeC - λ_obs_x_yk);
            log_tot_λ_pre -= premult_times_λ_obs * tildeC;
       else
            λ_obs_x = covariates.obs_x(ref_to_mean(pack));
            @inbounds λ_obs_x_kf = covariates.obs_x(pds.meantmp[i]);
            @inbounds pds.λ_obs_x_pre_tmp[i] = λ_obs_x;
            @inbounds pds.λ_obs_x_assoc_tmp[i] = premult_times_λ_obs * (λ_obs_x - λ_obs_x_kf);
            log_tot_λ_pre -= premult_times_λ_obs * λ_obs_x;
       end
    end
    # Correction term for associating new pack.
    @inbounds pds.λ_obs_x_assoc_tmp[n_prev + 1] = premult_times_λ_obs * (tildeC - λ_obs_x_yk);

    ## Table population starts here.

    # 1. column: case of nothing happens, next n equals n_prev.
    # NOTE: in lltmp it is taken into account if pack is newborn or not.
    nobirth_nodeath_common_terms = log_nobirth_nodeath + log_tot_λ_pre + log_λ_obs_t +
                                   log_λ_obs_x_yk;
    for c in 1:n_prev
        @inbounds table[c, 1] = nobirth_nodeath_common_terms + pds.λ_obs_x_assoc_tmp[c] +
                                log_λ_obs + pds.lltmp[c];
    end
    # Association with newborn.
    @inbounds table[n_prev + 1, 1] = -Inf; # Can't associate with new pack if one not born.
    # Association with clutter.
    @inbounds table[n_prev + 2, 1] = nobirth_nodeath_common_terms +
                                     log_λ_clutter +
                                     pds.lltmp[n_prev + 2];

    # Case of a birth, next n equals n_prev + 1.
    # NOTE: This case *should* be ok.
    birth_correction = -premult_times_λ_obs * tildeC; # Bc one more uniform pack should be in `log_tot_λ_pre`.
    birth_common_terms = log_birth + log_tot_λ_pre + log_λ_obs_t +
                         log_λ_obs_x_yk + birth_correction;
    for c in 1:n_prev
        @inbounds table[c, 2] = birth_common_terms + pds.λ_obs_x_assoc_tmp[c] +
                                log_λ_obs + pds.lltmp[c];
    end
    # Association with newborn. (same as existing newborn)
    @inbounds table[n_prev + 1, 2] = birth_common_terms +
                                     pds.λ_obs_x_assoc_tmp[n_prev + 1] +
                                     log_λ_obs +
                                     pds.lltmp[n_prev + 1];
    # Association with clutter.
    @inbounds table[n_prev + 2, 2] = birth_common_terms +
                                     log_λ_clutter +
                                     pds.lltmp[n_prev + 2];

    # Columns 3 - ...: case of death of pack i, next n equals n_prev - 1.
    for i in 3:(n_prev + 2)
        pack_i = i - 2; # Index of dying pack.
        # Bc one less pack should be in `log_tot_λ_pre`.
        death_correction = @inbounds premult_times_λ_obs * pds.λ_obs_x_pre_tmp[pack_i];
        death_common_terms = log_death + log_tot_λ_pre + death_correction +
                             log_λ_obs_t + log_λ_obs_x_yk;
        for c in 1:n_prev
            @inbounds table[c, i] = death_common_terms + pds.λ_obs_x_assoc_tmp[c] +
                                    log_λ_obs + pds.lltmp[c];
        end
        # Association with newborn.
        @inbounds table[n_prev + 1, i] = -Inf; # Can't associate with a new pack if one not born.
        # Association with clutter.
        @inbounds table[n_prev + 2, i] = death_common_terms +
                                         log_λ_clutter +
                                         pds.lltmp[n_prev + 2];
        @inbounds table[pack_i, i] = -Inf; # Can't associate with dying pack.
    end
    nothing;
end

"""
Populate the probability table in the object `pds`, for birth and death in
the case of no observation. In this case the association is not considered
(c = -1), since there is no observation to associate.
Only the first column vector of the probability table, `pds.prob_table`, is hence used here.
The indices of the column vector correspond to the following:
1 = probability that nothing happens.
2 = probability that a new pack is born.
3..N+2 = probabilities of deaths of packs 1, ..., N.

Arguments:
`pds`: Storage object keeping the probability table and related quantities.
`particle`: The particle.
`o`: An observation of type WolfpackMisObs.
`covariates`: A covariate object keeping covariate functions for observation,
birth and death rates.
`θ`: Parameters of the model.
"""
function populate_probability_table!(pds::ProposalDistStorage, particle::WolfpackParticle,
                                     o::WolfpackMisObs, covariates::IntensityCovariates,
                                     θ::WolftrackParameters)
    n_prev = particle.N[];
    table = pds.prob_table;
    dt = o.dt;

    # Birth and death probabilities.
    tbd = o.time - dt; # The time to compute birth rate and death rate.
    tot_λ_birth = n_prev * θ.λ_birth * covariates.birth(tbd) + θ.λ_birth0;
    tot_λ_death = n_prev * θ.λ_death * covariates.death(tbd);
    log_birth = log_birth_prob(tot_λ_birth, tot_λ_death, dt);
    log_death = log_death_prob(tot_λ_birth, tot_λ_death, dt, n_prev);
    log_nobirth_nodeath = log(1.0 - n_prev * exp(log_death) - exp(log_birth));

    # Observation related values.
    λ_obs = θ.λ_obs;
    λ_clutter = θ.λ_clutter;
    tildeC = θ._tildeC;
    λ_obs_t = covariates.obs_t(o.time);
    premult = -dt * λ_obs_t;

    # Compute the lambda_obs_x values at the pack means, and
    # the log total lambda for the packs at the previous time point.
    log_tot_λ_pre = λ_clutter * tildeC;
    for i in 1:n_prev
       @inbounds pack = particle.packs[i];
       if pack.newborn[]
            @inbounds pds.λ_obs_x_pre_tmp[i] = tildeC;
            log_tot_λ_pre += λ_obs * tildeC;
       else
            λ_obs_x = covariates.obs_x(ref_to_mean(pack));
            @inbounds pds.λ_obs_x_pre_tmp[i] = λ_obs_x;
            log_tot_λ_pre += λ_obs * λ_obs_x;
       end
    end

    ## Populate the probability table.
    # No birth, no death, next n = n_prev.
    log_obs_prob = premult * log_tot_λ_pre;
    @inbounds table[1] = log_nobirth_nodeath + log_obs_prob;

    # Birth, next n = n_prev + 1.
    birth_correction = λ_obs * tildeC;
    log_obs_prob = premult * (log_tot_λ_pre + birth_correction);
    @inbounds table[2] = log_birth + log_obs_prob;

    # Death of pack i: (depends on the type of pack d)
    # next n = n_prev - 1.
    for i in 3:(n_prev + 2)
        pack_i = i - 2;
        @inbounds death_correction = -λ_obs * pds.λ_obs_x_pre_tmp[pack_i];
        log_obs_prob = premult * (log_tot_λ_pre + death_correction);
        @inbounds table[i] = log_death + log_obs_prob;
    end
    nothing;
end

"""
Sample the birth, death and association from the probability table populated by
`populate_probability_table!`.

The output consists of a tuple of three integers in the order:
b, (0 or 1) (did a birth occur?)
d, (from 0 to n) (which pack died? 0 = no pack died)
c (from -1 to n + 1) (which pack got associated with the observation? -1 is
means no observation occured and happens deterministically when `arr` is a
vector, 0 = clutter and n + 1 means a new pack)

Arguments:
`rng`: The RNG to use.
`arr`: A matrix or vector populated with probabilites to use in sampling.
"""
function _rng_sample_bdc!(rng::AbstractRNG, arr::AArr{<: AFloat})
    # Normalise the computed probabilities.
    normalise_logweights!(arr); # This is the most costly operation.

    # Sample an outcome proportional on the weights.
    index = wsample_one(rng, view(arr, 1:length(arr)));

    # Reinterpret `index` as a CartesianIndex and get values for
    # b, d and c.
    bdc = get_sampling_outcome(size(arr), index);
    bdc;
end
@inline function sample_bdc!(arr::AArr{<: AFloat})
    _rng_sample_bdc!(Random.GLOBAL_RNG, arr);
end

"""
Determine which sampling outcome occured in the case of an observation.
"""
function get_sampling_outcome(s::Tuple{<: Integer, <:Integer}, i::Integer)
    cart_ind = @inbounds CartesianIndices(s)[i];
    c = @inbounds cart_ind[1]; # Association is given by the row number.
    if c == @inbounds s[1] # Observation is clutter.
        c = 0; # Clutter is 0.
    end
    if @inbounds cart_ind[2] == 1 # If we sampled no birth, no death.
        b = d = 0;
    elseif @inbounds cart_ind[2] == 2 # If we sampled a birth.
        b = 1; d = 0;
    else # If we sampled the death of some pack.
        b = 0;
        d = @inbounds cart_ind[2] - 2;
    end
    b, d, c;
end

"""
Determine which sampling outcome occured in the case of no observation.
Here c = -1 deterministically, since no observation is considered.
"""
function get_sampling_outcome(s::Tuple{<: Integer}, i::Integer)
    if i == 1 # If we sampled no birth, no death.
        b = d = 0;
    elseif i == 2 # If we sampled a birth.
        b = 1; d = 0;
    else # If we sampled the death of some pack.
        b = 0;
        d = i - 2;
    end
    b, d, -1;
end

function get_sampling_outcome(particle::WolfpackParticle,
                              obstype::Type{<: Observation},
                              i::Integer)
    ptsize = prob_table_size(particle, obstype);
    get_sampling_outcome(ptsize, i);
end

"""
Compute the log birth probability of the birth-death model.
"""
@inline function log_birth_prob(λ_birth_tot::Real, λ_death_tot::Real, dt::Real)
    log(1.0 - exp(-dt * λ_birth_tot)) - dt * λ_death_tot;
end
"""
Compute the log death probability of the birth-death model.
"""
@inline function log_death_prob(λ_birth_tot::Real, λ_death_tot::Real, dt::Real, n::Int)
    # n == 0 check avoids the case that the last two terms would evaluate to
    # log(0) - log(0) = NaN if lambda_death_tot = 0 as well.
    n == 0 ? -Inf : -dt * λ_birth_tot + log(1.0 - exp(-dt * λ_death_tot)) - log(n);
end

"""
The function updates a particle's state based on sampled values of birth, death
and association.

Arguments:
`particle`: The particle to be updated.
`bdc`: The tuple of values (b, d, c) output from `sample_bdc!`. This tuple tells
what happened in the sampling.
`o`: The observation. Used if the associated pack happens to be newborn.
`θ`: Parameters of the model.
"""
function update!(particle::WolfpackParticle{WPType}, bdc::NTuple{3, Int},
                 o::Observation, θ::WolftrackParameters) where WPType
    b, d, c = bdc;
    @assert !(b == 1 && d == 1) "`b` and `d` can't be 1 simultaneously.";
    if d == c && d != 0
        # Note, checking d != 0 sicne d = 0, c = 0 is possible with
        # clutter so want to avoid that.
        throw(ArgumentError("`d = c` is disallowed (there is something wrong)."));
    end

    # Figure out what happened and update the number of packs accordingly.
    # Only three events can happen:
    # - b = 0, d = 0 i.e nothing happens.
    # - b = 1, d = 0 i.e a birth happens.
    # - b = 0, d > 0 i.e no birth happens, and some pack dies.
    # In addition to these, the association of the observation (if any) must be
    # handled.
    if b == 1
        # There is a birth.
        # In this case, we simply add a pack to the particle.
        particle.N[] += 1;

        # We need to check that there is enough room in the particle.
        # If there is, the next pack in the particle has newborn = true by
        # default. If there is not, we must make sure that the pack gets
        # newborn = true (because it might have been alive at some point).
        if particle.N[] > length(particle.packs)
            # N got bigger that length of pack vector.
            push!(particle.packs, WPType());
        else
            # N is still less than length of pack vector.
            # Need to ensure that born pack has newborn = true!
            particle.packs[particle.N[]].newborn[] = true;
        end
    elseif d > 0
        # There is a death.
        # In this case, we delete the dth pack from the particle.
        particle.N[] -= 1; splice!(particle.packs, d);

        # If a pack died, the indexing of c might change in the vector, because
        # the packs with indices d + 1, ..., N will change to d, d + 1, ..., N - 1.
        # Hence, if c is greater than d, c needs to be decreased.
        # Note that `populate_probability_table!` makes sure c = d is
        # impossible.
        c > d && (c -= 1;)
    end
    # Handle association.
    c > 0 && associate!(particle.packs[c], o, θ);
    nothing;
end

"""
Associate the wolfpack `pack` with the observation `o`. The function updates
the location information in the wolfpack.
"""
@inline function associate!(pack::AbstractWolfpack, o::WolfpackObs{N},
                            θ::WolftrackParameters{N}) where N
    if pack.newborn[]
        associate_newborn!(pack, o, θ);
    else
        kalman_update!(pack, o, θ);
    end
    nothing;
end

"""
Associate a newborn pack with the observation `o`.
This means that the newborn pack gets the observation location as the mean,
and covariance is set to Σ_obs. (since initial distribution is uniform)
"""
@inline function associate_newborn!(pack::Wolfpack{1}, o::WolfpackObs{1},
                                    θ::WolftrackParameters{1})
    @inbounds pack.mean[1] = o.location[1];
    @inbounds pack.cov[1] = θ.Σ_obs[1];
    pack.newborn[] = false;
    nothing;
end
@inline function associate_newborn!(pack::Wolfpack{2}, o::WolfpackObs{2},
                                    θ::WolftrackParameters{2})
    @inbounds pack.mean[1] = o.location[1];
    @inbounds pack.mean[2] = o.location[2];
    @inbounds pack.cov[1] = θ.Σ_obs[1];
    @inbounds pack.cov[2] = θ.Σ_obs[2];
    @inbounds pack.cov[3] = θ.Σ_obs[3];
    @inbounds pack.cov[4] = θ.Σ_obs[4];
    pack.newborn[] = false;
    nothing;
end
@inline function associate_newborn!(pack::WolfpackConstDiag{2}, o::WolfpackObs{2},
                                    θ::WolftrackParameters{2})
    @inbounds pack.data[1] = o.location[1];
    @inbounds pack.data[2] = o.location[2];
    @inbounds pack.data[3] = θ.Σ_obs[1, 1];
    pack.newborn[] = false;
    nothing;
end
