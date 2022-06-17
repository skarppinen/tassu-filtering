## An experiment with Tassu data where the interest is looking at
# how many particles are sufficient for the filtering model with a fixed
# set data.
include(joinpath("../../../config.jl"));
include("script-config.jl");
args = ArgParse.parse_args(ARGS, ARGUMENT_CONFIG["npar-check"]);

include(joinpath(LIB_PATH, "master.jl"));
include(joinpath(LIB_PATH, "SimpleRaster.jl"));
include(joinpath(LIB_PATH, "n-dist.jl"));
using JLD2, Dates

## Arguments.
model = args["intensity-model"];
npar = args["npar"];
nreps = args["nreps"];
λ_birth = λ_death = args["lambda-bd"];
λ_birth0 = args["lambda-b0"];
λ_clutter = args["lambda-clutter"];
λ_obs = args["lambda-obs"];
diameter_km = args["diameter"];
contour_prob = args["contour-prob"];
use_known_locations = false; #args["use-known-locations"];
max_npacks = args["max-n-packs"];
verbose = args["verbose"];
outfolder = args["outfolder"];
use_simulated_data = true; #args["use-simulated-data"];
jobid = args["jobid"];

if use_known_locations && use_simulated_data
    msg = string("`use-known-locations = true` disallowed with `use-simulated-data = true`");
    throw(ArgumentError(msg));
end

experiment_name = let
    name = "npar-check-tassu";
    name *= string("-m", model);
    if use_known_locations
        name *= "-known-prior-locs";
    end
    if use_simulated_data
        name *= "-simdata";
    end
    name;
end
if isnothing(outfolder)
    experiment_path = joinpath(pwd(), experiment_name);
else
    experiment_path = joinpath(outfolder, experiment_name);
end
isnothing(jobid) && (jobid = abs(rand(Int)));

## Load data.
dateformat = DateFormat("y-m-d H:M:S");
#d = read_jld2_dataframe("tassu-s2019-s2020-groups-only");
#sort!(d, :time);
startmonth = 4;#month(d[1, :time]);
startyear = 2019;#year(d[1, :time]);
startdatetime = let
    startdate = string(startyear, "-", lpad(startmonth, 2, "0"), "-", "01");
    starttime = "00:00:00";
    DateTime(startdate * " " * starttime, dateformat);
end
enddatetime = startdatetime + Year(1);

## Get weekly spaced timepoints to save. (first interval is a bit longer)
first_interval_day_offset = Day(enddatetime - (startdatetime + Week(52)));
save_dttms = collect(startdatetime:Week(1):enddatetime);
for i in 2:length(save_dttms)
    save_dttms[i] += first_interval_day_offset;
end
save = cumsum(getfield.(diff(save_dttms), :value) ./ 1000 ./ 60 ./ 60 ./ 24);
T = save[end];

## Some fixed values.
discretiser_seed = 5052021;
resampling = OptimalResampling;
cellsize = 1000 * 1000;
n_threads = 1;
gating = NoGating();

## Define covariates.
σ_obs_sq_km = sigmaobssq_diameter(contour_prob, diameter_km);
obs_t, obs_x = get_covariate_data("timepred-linreg", model = model,
                                  σ_km = sqrt(σ_obs_sq_km));
covariates = IntensityCovariates(obs_t = obs_t, obs_x = obs_x,
                                 birth = PiecewiseConstant((0.0, nextfloat(T))),
                                 death = PiecewiseConstant((0.0, nextfloat(T))));
obs_x_I = integral(obs_x);
area = length(obs_x.r[obs_x.r .!= obs_x.mis]) * cellsize;

## Get dataset.
# d = filter(r -> obs_x((r[:x], r[:y])) != obs_x.mis, d);
# d[!, :time_days] = let
#     @assert issorted(d[!, :time]) "`time` column must be sorted."
#     times = diff(vcat([startdatetime], d[:, :time]));
#     values = getfield.(times, :value) ./ (1000 * 60 * 60 * 24);
#
#     # Rounding here a bit since sometimes the numerical precision suffers
#     # and the values don't come out as sorted. (makes no practical difference)
#     cumsum(round.(values, digits = 8));
# end;
# Random.seed!(discretiser_seed);
# data = discretiser(d[:, :time_days],
#                    map((x, y) -> [x, y], d[:, :x], d[:, :y]), T, shuffle = true);
# Random.seed!();

## Initial distribution.
init = if use_known_locations && !use_simulated_data
    pack_locations = read_jld2_dataframe("pack-locations-simple-early-2019") |>
        x -> copy.(eachrow(x)) |>
        x -> map(y -> [y[:X], y[:Y]], x);
else
    mu = 45.0;
    sigma = 10.0;
    rnb = mu ^ 2.0 / (sigma ^ 2.0 - mu);
    pnb = mu / sigma ^ 2.0;
    N_init_dist = truncated(NegativeBinomial(rnb, pnb), 30, 70);
end

## Parameters.
σ_obs_sq = (sqrt(σ_obs_sq_km) * 1000) ^ 2;
θ = WolftrackParameters(λ_obs = λ_obs,
                        λ_clutter = λ_clutter,
                        λ_death = λ_death,
                        λ_birth = λ_birth,
                        λ_birth0 = λ_birth0,
                        Σ_obs = [σ_obs_sq 0.0; 0.0 σ_obs_sq],
                        inv_area = inv(area),
                        λ_obs_spat_I = obs_x_I);

if use_simulated_data && !use_known_locations
    Random.seed!(discretiser_seed);
    startN = 47;#rand(N_init_dist); # Get initial N.

    # Build uniform distribution on the domain specified by the spatial
    # covariate.
    dist = let
       obs_x = covariates.obs_x;
       cart_pos_intensity = CartesianIndices(obs_x.r)[obs_x.r .> obs_x.mis];
       midpoints = map(cart_pos_intensity) do cart
           pixel_midpoint(obs_x.info, cart[2], cart[1]);
       end
       UniformGridDistr(midpoints, obs_x.info.ps_x);
    end
    # Simulate territories.
    wph = sim_latent_dynamic_nhpp(θ, startN, covariates.birth, covariates.death,
                                  dist; om = NoOverlap());

    # Simulate observations.
    bs = let pixel_size = 1000.0
        rbbox = covariates.obs_x.info.bbox;
        bounds_x = (rbbox[1], rbbox[2]);
        bounds_y = (rbbox[3], rbbox[4]);
        ((0.0, T), bounds_x, bounds_y);
    end
    λ_tot_max = λ_obs_tot_max(wph, θ, covariates);
    λ = let wph = wph, θ = θ, covariates = covariates
        function(x)
            λ_obs_tot(x, wph, θ, covariates);
        end
    end
    sim = sim_hpp(bs, λ_tot_max, λ);
    sort!(sim);
    data = discretiser(map(first, sim), map(x -> [x[2], x[3]], sim), T)
    Random.seed!()
end
thicken_border!(obs_x.r, obs_x.mis, ceil(Int, 2 * sqrt(σ_obs_sq_km)));

## Run filter.
verbose && println("Starting filtering..");
verbose && println("Parameters are: $θ");
mkpath(experiment_path);
result_path = joinpath(experiment_path, string(npar));
mkpath(result_path);
verbose && println("Path for results is $result_path");

for i in 1:nreps
    verbose && println("Iteration ($i / $nreps)");
    verbose && print("Running filter..");
    verbose && (iter_start_t = time();)

    out = run_wolfpack_filter(data, θ, init, save;
                              covariates = covariates, max_npacks = max_npacks,
                              npar = npar, gating = gating,
                              resampling = resampling, n_threads = n_threads);
    if verbose
       filter_end_t = time();
       println(get_progress_msg(iter_start_t, filter_end_t, " finished."));
    end

    let
        filename = string("jobid-", jobid, "-", randstring(10), ".jld2");
        path = joinpath(result_path, filename);
        jldopen(path, "w"; iotype = IOStream) do file
            file["args"] = args;
            file["time"] = save;
            file["N_dist_est"] = get_discrete_distribution(out);
        end
    end

    if verbose
       println(get_progress_msg(iter_start_t, time(), "Finished iteration"));
    end
end
