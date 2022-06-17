## Simulation experiment where the spatial covariate is a piecewise constant
# on a grid. The observations from the packs are sampled such that they will
# not fall to the borders where the covariate changes value, or to the edges of
# the simulation area.
include("../../../config.jl");
include("script-config.jl");
args = ArgParse.parse_args(ARGS, ARGUMENT_CONFIG["bias-check"]);

include(joinpath(LIB_PATH, "master.jl"));
include(joinpath(LIB_PATH, "observation-simulator.jl"));
include(joinpath(LIB_PATH, "SimpleRaster.jl"));
include(joinpath(LIB_PATH, "n-dist.jl"));
include(joinpath(LIB_PATH, "spatial-operations.jl"));
include(joinpath(LIB_PATH, "phd.jl"));
using JLD2

## Read parameters.
npar = args["npar"];
nreps = args["nreps"];
N_init = args["N-init"];
avoid_boundary = true;
λ_obs = args["lambda-obs"];
λ_birth = λ_death = args["lambda-birth-death"];
λ_birth0 = args["lambda-birth0"];
λ_clutter = args["lambda-clutter"];
σobs = args["sigmaobs"];
outfolder = args["outfolder"];
verbose = args["verbose"];
jobid = args["jobid"];

## Checks.
if avoid_boundary
    @assert λ_clutter <= 0.0 "`avoid-boundary` only possible if clutter intensity is 0.0";
    @assert λ_obs == 1.0 "`avoid-boundary` only possible if intensity scaling is 1.0";
end

## Some settings and derived values.
isnothing(jobid) && (jobid = abs(rand(Int)));

experiment_name = let
    name = "bias-check";
    if avoid_boundary
        name *= "-avbound";
        name *= string("-sigma", σobs);
    end
    name *= string("-bd", λ_birth);
    name;
end
if isnothing(outfolder)
    experiment_path = joinpath(pwd(), experiment_name);
else
    experiment_path = joinpath(outfolder, experiment_name);
end

# Draw observations (practically) from R^2, if `avoid_boundary = true`.
T = 50.0;
area_side_length = 100.0;
reject_region_size = 6 * σobs;
obs_x_bbox = BoundingBox((-reject_region_size, area_side_length + reject_region_size,
                              -reject_region_size, area_side_length + reject_region_size));
obs_x = SimpleRaster(ones(Float64, 1, 1), 0.0,
                         RasterInformation(obs_x_bbox, 1, 1));
obs_t = PiecewiseConstant((0.0, nextfloat(T)), 1.0);
pack_area_bb = BoundingBox((0.0, area_side_length, 0.0, area_side_length));
raster_integral = area_side_length * area_side_length;


# Some fixed values.
T = round(obs_t.t[end], digits = 6);
save = 1.0:1.0:T;
N_init_dist = truncated(Poisson(N_init), max(N_init - 10, 0), N_init + 10);
resampling = OptimalResampling;

# Bounds for temporal and spatial coordinates.
covar_bb = obs_x.info.bbox;
bs = ((0.0, T), (covar_bb[1], covar_bb[2]), (covar_bb[3], covar_bb[4]));

# Pack location distribution.
dist = product_distribution([Uniform(pack_area_bb[1], pack_area_bb[2]),
                             Uniform(pack_area_bb[3], pack_area_bb[4])]);
overlap_metric = NoOverlap();

## Covariate object.
covariates = IntensityCovariates(obs_t = obs_t,
                                 birth = PiecewiseConstant((0.0, nextfloat(T))),
                                 death = PiecewiseConstant((0.0, nextfloat(T))),
                                 obs_x = obs_x);
inv_area = inv((pack_area_bb[2] - pack_area_bb[1]) *
               (pack_area_bb[4] - pack_area_bb[3]));

## Parameters.
θ = WolftrackParameters(λ_obs = λ_obs,
                        λ_birth = λ_birth,
                        λ_birth0 = λ_birth0,
                        λ_death = λ_death,
                        λ_clutter = λ_clutter,
                        Σ_obs = [σobs^2.0 0.0; 0.0 σobs^2.0],
                        inv_area = inv_area,
                        λ_obs_spat_I = raster_integral);

## Build folder structure.
mkpath(experiment_path);
result_path = joinpath(experiment_path, string(npar));
mkpath(result_path);
verbose && println("Path for results is $result_path");

## Run simulation.
mask = nonmissingmask(covariates.obs_x);
discr_tmp = similar(obs_x.r); # Temporary for computing discrepancy measure.
for i in 1:nreps
    verbose && println("Iteration $i / $nreps.");
    t1 = time();

    ## Simulate packs.
    verbose && print("Simulating packs..");
    startN = rand(N_init_dist); # Get initial N.
    wph = sim_latent_dynamic_nhpp(θ, startN, covariates.birth, covariates.death,
                                  dist; om = overlap_metric);
    true_N = map(t -> n_packs_alive(wph, t), save);
    verbose && println(get_progress_msg(t1, time(), " finished!"));

    ## Simulate observations.
    t_start_obs_sim = time();
    verbose && print("Simulating observations..");
    sim = obs_given_packs(wph, covariates, θ, nothing);
    data = discretiser(map(first, sim), map(x -> [x[2], x[3]], sim), T)
    verbose && println(get_progress_msg(t_start_obs_sim, time(), " finished!"));

    ## Run filter.
    t_start_filter = time();
    verbose && print("Running filter..");
    out = run_wolfpack_filter(data, θ, N_init_dist, save; npar = npar,
                              n_threads = 1, resampling = resampling,
                              covariates = covariates)
    verbose && println(get_progress_msg(t_start_filter, time(), " finished!"));

    let
        filename = string("jobid-", jobid, "-", randstring(10), ".jld2");
        path = joinpath(result_path, filename);
        jldopen(path, "w"; iotype = IOStream) do file
            file["args"] = args;
            file["time"] = save;
            file["N_dist_est"] = get_discrete_distribution(out);
            file["true_N"] = true_N;
        end
    end
    verbose && println(get_progress_msg(t1, time(), "Finished iteration."));
end
