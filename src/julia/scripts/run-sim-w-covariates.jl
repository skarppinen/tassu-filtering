## Fix packs and run filter multiple times to see if packs can be inferred.
include("../../../config.jl");
include("script-config.jl");
args = ArgParse.parse_args(ARGS, ARGUMENT_CONFIG["sim-w-covariates"]);

include(joinpath(LIB_PATH, "master.jl"));
include(joinpath(LIB_PATH, "SimpleRaster.jl"));
include(joinpath(LIB_PATH, "n-dist.jl"));
include(joinpath(LIB_PATH, "phd.jl"));
using Dates
using JLD2

# Arguments.
model = args["intensity-model"];
npar = args["npar"];
λ_birth = λ_death = args["lambda-bd"];
λ_obs = args["lambda-obs"];
λ_clutter = args["lambda-clutter"];
λ_birth0 = args["lambda-b0"];
M = args["nreps"];
contour_prob = args["contour-prob"];
diameter_km = args["diameter"];
max_npacks = args["max-n-packs"];
verbose = args["verbose"];
outfolder = args["outfolder"];
#thicken = args["thicken"];
jobid = args["jobid"];

## Some settings and derived values.
isnothing(jobid) && (jobid = abs(rand(Int)));

experiment_name = let
    name = "sim-w-covariates";
    name *= string("-m", model);
    name *= string("-prob", contour_prob);
    name *= string("-diam", diameter_km);
    name *= string("-bd", λ_birth);

    if λ_clutter > 0.0
        name *= string("-clut", λ_clutter);
    end

    #if thicken
    #    name *= "-thicken";
    #end
    #if half
#        name *= "-half";
#    end
    name;
end
if isnothing(outfolder)
    experiment_path = joinpath(pwd(), experiment_name);
else
    experiment_path = joinpath(outfolder, experiment_name);
end

# Some fixed values.
resampling = OptimalResampling;
n_threads = 1;
gating = NoGating();

#inflation = 1.5; # Variance inflation.
#A = 1000; # Assumed size of habitat for computing sigmaobs without inflation.
σ_km = sqrt(sigmaobssq_diameter(contour_prob, diameter_km));
σsq_km = σ_km * σ_km;
σ_m = σ_km * 1000.0;
σsq_m = σ_m * σ_m;

## Times to save.
# Get weekly spaced timepoints to save. (first interval is a bit longer)
startyear = 2019; startmonth = 4;
startdatetime = DateTime(startyear, startmonth, 1);
enddatetime = startdatetime + Year(1);
first_interval_day_offset = Day(enddatetime - (startdatetime + Week(52)));
save_dttms = collect(startdatetime:Week(1):enddatetime);
for i in 2:length(save_dttms)
    save_dttms[i] += first_interval_day_offset;
end
save = cumsum(getfield.(diff(save_dttms), :value) ./ 1000 ./ 60 ./ 60 ./ 24);
T = save[end];

## Covariates and parameters.
ofun, raster = if model in (2, 3)
    # Apply Gaussian blur.
    get_covariate_data("timepred-linreg",
                       model = model, σ_km = σ_km);
else
    get_covariate_data("timepred-linreg",
                       model = model);
end

# Make non-zero intensity areas a bit larger.
raster_thick = deepcopy(raster);
thicken_border!(raster_thick.r, raster_thick.mis, ceil(Int, 2 * sqrt(σsq_km)));

# Build covariates.
covariates = IntensityCovariates(obs_t = ofun, obs_x = raster_thick,
                                 birth = PiecewiseConstant((0.0, nextfloat(T))),
                                 death = PiecewiseConstant((0.0, nextfloat(T))));

## Pack locations.
pack_locations = read_jld2_dataframe("pack-locations-simple-early-2019") |>
    x -> filter(x) do r
        raster_thick((r[:X], r[:Y])) != raster_thick.mis;
    end;

wph = let
    pack_i = 1:nrow(pack_locations);
    #pack_i = if half
        # Pick only half of packs. (randomly)
    #    total_npacks = nrow(pack_locations);
    #    sample(1:total_npacks, div(total_npacks, 2), replace = false);
    #else

    #end
    lifetimes = map(pack_i) do i
        x = pack_locations[i, :X];
        y = pack_locations[i, :Y];
        WolfpackLifetime(0.0, T, [x, y]);
    end
    WolfpackHistory(0.0, T, lifetimes)
end

N_packs_true = length(wph.lifetimes);
N_init_dist = truncated(Poisson(N_packs_true),
                        N_packs_true - 10, N_packs_true + 10);

## Bounds for simulating the model by thinning.
#bs = let pixel_size = 1000.0
#    rbbox = raster_thick.info.bbox;
#    bounds_x = (rbbox[1], rbbox[2]);
#    bounds_y = (rbbox[3], rbbox[4]);
#    ((0.0, T), bounds_x, bounds_y);
#end

## Clutter distribution
clutter_dist = let
   obs_x = covariates.obs_x;
   cart_pos_intensity = CartesianIndices(obs_x.r)[obs_x.r .> obs_x.mis];
   midpoints = map(cart_pos_intensity) do cart
       pixel_midpoint(obs_x.info, cart[2], cart[1]);
   end
   UniformGridDistr(midpoints, obs_x.info.ps_x);
end

## Parameters.
# NOTE: Area and integral correspond to non-modified raster.
raster_integral = integral(raster);
area = sum(raster.r .!= raster.mis) * 1000.0 * 1000.0;
inv_area = inv(area);
θ = WolftrackParameters(λ_obs = λ_obs,
                        λ_clutter = λ_clutter,
                        λ_death = λ_death,
                        λ_birth = λ_birth,
                        λ_birth0 = λ_birth0,
                        Σ_obs = [σsq_m 0.0; 0.0 σsq_m],
                        inv_area = inv_area,
                        λ_obs_spat_I = raster_integral);

## For simulating observations.
#λ_tot_max = λ_obs_tot_max(wph, θ, covariates);
#λ = let wph = wph, θ = θ, covariates = covariates
#    function(x)
#        λ_obs_tot(x, wph, θ, covariates);
#    end
#end

## Do computation.
mkpath(experiment_path);
result_path = joinpath(experiment_path, string(npar));
mkpath(result_path);
verbose && println("Path for results is $result_path");
done_str = " finished!";

mask = nonmissingmask(raster, covariates.obs_x.info);
discr_tmp = similar(covariates.obs_x.r);
for i in 1:M
    t1 = time();
    verbose && println("Iteration ($i / $M)");

    verbose && print("Simulating observations..");
    sim = obs_given_packs(wph, covariates, θ, clutter_dist);
    #sim = sim_hpp(bs, λ_tot_max, λ);
    #sort!(sim);
    data = discretiser(map(first, sim), map(x -> [x[2], x[3]], sim), T)
    verbose && println(get_progress_msg(t1, time(), done_str));

    verbose && print("Running filter..");
    t_start_filter = time();
    out = run_wolfpack_filter(data, θ, N_init_dist, save;
                                    covariates = covariates, max_npacks = max_npacks,
                                    npar = npar, gating = gating,
                                    resampling = resampling,
                                    n_threads = n_threads);
    verbose && println(get_progress_msg(t_start_filter, time(), done_str));

    let
        filename = string("jobid-", jobid, "-", randstring(10), ".jld2");
        path = joinpath(result_path, filename);
        jldopen(path, "w"; iotype = IOStream) do file
            file["time"] = save;
            file["N_dist_est"] = get_discrete_distribution(out);
            file["true_N"] = N_packs_true;
            file["args"] = args;

        end
    end
    verbose && println(get_progress_msg(t1, time(), "Finished iteration."));
end
