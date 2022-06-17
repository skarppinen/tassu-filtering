## Run filter with real Tassu data.
println("Unable to run script (input data not available, since it is not public). This script may only be viewed.")
exit(1);

println(string("Script is ", @__FILE__));
include("../../../config.jl");
include("script-config.jl");
args = ArgParse.parse_args(ARGS, ARGUMENT_CONFIG["filter-tassu"]);
include(joinpath(LIB_PATH, "master.jl"));
include(joinpath(LIB_PATH, "SimpleRaster.jl"));
using Dates, JLD2

## Arguments.
model = args["intensity-model"];
npar = args["npar"];
λ_birth = λ_death = args["lambda-bd"];
λ_birth0 = args["lambda-b0"];
λ_clutter = args["lambda-clutter"];
λ_obs = args["lambda-obs"];
use_gating = args["gating"];
diameter_km = args["diameter"];
contour_prob = args["contour-prob"];
max_npacks = args["max-n-packs"];
randomize = args["randomize"];
use_known_locations = args["use-known-locations"];
jobid = args["jobid"];
verbose = args["verbose"];
outfolder = args["outfolder"];

discretiser_seed = 5052021;
if randomize
    Random.seed!(); filtering_seed = rand(UInt);
else
    filtering_seed = 12022021;
end

experiment_basename = "filter-tassu";
experiment_name = let
    name = experiment_basename;
    name *= string("-npar", npar);
    name *= string("-m", model);
    if use_known_locations
        name *= "-locprior";
    else
        name *= "-unifprior";
    end
    name *= "-prob" * string(contour_prob);
    name *= "-diam" * string(diameter_km);
    name *= string("-bd", λ_birth);
    if λ_clutter > 0.0
        name *= "-clut" * string(λ_clutter);
    end
    if randomize
        name *= string("-r", jobid);
    end

    name;
end
if isnothing(outfolder)
    experiment_path = joinpath(pwd(), experiment_basename);
else
    experiment_path = joinpath(outfolder, experiment_basename);
end
verbose && println("Input arguments are $args");

## Load data.
dateformat = DateFormat("y-m-d H:M:S");
d = read_jld2_dataframe("tassu-s2019-s2020-groups-only");
sort!(d, :time);
startmonth = month(d[1, :time]);
startyear = year(d[1, :time]);
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

## Some values.
resampling = OptimalResampling;
cellsize = 1000 * 1000;
n_threads = 1;

## Define covariates.
#σ_obs_sq_km = inflation * sigmaobssq(prob_area, A);
σ_obs_sq_km = sigmaobssq_diameter(contour_prob, diameter_km);
obs_t, obs_x = get_covariate_data("timepred-linreg",
                                  model = model,
                                  σ_km = sqrt(σ_obs_sq_km));

covariates = IntensityCovariates(obs_t = obs_t, obs_x = obs_x);
obs_x_I = integral(obs_x);
area = length(obs_x.r[obs_x.r .!= obs_x.mis]) * cellsize;

# Make non-zero intensity areas a bit larger.
thicken_border!(obs_x.r, obs_x.mis, ceil(Int, 2 * sqrt(σ_obs_sq_km)));

## Subset to data within raster only and build discretised dataset.
d = filter(r -> obs_x((r[:x], r[:y])) != obs_x.mis, d);
d[!, :time_days] = let
    @assert issorted(d[!, :time]) "`time` column must be sorted."
    times = diff(vcat([startdatetime], d[:, :time]));
    values = getfield.(times, :value) ./ (1000 * 60 * 60 * 24);

    # Rounding here a bit since sometimes the numerical precision suffers
    # and the values don't come out as sorted. (makes no practical difference)
    cumsum(round.(values, digits = 8));
end;

Random.seed!(discretiser_seed);
data = discretiser(d[:, :time_days],
                   map((x, y) -> [x, y], d[:, :x], d[:, :y]), T, shuffle = true);

## Initial distribution.
init = if use_known_locations
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

σ_obs_sq = (sqrt(σ_obs_sq_km) * 1000) ^ 2;
θ = WolftrackParameters(λ_obs = λ_obs,
                        λ_clutter = λ_clutter,
                        λ_death = λ_death,
                        λ_birth = λ_birth,
                        λ_birth0 = λ_birth0,
                        Σ_obs = [σ_obs_sq 0.0; 0.0 σ_obs_sq],
                        inv_area = inv(area),
                        λ_obs_spat_I = obs_x_I);

if use_gating
    gating = ChiSquareGating(0.001, θ);
else
    gating = NoGating();
end

## Run filter.
verbose && println("Starting filtering..");
verbose && println("Parameters are: $θ");
Random.seed!(filtering_seed);
out = run_wolfpack_filter(data, θ, init, save;
                          covariates = covariates, max_npacks = max_npacks,
                          npar = npar, gating = gating,
                          resampling = resampling, n_threads = n_threads);
drop_unused!(out); # Trim preallocated unused packs from result.

## Save data.
verbose && println("Saving output..");

# Get data.
datadf = let
    d = filter(x -> x isa WolfpackObs, data)
    t = map(x -> x.time, d);
    dx = map(x -> x.location[1], d);
    dy = map(x -> x.location[2], d);
    DataFrame(t = t, x = dx, y = dy);
end

# Output JLD2s.
filename = experiment_name * "-plot-data" * ".jld2";
jld2_export_filtering_result(out, θ, covariates, datadf;
                             verbose = verbose,
                             outpath = joinpath(experiment_path, filename))

filename = experiment_name * ".jld2";
jldopen(joinpath(experiment_path, filename), "w"; iotype = IOStream) do file
    file["args"] = args;
    file["time"] = save;
    file["theta"] = θ;
    file["N_dist_est"] = get_discrete_distribution(out);
    file["N_init_dist"] = init;
end
verbose && println("Done.");
