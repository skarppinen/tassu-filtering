include("../../../config.jl");
if !isdir(SIMEXP_PATH)
    println("No folder $SIMEXP_PATH found, cannot proceed. Download simulation data first using `download-simulation-data.jl`.");
    exit(1);
end
include(joinpath(LIB_PATH, "master.jl"));
include(joinpath(LIB_PATH, "plot-functions.jl"));
using Distributions, JLD2
using DataFrames
using RCall
plots_path = joinpath(PROJECT_ROOT, "output", "plots");
mkpath(plots_path);
R"""
source(file.path($PROJECT_ROOT, "src/R/r-plot-helpers.R"))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(latex2exp))
"""

midpoint(x, y) = (x + y) / 2.0;
function process_bias_check(folder::AString; npars::AVec{<: Integer} = Int[],
                            niter::Int = -1)
    let out = DataFrame()
        inputfolder = folder;
        stat_funs = [mean, median, mode];

        # Get particle counts in experiment.
        if isempty(npars)
            npars = Int[];
            for str in readdir(inputfolder)
                try
                    push!(npars, parse(Int, str));
                catch
                    continue
                end
            end
        end

        # Build DataFrame from results.
        for npar in npars
            path = joinpath(inputfolder, string(npar));
            if niter <= 0
                files = readdir(path)
            else
                files = readdir(path)[1:niter]
            end
            for (i, name) in enumerate(files)
                jldopen(joinpath(path, name), "r") do file
                    datadf = DataFrame(time = file["time"],
                    iteration = i,
                    npar = npar,
                    trueN = file["true_N"]);

                    statdf = DataFrame()
                    for f in stat_funs
                        statdf[!, string(f)] = f.(file["N_dist_est"]);
                    end
                    append!(out, hcat(datadf, statdf));
                end
            end

        end
        out;
    end
end

function process_npar_check(inputfolder::AString;
        npars::Vector{Int} = Int[],
        niter::Int = -1,
        stat_funs = [mean])
    # Get particle counts in experiment.
    if isempty(npars)
        npars = Int[];
        for str in readdir(inputfolder)
            try
                push!(npars, parse(Int, str));
            catch
                continue
            end
        end
    end

    out = DataFrame();
    for npar in npars
        path = joinpath(inputfolder, string(npar));
        files = readdir(path);
        if niter > 0
            files = files[1:niter];
        end
        for (i, name) in enumerate(files)
            jldopen(joinpath(path, name), "r") do file
               datadf = DataFrame(iteration = i,
                                  time = file["time"],
                                  npar = npar);
               @assert file["args"]["npar"] == npar "oops"
               statdf = DataFrame();
               for f in stat_funs
                   if f isa Tuple
                      statdf[!, f[1]] = f[2].(file["N_dist_est"]);
                   else
                      statdf[!, string(f)] = f.(file["N_dist_est"]);
                   end
               end
                   append!(out, hcat(datadf, statdf));
            end
        end
    end
    out;
end

## Comparison between ideal and approximate birth & death model.
λ_bds = [0.0010, 0.0015, 0.0020, 0.0025];
out_df = map(λ_bds) do λ_bd
    N_init = 47;
    Δ = 1.0;
    T = 365.0;
    nsim = 10000;
    θ = WolftrackParameters(λ_birth = λ_bd,
                            λ_birth0 = 0.0,
                            λ_death = λ_bd,
                            λ_obs = 1.0,
                            λ_clutter = 0.0,
                            Σ_obs = [1.0 0.0; 0.0 1.0],
                            inv_area = 1.0);

    # Discrete.
    out_discrete = map(1:nsim) do i
        sim_wolf_birth_model_discrete(θ, Int(T), Δ, N_init);
    end |> x -> hcat(x...);

    # Continuous.
    out_continuous = map(1:nsim) do i
        sim_wolf_birth_model_continuous(θ, T, Δ, N_init)
    end |> x -> hcat(x...);

    df_discrete = DataFrame(type = "discrete", time = 1.0:1.0:T,
                            var = mapslices(var, out_discrete; dims = 2)[:, 1],
                            lambda_bd = λ_bd);
    df_continuous = DataFrame(type = "continuous", time = 1.0:1.0:T,
                              var = mapslices(var, out_continuous; dims = 2)[:, 1],
                              lambda_bd = λ_bd);
    vcat(df_discrete, df_continuous);
end |> x -> vcat(x...);

R"""
filename <- "approx-ideal-bd-model-comparison.pdf"
d <- $out_df
d$lambda_bd <- make_particle_factor_name(d$lambda_bd, textprefix = "lambda[bd] == ")

plt <- ggplot(d, aes(x = time, y = sqrt(var), linetype = type)) +
    geom_line() +
    facet_wrap(~ lambda_bd, labeller = label_parsed) +
    scale_linetype_discrete(labels = c("Ideal model", "Approximate model")) +
    scale_x_continuous(expand = rep(0.02, 2)) +
    scale_y_continuous(breaks = seq(0, 10, by = 1)) +
    labs(x = "Time", y = "Sample standard deviation of number of territories") +
    theme_bw() +
    theme(legend.title = element_blank())

if (!(filename == "")) {
        ggsave(filename = file.path($plots_path, filename), plot = plt)
}
plt
"""

## Synthetic experiment.
terr_size_vs_area_summ = let
    bd = 0.0015;
    sigmas = [15.0, 10.0, 5.0, 2.0, 1.0];
    map(sigmas) do sigma
        experiment = "bias-check-avbound-sigma" * string(sigma) * string("-bd", bd);
        d = joinpath(SIMEXP_PATH, experiment) |>
            x -> process_bias_check(x);
        gdf = groupby(d, [:npar, :time]);
        out = combine(gdf, [:mean, :trueN] => ( (m, N) -> mean(m .- N) ) => :grand_mean);
        out[!, :sigma] .= sigma;
        out[!, :lambda_bd] .= bd;
        out;
    end |> x -> vcat(x...)
end;

R"""
d <- $terr_size_vs_area_summ
bd_size <- 0.0015
d <- filter(d, lambda_bd == bd_size)
d[["npar_fact"]] <- make_particle_factor_name(d[["npar"]])
sigmacolors <- GRAY_TO_BLACK_COLOR_FUN(length(unique(d[["sigma"]])))

plt <- ggplot(d, aes(x = time, y = grand_mean, color = factor(sigma))) +
    geom_line(size = 0.25) +
    facet_wrap(~ factor(npar_fact)) +
    scale_y_continuous(expand = rep(0.01, 2)) +
    scale_color_manual(values = sigmacolors, name = TeX("$\\sigma_{obs}$")) +
    labs(x = "Time", y = TeX("$E\\[\\hat{N}_t - N_t\\]$")) +
    THEME_FACETED +
    theme(panel.grid.minor = element_blank())

    h <- 2
    w <- 3.5
    scaling <- 1.0
    ggsave(filename = file.path($plots_path,
                                paste0("bias-check-avbound-summary-bd",
                                        gsub("\\.", "-", bd_size), ".pdf")),
                                plot = plt, width = w * scaling, height = h * scaling)
plt
"""

## Semisynthetic experiment.
experiment = "sim-w-covariates-m3-prob0.95-diam65.48545-bd0.0015-clut0.475"
plot_data = let experiment = experiment
    joinpath(SIMEXP_PATH, experiment) |>
    x -> process_bias_check(x);
end;
filename = replace(experiment, "." => "-") * ".pdf";
gdf = groupby(plot_data, [:time, :npar])
out = combine(gdf, [:mean, :trueN] => ((m, N) -> mean(m .- N)) => :mean_dev,
             [:mean, :trueN] => ((m, N) -> mean(abs.(m .- N))) => :mean_abs_dev);
mean_df = stack(out, [:mean_dev, :mean_abs_dev]);

R"""

    mean_data <- $mean_df
    d <- $plot_data
    d$npar_fact <- make_particle_factor_name(d$npar, textprefix = "M = ")
    mean_data$npar_fact <- make_particle_factor_name(mean_data$npar, textprefix = "M = ")
    var_lookup <- c(mean_dev = "Average deviation", mean_abs_dev = "Average absolute deviation")
    mean_data$variable_fact <- factor(unname(var_lookup[mean_data$variable]), levels = unname(var_lookup))

    plt <- ggplot(d) +
    geom_line(aes(x = time, y = mean - trueN, group = iteration, color = "Deviation in individual simulations"),
                    alpha = 0.1) +
    geom_line(data = mean_data, aes(x = time, y = value, color = variable_fact)) +
    scale_x_continuous(expand = c(0.01, 0.01)) +
    scale_color_manual(values = c("cyan", "orange", "black")) +
    facet_wrap(~ npar_fact) +
    geom_hline(aes(yintercept = 0), color = "red") +
    labs(y = "Deviation", x = "Time") +
    theme_bw() +
    theme(legend.position = "top",
          legend.title = element_blank())

    if (!($filename == "")) {
        ggsave(filename = file.path($plots_path, $filename), plot = plt)
    }
    plt
"""

out_tassu = let
    d = process_npar_check(joinpath(SIMEXP_PATH,
                           "npar-check-tassu-m3-known-prior-locs"), stat_funs = [mean]);
    gdf = groupby(d, [:time, :npar]);
    out = combine(gdf, :mean => std => :sdmean);
    out[!, :type] .= "tassu";
    out;
end
out_simdata = let
    d = process_npar_check(joinpath(SIMEXP_PATH,
                           "npar-check-tassu-m3-simdata"), stat_funs = [mean]);
    gdf = groupby(d, [:time, :npar]);
    out = combine(gdf, :mean => std => :sdmean);
    out[!, :type] .= "simdata";
    out;
end
out = vcat(out_tassu, out_simdata);

R"""
   d <- $out
   d$npar_fact <- make_particle_factor_name(d$npar, textprefix = "")
   type_lookup <- c(tassu = "Tassu dataset", simdata = "Simulated dataset")
   d$type <- factor(type_lookup[d$type], levels = unname(type_lookup))
   colorfun <- GRAY_TO_BLACK_COLOR_FUN(length(levels(d$npar_fact)))

   plt <- ggplot(d, aes(x = time, y = sdmean)) +
     geom_line(aes(color = npar_fact)) +
     scale_color_manual(values = colorfun, name = "M") +
     scale_x_continuous(expand = rep(0.01, 2)) +
     scale_y_continuous(expand = rep(0.01, 2)) +
     facet_wrap(~ type) +
     labs(y = "Standard deviation of mean estimates", x = "Time") +
     THEME_FACETED +
     theme(legend.key.height = unit(5, 'pt'))

   pdf(file = file.path($plots_path, "npar-check-tassu-vs-simdata.pdf"),
       height = 2.0, width = 3)
    print(plt)
   dev.off()
   plt
"""

## Intensity functions.
obs_t, obs_x, raw_obs_x = let
    contour_prob = 0.95;
    diameter_km = 65.48545
    model = 3;
    σ_obs_sq_km = sigmaobssq_diameter(contour_prob, diameter_km);
    obs_t, obs_x = get_covariate_data("timepred-linreg",
                                      model = model,
                                      σ_km = sqrt(σ_obs_sq_km));
    thicken_border!(obs_x.r, obs_x.mis, ceil(Int, 2 * sqrt(σ_obs_sq_km)));
    raw_obs_x = get_raw_spatial_grid(model);

    obs_t, obs_x, raw_obs_x;
end

plt_obs_t = plot_piecewise(obs_t, args = (xlabel = "Time (days)", ylabel = "Intensity", color = :black))
add_months_s2019_s2020!(plt_obs_t, 3.2)
savefig(plt_obs_t, joinpath(plots_path, "obs-t-m3.pdf"))

plt_obs_x = plot_raster(obs_x, args = (grid = false, c = cgrad([:white, :black]),
                                       aspect_ratio = 1.0, size = (600, 500)))
savefig(plt_obs_x, joinpath(plots_path, "obs-x-m3.pdf"))

plt_raw_obs_x = plot_raster(raw_obs_x, args = (grid = false, c = cgrad([:white, :black]), aspect_ratio = 1.0,
                                           size = (600, 500)))
savefig(plt_raw_obs_x, joinpath(plots_path, "raw-obs-x-m3.pdf"))

## Study area and continental map.
script_path = joinpath(SRC_PATH, "R", "plot-study-area-and-continental-map.R");
run(`Rscript $script_path`);

## PHDs.
script_path = joinpath(SRC_PATH, "R", "plot-phds.R");
run(`Rscript $script_path`);
