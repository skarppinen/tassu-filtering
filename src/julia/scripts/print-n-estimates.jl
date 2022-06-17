include("../../../config.jl");
using JLD2, Distributions, DataFrames, RCall

function print_n_dist_stats(ndist)
    println("Mean: $(mean(ndist))")
    println("Median: $(median(ndist))")
    println("Mode: $(mode(ndist))")
    println("Standard deviation: $(std(ndist))")
    println("Variance: $(var(ndist))")
    println("99% probability interval: $(quantile.(ndist, [0.005, 0.995]))")
    println("95% probability interval: $(quantile.(ndist, [0.025, 0.975]))")
    println("90% probability interval: $(quantile.(ndist, [0.05, 0.95]))")
end

folder = joinpath(JLD2_PATH);
fns = filter(s -> occursin(r"^n-est", s), readdir(folder));
models_fp = [
    only(filter(s -> !occursin("clut0.475", s) && occursin("m3", s), fns))
    only(filter(s -> occursin("clut0.475", s) && occursin("m3", s), fns))
    only(filter(s -> !occursin("clut0.475", s) && occursin("m1", s), fns))
    only(filter(s -> occursin("clut0.475", s) && occursin("m1", s), fns))
] |> x -> joinpath.(folder, x);

# Luke.
terr_est_2020 = rcopy(R"""
    out <- readRDS(file.path($PROJECT_ROOT, "data/rds/official-territory-count-estimates-2020.rds"))
""");
println("## Luke ##");
DiscreteNonParametric(parse.(Int, terr_est_2020[:, :count]), terr_est_2020[:, :probability]) |>
    x -> print_n_dist_stats(x)
println();

# Models 1 - 4.
for (i, model_fp) in enumerate(models_fp)
    dist = jldopen(model_fp, "r") do file
        file["N_dist_est"];
    end
    println("## MODEL $i ##")
    print_n_dist_stats(dist)
    println()
end
