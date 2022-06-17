include("../../../config.jl");
include("wolfpack-types.jl");
include("overlapping.jl");
include("bbox.jl");
include("n-dist.jl");
include("covariates.jl");
using Plots

function plot_piecewise(pc::PiecewiseConstant; plt::Plots.Plot = Plots.plot(),
                        args::NamedTuple = NamedTuple())
    Plots.plot!(plt, pc.t, vcat(pc.y, [pc.y[end]]), seriestype = :steppost,
                legend = false; args...);
    plt;
end

function plot_raster(raster::SimpleRaster; args::NamedTuple = NamedTuple(),
                     misval::Real = 0.0, n::Integer = 0)
    xlim = (raster.info.bbox[1], raster.info.bbox[2]);
    ylim = (raster.info.bbox[3], raster.info.bbox[4]);
    if n <= 0
        x = LinRange(xlim[1], xlim[2], raster.info.n_x);
        y = LinRange(ylim[1], ylim[2], raster.info.n_y);
    else
        x = LinRange(xlim[1], xlim[2], n);
        y = LinRange(ylim[1], ylim[2], n);
    end
    f = let misval = misval, raster = raster
        function(x, y)
            v = raster((x, y));
            v == raster.mis && (return misval;)
            v;
        end
    end
    Plots.plot(x, y, f, seriestype = :heatmap, xlim = xlim,
    ylim = ylim; args...)
end

function add_months_s2019_s2020!(plt, height = 3.5)
    ts = cumsum(vcat([0.0], month_length_days.(4:12, 2019), month_length_days.(1:3, 2020)));
    mtext = ["apr", "may", "june", "july", "aug", "sep", "oct", "nov", "dec", "jan", "feb", "mar"];
    years = vcat(repeat([2019], 9), repeat([2020], 3));
    mtext = mtext .* "\n" .* string.(years);

    for (i, v) in enumerate(ts)
        Plots.plot!(plt, [v], seriestype = :vline, color = :black, alpha = 0.2,
                    legend = false, grid = false)
        if i != 1
            Plots.annotate!(plt, midpoint(ts[i], ts[i - 1]), height, mtext[i - 1],
                            alpha = 0.3, font(8, "Times New Roman"));
        end
    end
    plt;
end
