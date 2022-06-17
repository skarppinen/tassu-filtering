include("../../../config.jl");
include(joinpath(LIB_PATH, "n-dist.jl"));
include(joinpath(LIB_PATH, "phd.jl"));

function get_progress_msg(t1::Real, t2::Real, msg::AString, unit::AString = "s")
    msg * string(" (took ", round(t2 - t1, digits = 3), unit, ")");
end

function most_likely_locations(out)

    # Get the mode of territory count at each time point.
    modes = mode.(get_discrete_distribution(out));

    # Loop through each time point and find among particles
    # that have N == mode the particle with the maximum weight.
    # For that particle, get the locations of the territories and
    # also an indicator vector telling whether the territory is
    # newborn.
    result = map(1:length(modes)) do j
        m = modes[j];
        i_w_N_mode = filter(i -> out.X[i, j].N[] == m, 1:size(out.X, 1));
        ind = findmax(out.W[i_w_N_mode, j])[2]
        i_chosen_particle = i_w_N_mode[ind];

        # Get data from found particle.
        particle = out.X[i_chosen_particle, j];
        res = map(1:particle.N[]) do i
            Vector{Float64}(particle.packs[i].data),
            particle.packs[i].newborn[]
        end
        getindex.(res, 1), getindex.(res, 2)
    end
    result;
end


function jld2_export_filtering_result(out, θ, covariates, datadf;
                                      plot_times::AVec{<: AFloat} = Float64[],
                                      outpath::AString = joinpath("output", "jld2", "filter-plot-data.jld2"),
                                      verbose::Bool = true)
    @assert issorted(plot_times) "`plot_times` should be sorted.";
    if !endswith(outpath, ".jld2")
        outpath = outpath * ".jld2";
    end
    mkpath(dirname(outpath));
    verbose && println("Output file is $outpath");

    # Indices of requested plotting times in filtering result.
    isempty(plot_times) && (plot_times = copy(out.time);)
    plot_times_i = map(plot_times) do t
        findfirst(x -> isapprox(x, t), out.time)
    end;
    # Probabilities of different N counts at requested time points.
    Nprobs = dist_of_N(out)[:, plot_times_i];

    # Locations of territories for most likely particle.
    mll = most_likely_locations(out)[plot_times_i];

    # Compute PHD estimate raster at each requested time point.
    verbose && println("Computing PHD estimate at different time points.")
    phdarr = zeros(covariates.obs_x.info.n_y, covariates.obs_x.info.n_x,
                   length(plot_times));
    mask = nonmissingmask(covariates.obs_x);
    for i in 1:length(plot_times)
        t = plot_times[i];
        r = view(phdarr, :, :, i);

        ti = plot_times_i[i];
        x = view(out.X, :, ti);
        w = view(out.W, :, ti);
        phd!(r, x, w, covariates.obs_x.info, θ, mask; level = 0.99);
        verbose && println("Done $i of $(length(plot_times))");
    end

    jldopen(outpath, "w") do file
        info = covariates.obs_x.info;
        bbox = info.bbox;
        xlim = [bbox[1], bbox[2]]
        ylim = [bbox[3], bbox[4]];

        file["times"] = plot_times;

        file["phd"] = phdarr;
        file["phd-nx"] = info.n_x;
        file["phd-ny"] = info.n_y;
        file["phd-psx"] = info.ps_x;
        file["phd-psy"] = info.ps_y;
        file["phd-xlim"] = xlim;
        file["phd-ylim"] = ylim;

        file["Nprobs"] = Nprobs;

        file["data-t"] = datadf.t;
        file["data-x"] = datadf.x;
        file["data-y"] = datadf.y;

        file["obs-x"] = covariates.obs_x.r;
        file["obs-t-t"] = covariates.obs_t.t;
        file["obs-t-y"] = covariates.obs_t.y;

        file["most-likely-locs"] = getindex.(mll, 1);
        file["most-likely-packtypes"] = getindex.(mll, 2);

        file["theta-lambdaobs"] = θ.λ_obs;
        file["theta-lambdac"] = θ.λ_clutter;
        file["theta-sigmaobs"] = sqrt(θ.Σ_obs[1, 1]);
        file["theta-birthrate"] = θ.λ_birth;
        file["theta-birthrate0"] = θ.λ_birth0;
        file["theta-deathrate"] = θ.λ_death;
    end
    verbose && println("Output saved to $outpath");
end
