## Functions for computing the PHD estimate.
include("../../../config.jl");
include(joinpath(LIB_PATH, "wolfpack-types.jl"));
include(joinpath(LIB_PATH, "SimpleRaster.jl"));
using JLD2, Dates

"""
Function adds the density induced by the pack `pack` to the raster `r`.
Note that in the case of a newborn pack, the density is added to all pixels.
"""
function add_pack_density!(r, pack::WolfpackConstDiag, w::AFloat,
                           θ, rinfo::RasterInformation;
                           level::AFloat = 0.99,
                           obslevel::Bool = true)

    if pack.newborn[]
        r .+= w * θ.inv_area;
    else
        # Pack is "normal".
        if obslevel
            σ = sqrt(pack.data[3] + θ.Σ_obs[1, 1]);
        else
            σ = sqrt(pack.data[3]);
        end
        r_pixel_area = rinfo.ps_x * rinfo.ps_y;

        # Get bounding box of pack corresponding to the given probability region.
        pack_bbox = BoundingBox(view(pack.data, 1:2), σ * σ;
                                level = level);

        # Find out pixel values in `r` that contain the probability mass.
        # Any one of these might be -1 = "out of bounds".
        ymini, xmini = get_pixel_coordinates(rinfo, pack_bbox[1], pack_bbox[3]);
        ymaxi, xmaxi = get_pixel_coordinates(rinfo, pack_bbox[2], pack_bbox[4]);

        # Handle y and x bounds being "out of bounds".
        ymini <= 0 && (ymini = 1;)
        xmini <= 0 && (xmini = 1;)
        ymaxi <= 0 && (ymaxi = size(r, 1);)
        xmaxi <= 0 && (xmaxi = size(r, 2);)

        # Write PHD density of pack to pixels.
        for xi in xmini:xmaxi
            for yi in ymini:ymaxi
                # Get midpoint of pixel.
                mp = pixel_midpoint(rinfo, xi, yi);

                # Approximate density at pixel midpoint..
                @inbounds v = pdf(Normal(pack.data[1], σ), mp[1]) *
                              pdf(Normal(pack.data[2], σ), mp[2]) *
                              r_pixel_area;

                # ..scaled by weight.
                r[yi, xi] += w * v;
            end
        end
    end
    nothing;
end

"""
Function computes the PHD to the raster `r` based on particles in `x` and
weights in `w`. `rinfo` should contain the raster data about `r`. `θ` should
contain the parameters used in the filtering. The relevant parameters are
`Σ_obs` and `inv_area`, others are not used.
`obslevel` controls whether the PHD corresponds to density in the level of
observations or means. `level` controls computation of bounding boxes for packs
that have been associated at least once, the default should suffice.
`mask` may be set to specify some irregular region inside `r`, outside of which
the PHD should always be zero.
"""
function phd!(r::AMat{<: AFloat},
              x::AVec{<: WolfpackParticle{WolfpackConstDiag{2, 3}}},
              w::AVec{<: AFloat},
              rinfo::RasterInformation,
              θ,
              mask::Union{Nothing, BitArray{2}} = nothing;
              obslevel::Bool = true,
              level::AFloat = 0.99)
    r .= zero(eltype(r));
    for i in 1:length(x)
        @inbounds weight = w[i];
        @inbounds particle = x[i];
        N = particle.N[];
        for j in 1:N
            @inbounds pack = particle.packs[j];
            add_pack_density!(r, pack, weight, θ, rinfo;
                              level = level, obslevel = obslevel)
        end
    end
    if mask != nothing
        r[.!mask] .= 0.0;
    end
    nothing;
end

function phd(x::AVec{<: WolfpackParticle{WolfpackConstDiag{2, 3}}},
             w::AVec{<: AFloat},
             rinfo::RasterInformation,
             θ; obslevel::Bool = true,
             mask::Union{Nothing, BitArray{2}} = nothing,
             level::AFloat = 0.99)
    r = zeros(rinfo.n_y, rinfo.n_x);
    phd!(r, x, w, rinfo, θ, mask; obslevel = obslevel, level = level);
    r;
end
