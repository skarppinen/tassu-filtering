include("../../../config.jl");
include(joinpath(LIB_PATH, "SimpleRaster.jl"));
using ImageFiltering

"""
Find the coordinates of the adjacent pixels (above, right, below, left) of
the pixel [i, j]. The output is always a 4-tuple of tuples in the order
(top, right, bottom, left). If no adjacent pixel is found (pixel at boundary),
the corresponding pixel is (-1, -1) in the output.
"""
function adjacent_pixels(r::AMat, i::Integer, j::Integer)
    top = (i - 1, j);
    right = (i, j + 1);
    bottom = (i + 1, j);
    left = (i, j - 1);
    if i == 1
        top = (-1, -1);
    end
    if i == size(r, 1)
        bottom = (-1, -1);
    end
    if j == 1
        left = (-1, -1);
    end
    if j == size(r, 2)
        right = (-1, -1);
    end
    (top, right, bottom, left);
end

"""
Check if the pixel at [i, j] is a "border pixel", i.e a pixel that has:
1) a value equal to `mis`
2) an adjacent pixel that does not have the value `mis`.
"""
function is_border_pixel(r::AMat{T}, i::Integer, j::Integer, mis::T) where {T <: Real}
    # Check that the pixel has a missing value. Otherwise it cannot be a
    # border pixel.
    r[i, j] != mis && (return false;)

    # Check that there is at least one adjacent pixel that does not have
    # a missing value.
    adj = adjacent_pixels(r, i, j);
    is_border_pixel(r, adj, mis);
end

"""
Check if a pixel is a border pixel given the adjacent pixels.
"""
function is_border_pixel(r::AMat{T}, adj, mis::T) where {T <: Real}
    for a in adj
        a[1] == -1 && (continue;)
        r[a[1], a[2]] != mis && (return true;)
    end
    false;
end

"""
Function finds all border pixels in the matrix `r` and writes the maximum of
the adjacent pixels to them.
"""
function thicken_border!(r::AMat{T}, mis::T) where {T <: Real}
    cind = CartesianIndices(r);
    tmp = Vector{Tuple{Tuple{Int, Int}, Float64}}(undef, 0);
    for ci in cind
        i = ci[1]; j = ci[2];
        r[i, j] != mis && (continue;)
        adj = adjacent_pixels(r, i, j);
        !is_border_pixel(r, adj, mis) && (continue;)

        # Compute maximum.
        s = 0.0;
        for a in adj
            if a[1] == -1
                continue;
            else
                s = max(s, r[a[1], a[2]]);
            end
        end
        push!(tmp, ((i, j), s));
    end
    for t in tmp
        i = t[1][1];
        j = t[1][2];
        v = t[2];
        r[i, j] = v;
    end
    nothing;
end

"""
Thicken the border in the matrix `r` `times` times.
"""
function thicken_border!(r::AMat{T}, mis::T, times::Integer) where {T <: Real}
    for i in 1:times
        thicken_border!(r, mis);
    end
    nothing;
end

"""
The function builds a kernel for Gaussian blurring the raw spatial covariate
for λ_obs. The output function does the blurring in a "border preserving" way:
- if the current point (i.e the point in the raster that is the current midpoint of
the filtering window) is `mis`, `mis` is returned.
- otherwise, the output is the convolution over the filtering weights
and the values in the raster not `mis`, such that the corresponding filter weights
have been normalised to one. The unnormalised weights correspond to an (`l` x `l`)
Gaussian blur, whose blur parameter is `σ`.

`σ`: The blurring parameter for the Gaussian blur.
`l`: Determines the size of the filtering window (l x l). Should be odd.
`mis`: The value assumed for "missing values" in the raster to be smoothed.
"""
function build_blur_kernel(σobs_km::Real, l::Integer, mis::Real)
    @assert isodd(l) "`l` must be odd.";
    @assert σobs_km > 0.0 "`σobs_km` must be strictly positive.";
    g = ImageFiltering.Kernel.gaussian((σobs_km, σobs_km), (l, l));
    midpoint = ceil(Int, l / 2);

    let g = g, midpoint = midpoint, mis = mis
        function(buf)
            # `buf` is the part of the data that we are filtering.
            # If the centerpoint is zero (NA), return immediately `mis`.
            buf[midpoint, midpoint] == mis && (return mis;);

            # Compute the sum of elements in the gaussian kernel that correspond
            # to nonmissing values. Also compute sum of weights.
            s = 0.0; v = 0.0;
            # NOTE: Indexing g this way is ok although [0, 0] is understood as the
            # midpoint with the defaults of ImageFiltering.jl.
            for i in eachindex(buf)
                if buf[i] != mis
                    s += g[i];
                    v += g[i] * buf[i];
                end
            end
            v / s;
        end
    end
end


"""
Apply Gaussian blur to a raster image `r` with the smoothing parameter `σ`.
The window size parameter of blurring kernel is computed based on `σ` such
that the window looks 2σ (in x and y directions) away from each point in the output image.
`mis` stands for the value used for "missing values" in the image. The Gaussian
blur kernel used does not use these values in the blurring computations.
"""
function blur(r::AMat{T}, σ::Real, mis::T = zero(T)) where {T <: AFloat}
    @assert σ > 0.0 "`σ` must be positive.";
    l = nextodd(ceil(Int, 2 * 2 * σ));
    rout = similar(r);
    kernel = build_blur_kernel(σ, l, mis);
    mapwindow!(kernel, rout, r, (l, l), border = Fill(mis));
    rout;
end

function blur(sr::SimpleRaster, σ::Real)
    rout = blur(sr.r, σ, sr.mis);
    SimpleRaster(rout, sr.mis, sr.info);
end

function discrepancy_measure!(r::AMat{<: AFloat}, fo::WolfpackFilterOutput,
                              covariates::IntensityCovariates, wph::WolfpackHistory{2},
                              mask::BitArray{2}, θ; level::AFloat = 0.99)
    discr = zeros(length(fo.time));
    for (i, t) in enumerate(fo.time)
        x = view(fo.X, :, i);
        w = view(fo.W, :, i);
        discr[i] = discrepancy_measure!(r, x, w, t,
                                        covariates.obs_t, covariates.obs_x,
                                        covariates.obs_x.info,
                                        wph, mask, θ; level = level);
    end
    discr;
end

function discrepancy_measure(fo::WolfpackFilterOutput,
                             covariates::IntensityCovariates, wph::WolfpackHistory{2},
                             mask::BitArray{2}, θ; level::AFloat = 0.99)
    r = similar(covariates.obs_x.r);
    discrepancy_measure!(r, fo, covariates, wph, mask, θ; level = level);
end



function discrepancy_measure!(r::AMat{<: AFloat}, x::AVec{WolfpackParticle{WolfpackConstDiag{2, 3}}},
                              w::AVec{<: AFloat}, t::AFloat,
                              obs_t, obs_x, rinfo::RasterInformation,
                              wph::WolfpackHistory{2}, mask::BitArray{2}, θ;
                              level::AFloat = 0.99)
    pixel_area = rinfo.ps_x * rinfo.ps_y;
    λ_obs_t = obs_t(t);
    σobs = sqrt(θ.Σ_obs[1, 1]);
    s = 0.0;
    phd!(r, x, w, rinfo, θ, mask; level = level, obslevel = true);
    for j in 1:size(r, 2)
        for i in 1:size(r, 1)
            xi = j; yi = i;
            if mask[yi, xi]
                k = pixel_midpoint(rinfo, xi, yi);
                lambda_obs_x = obs_x(k);
                d = (r[yi, xi] - total_pack_intensity(wph, t, k, σobs)) ^ 2.0;
                s += lambda_obs_x ^ 2.0 * pixel_area *  d;
            end
        end
    end
    (θ.λ_obs * λ_obs_t) ^ 2.0 * s;
end

"""
Return y and x indexes (in this order) corresponding to a raster that is an upsampled
version of a smaller raster (i.e each cell in the original raster is represented
by multiple in the output raster whose indices are returned).
For example inputs (i = 1, j = 2, 20) return the indices (1:20, 21:40).
"""
function get_upsampled_indices(i, j, n)
    ((i - 1) * n + 1):(i * n),
    ((j - 1) * n + 1):(j * n)
end
