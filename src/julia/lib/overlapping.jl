###############################################################################
#     A file containing overlapping metrics and their associated              #
#     acceptance_probability functions.                                       #
###############################################################################
include("../../../config.jl");
include(joinpath(LIB_PATH, "bbox.jl"));
using StaticArrays
using Distributions

"""
An abstract type for the overlapping metrics.
"""
abstract type OverlapMetric end

##################
### NO OVERLAP ###
##################

"""
A type for the situation where one doesn't account for the overlaps.
"""
struct NoOverlap <: OverlapMetric end

"""
A function that computes the acceptance probability in the case of a NoOverlap
metric.

Fields :
- `μ₁` : a vector corresponding to the mean of the first object
- `Σ₁` : a matrix corresponding to the covariance of the first object
- `μ₂` : a vector corresponding to the mean of the second object
- `Σ₂` : a matrix corresponding to the covariance of the second object
- `no`: a NoOverlap object
"""
@inline function acceptance_probability(μ₁::AVec{<: Real}, Σ₁::AMat{<: Real},
                                        μ₂::AVec{<: Real}, Σ₂::AMat{<: Real},
                                        no::NoOverlap)
    return 1.0;
end

######################
### CIRCLE OVERLAP ###
######################

"""
A type representing an overlapping metric that defines two packs to overlap
if the circular level curves of the covariances of the two packs overlap.

Field:
- `alpha`: a float between 0 and 1 specifying the level curve used to
specify the radius of the circle used in the overlapping computation.
"""
struct CircleOverlap <: OverlapMetric
    critical_value::Float64
    function CircleOverlap(alpha::Float64)
        @assert 0 < alpha < 1 "`alpha_radius` must be between 0 and 1."
        critical_value = quantile(Chisq(2), alpha);
        new(critical_value);
    end
end

"""
Compute the acceptance probability in the case of the CircleOverlap metric.
In the current implementation, the acceptance probabilty is 1 if the circles
constructed based on the level curves of the Gaussian distribution do not overlap,
and 0 if they overlap.

Fields :
- `μ₁` : a vector corresponding to the mean of the first object
- `Σ₁` : a matrix corresponding to the covariance of the first object
- `μ₂` : a vector corresponding to the mean of the second object
- `Σ₂` : a matrix corresponding to the covariance of the second object
- `co` : a CircleOverlap object.
"""
function acceptance_probability(μ₁::AVec{<: Real}, Σ₁::AMat{<: Real},
                                μ₂::AVec{<: Real}, Σ₂::AMat{<: Real},
                                co::CircleOverlap)
    # Check dimensions.
    @assert length(μ₁) == 2 "Circular overlap only works in the two dimensional case.";

    # Compute the radii associated with the packs.
    scaling_factor = co.critical_value;
    ra = sqrt(Σ₁[1, 1] * scaling_factor);
    rb = sqrt(Σ₂[1, 1] * scaling_factor);

    # Check whether the circles overlap.
    ra + rb <= norm((μ₁[1] - μ₂[1], μ₁[2] - μ₂[2])) ? 1.0 : 0.0;
end

######################
### VOLUME OVERLAP ###
######################

"""
A type used in computing overlapping volumes of 2D Gaussians (with a constant diagonal
covariance, e.g C = c * I).
The overlapping computation relies on truncating a Taylor series expansion
(infinite series) with N terms.
During the computation, two temporary vectors, `g` and `c` are populated.
`beta` is a tuning parameter in the computation and is by default set to
29 / 32 given in Sheil and O’Muircheartaigh (1977).
"""
struct VolumeOverlap <: OverlapMetric
    g::Vector{Float64}
    c::Vector{Float64}
    beta::Float64
    function VolumeOverlap(N::Int = 15, beta::AFloat = 29 / 32)
        @assert N >= 1 "`N` must be >= 1.";
        g = zeros(Float64, N - 1);
        c = zeros(Float64, N);
        new(g, c, beta);
    end
end



"""
Compute the acceptance probability when the overlap of two 2D Gaussians is
measured using the volume of their overlap.
"""
@inline function acceptance_probability(mu1::Vec2D{<: Real}, cov1::Cov2D{<: Real},
                                mu2::Vec2D{<: Real}, cov2::Cov2D{<: Real},
                                vo::VolumeOverlap)
    1.0 - normal_2d_overlap_vol!(mu1, cov1, mu2, cov2, vo);
end

"""
Compute the overlapping volume of two 2D Gaussian
distributions N(mu1, C1) and N(mu2, C2). The function assumes that either
1) C1 == C2
2) C1 and C2 are constant diagonal, e.g C1 = c1 * I, C2 = c2 * I.
if these conditions are not met, an assertion error will be thrown.
"""
function normal_2d_overlap_vol!(mu1::Vec2D{<: Real}, cov1::Cov2D{<: Real},
                                mu2::Vec2D{<: Real}, cov2::Cov2D{<: Real},
                                vo::VolumeOverlap)
    if isapprox(cov1, cov2)
        V = normal_2d_samecov_overlap_vol(mu1, mu2, cov1);
    else
        @assert isconstdiag(cov1) && isconstdiag(cov2) "`cov1` and `cov2` must be constant diagonal";
        c1 = cov1[1, 1]; c2 = cov2[1, 1];
        v1 = pdf1_lt_pdf2_vol!(mu1, c1, mu2, c2, vo);
        v2 = pdf1_lt_pdf2_vol!(mu2, c2, mu1, c1, vo);
        V = v1 + v2;
    end
    V;
end

"""
Compute the overlapping volume of two 2D Gaussians that have the same covariance.
"""
function normal_2d_samecov_overlap_vol(mu1::Vec2D{<: Real}, mu2::Vec2D{<: Real}, cov::Cov2D{<: Real})
    if isapprox(mu1, mu2)
        V = 1.0;
    else
        muminus_trans = transpose(mu2 - mu1);
        dmu1 = muminus_trans * (cov \ mu1);
        dmu2 = muminus_trans * (cov \ mu2);
        dC = muminus_trans * inv(cov) * transpose(muminus_trans);
        lower = (dmu1 + dmu2) / 2.0;
        mu_hat = min(dmu1, dmu2);
        V = 2.0 * (1.0 - cdf(Normal(mu_hat, sqrt(dC)), lower));
    end
    V;
end

"""
Compute the integral of the two dimensional Gaussian distribution
N(mu1, c1 * I) over the region {x | N(x; mu1, c1 * I) <= N(x; mu2, c2 * I)}.
This function assumes that c1 != c2.
"""
function pdf1_lt_pdf2_vol!(mu1::Vec2D{<: Real}, c1::Real,
                           mu2::Vec2D{<: Real}, c2::Real,
                           vo::VolumeOverlap)
    # Construct values related to the integral computation.
    K = 1.0 / c2 - 1.0 / c1;
    v = mu1 / c1 - mu2 / c2;
    mu_const = dot(mu1, mu1) / c1 - dot(mu2, mu2) / c2;
    log_const = log(c1 / c2);
    t = (2.0 * log_const + mu_const + dot(v, v) / K) / K;
    delta = SVector{2, Float64}((-mu1[1] - v[1] / K) / sqrt(c1), (-mu1[2] - v[2] / K) / sqrt(c1));
    d = SVector{2, Float64}(c1, c1);

    # Compute integral.
    V = elliptic_int_2d_normal!(t, d, delta, vo);
    K < 0 && (V = 1.0 - V);
    V;
end

"""
Compute an elliptical integral for a 2D Gaussian pdf as given in
"Computation of Multivariate Normal and t Probabilities" (2009), p. 13 - 14.
"""
function elliptic_int_2d_normal!(t::Real, d::Vec2D{<: Real}, delta::Vec2D{<: Real}, vo::VolumeOverlap)
    N = length(vo.c);
    beta = minimum(d) * vo.beta;
    lambda = dot(delta, delta);
    A = beta / sqrt(prod(d));
    c0 = A * exp(-lambda / 2.0);

    # Compute g's.
    @inbounds gamma1 = 1.0 - beta / d[1];
    @inbounds gamma2 = 1.0 - beta / d[2];
    for j in 1:(N - 1)
        @inbounds vo.g[j] = (gamma1 ^ (j - 1) * (j * delta[1] ^ 2.0 * (1.0 - gamma1) + gamma1) +
                             gamma2 ^ (j - 1) * (j * delta[2] ^ 2.0 * (1.0 - gamma2) + gamma2)) / 2.0;
    end

    # Compute c's.
    @inbounds vo.c[1] = c0;
    for i in 2:N
        j = i - 1;
        gcsum = 0.0;
        for k in 1:j
            @inbounds gcsum += vo.g[j - k + 1] * vo.c[k];
        end
        @inbounds vo.c[i] = gcsum / j;
    end

    # Compute the return value (N first terms of the Taylor expansion)
    s = 0.0;
    for i in 1:N
        @inbounds s += vo.c[i] * cdf(Chisq(2.0 + 2.0 * (i - 1)), t / beta);
    end
    s;
end

function get_nxn_reject_region(n::Integer, bb::BoundingBox{Float64, 2, 4},
                               reject_width::Real;
                               outer_border_only::Bool = false)
    x_l = bb[1]; x_u = bb[2];
    y_l = bb[3]; y_u = bb[4];
    @assert x_l < x_u "`l < u` not satisfied for x-axis.";
    @assert y_l < y_u "`l < u` not satisfied for y-axis.";
    @assert isapprox(x_u - x_l, y_u - y_l) "area of `bb` is not square";
    tile_size = (x_u - x_l) / n;

    # Rejection regions at the boundaries.
    top = BoundingBox((x_l, x_u, y_u - reject_width / 2, y_u));
    right = BoundingBox((x_u - reject_width / 2, x_u, y_l, y_u));
    bottom = BoundingBox((x_l, x_u, y_l, y_l + reject_width / 2));
    left = BoundingBox((x_l, x_l + reject_width / 2, y_l, y_u));

    if outer_border_only
        return UnionOfRects(vcat([top, left, bottom, right]));
    end

    # Rejection regions horizontally and vertically, not at boundaries.
    verticals = Vector{BoundingBox{Float64, 2, 4}}(undef, n - 1);
    horizontals = Vector{BoundingBox{Float64, 2, 4}}(undef, n - 1);
    for i in 1:length(verticals)
        xl = x_l + tile_size * i - reject_width / 2;
        xu = xl + reject_width;
        yl = y_l;
        yu = y_u;
        verticals[i] = BoundingBox((xl, xu, yl, yu));
    end
    for i in 1:length(horizontals)
        xl = x_l;
        xu = x_u;
        yl = y_l + tile_size * i - reject_width / 2;
        yu = yl + reject_width;
        horizontals[i] = BoundingBox((xl, xu, yl, yu));
    end
    UnionOfRects(vcat([top, left, bottom, right],
                      horizontals, verticals));
end

function get_boundary_reject_region(bb::BoundingBox{Float64, 2, 4},
                                    reject_width::Real)
    x_l = bb[1]; x_u = bb[2];
    y_l = bb[3]; y_u = bb[4];
    @assert x_l < x_u "`l < u` not satisfied for x-axis.";
    @assert y_l < y_u "`l < u` not satisfied for y-axis.";
    @assert isapprox(x_u - x_l, y_u - y_l) "area of `bb` is not square";

    # Rejection regions at the boundaries.
    top = BoundingBox((x_l, x_u, y_u - reject_width, y_u));
    right = BoundingBox((x_u - reject_width, x_u, y_l, y_u));
    bottom = BoundingBox((x_l, x_u, y_l, y_l + reject_width));
    left = BoundingBox((x_l, x_l + reject_width, y_l, y_u));

    UnionOfRects(vcat([top, left, bottom, right])),
    BoundingBox((reject_width, bb[2] - reject_width,
                 reject_width, bb[4]- reject_width));
end

"""
A type representing a union of 2D rectangles.
"""
struct UnionOfRects
    areas::Vector{BoundingBox{Float64, 2, 4}}
end
import Base.in
function in(p::AVec{T}, uor::UnionOfRects) where T
    for a in uor.areas
        p in a && (return true);
    end
    false;
end

struct DistrWithRejection{D <: MultivariateDistribution{Continuous}} <: MultivariateDistribution{Continuous}
    base_dist::D
    uor::UnionOfRects
    function DistrWithRejection(base_dist::MultivariateDistribution{Continuous},
                                uor::UnionOfRects)
        new{typeof(base_dist)}(base_dist, uor);
    end
end

import Random.rand!
function rand!(dwr::DistrWithRejection, x::AVec{Float64})
    success = false;
    while !success
        rand!(dwr.base_dist, x);
        success = !(x in dwr.uor)
    end
    x;
end

import Random.rand
function rand(dwr::DistrWithRejection)
    l = length(dwr.base_dist);
    x = zeros(l);
    rand!(dwr, x);
    x;
end
