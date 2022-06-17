include("../../../config.jl");
include(joinpath(LIB_PATH, "wolfpack-types.jl"));

"""
A struct representing a 1D or 2D bounding box.
In the 1D case, `lim` = (lower, upper).
In the 2D case, `lim` = (lower_x, upper_x, lower_y, upper_y).
"""
struct BoundingBox{T <: Real, N, L}
   lim::NTuple{L, T}
   function BoundingBox(lim::NTuple{L, T}) where {L, T <: Real}
      l = length(lim);
      dim = convert(Int, l / 2);
      @assert l in (2, 4) "`length(lim)` must be 2 or 4."
      if l == 2
         @assert lim[1] <= lim[2] "inconsistent dimensions";
      else
         @assert lim[1] <= lim[2] && lim[3] <= lim[4] "inconsistent dimensions";
      end
      new{T, dim, l}(lim);
   end
end

"""
Constructor for a dummy bounding box.
"""
function BoundingBox(d::Integer)
    @assert 1 <= d <= 2 "`d` must be 1 or 2.";
    d == 1 && return BoundingBox((0.0, 0.0));
    return BoundingBox((0.0, 0.0, 0.0, 0.0));
end
@inline function Base.getindex(b::BoundingBox, i::Integer)
   b.lim[i];
end

"""
Construct a BoundingBox from a one dimensional normal distribution.
`level` in (0, 1) controls the base size for computing the bounds, and
`offset` can be used to add additional space to the box.
"""
function BoundingBox(mean::MVector{1, T}, cov::MMatrix{1, 1, T};
                     level::Float64 = 0.99) where T <: Real
   q = quantile(Normal(0, 1), level);
   m = mean[1]; sd = sqrt(cov[1, 1]);
   BoundingBox((mean[1] - q * sd, mean[1] + q * sd));
end

"""
Construct a BoundingBox from a two dimensional normal distribution.
`level` in (0, 1) controls the base size for computing the bounds, and
`offset` can be used to add additional space to the box.
"""
function BoundingBox(mean::StaticArray{Tuple{2}, T, 1}, cov::StaticArray{Tuple{2, 2}, T, 2};
                     level::Float64 = 0.99) where T <: Real
   q = quantile(Normal(0, 1), level);
   m1 = mean[1]; m2 = mean[2];
   sd1 = sqrt(cov[1, 1]); sd2 = sqrt(cov[2, 2]);
   BoundingBox((m1 - q * sd1, m1 + q * sd1,
                m2 - q * sd2, m2 + q * sd2));
end

function BoundingBox(mean::AVec{<: Real}, cov::AFloat;
                     level::Float64 = 0.99)
    q = quantile(Normal(0, 1), level);
    m1 = mean[1]; m2 = mean[2];
    sd = sqrt(cov);
    BoundingBox((m1 - q * sd, m1 + q * sd,
                 m2 - q * sd, m2 + q * sd));
end

function BoundingBox(wp::Wolfpack{N}; level::Float64 = 0.99) where N
    wp.newborn[] && (return nothing);
    BoundingBox(wp.mean, wp.cov, level = level);
end

function BoundingBox(wp::WolfpackConstDiag{2}; level::Float64 = 0.99)
    wp.newborn[] && (return nothing);
    BoundingBox(view(wp.data, 1:2), wp.data[3]; level = level);
end

function BoundingBox(x::WolfpackParticle; level::Float64 = 0.99)
    BoundingBox(view(x.packs, 1:x.N[]); args = (level = level,));
end

function BoundingBox(v::AArr{<: Any}; args::NamedTuple = NamedTuple())
    #length(v) <= 0 && return nothing;
    #init = BoundingBox(v[1]);
    reduce((x, y) -> combine(x, BoundingBox(y; args...)), v, #view(v, 2:length(v)),
           init = nothing);#init);
end

function BoundingBox(fo::WolfpackFilterOutput; level::Float64 = 0.99)
    BoundingBox(fo.X; args = (level = level,) );
end

function BoundingBox(wpl::WolfpackLifetime{2}; offset::Float64 = 5.0)
    BoundingBox((wpl.location[1] - offset,
                 wpl.location[1] + offset,
                 wpl.location[2] - offset,
                 wpl.location[2] + offset));
end

function BoundingBox(wph::WolfpackHistory; offset::Float64 = 5.0)
    d = dimension(wph);
    BoundingBox(wph.lifetimes);
end

function BoundingBox(obs::WolfpackObs{2}; offset::Float64 = 0.0)
    BoundingBox((obs.location[1] - offset,
                 obs.location[1] + offset,
                 obs.location[2] - offset,
                 obs.location[2] + offset));
end

function BoundingBox(obs::WolfpackMisObs)
    nothing;
end

function xlim(bb::BoundingBox)
    (bb[1], bb[2])
end

function ylim(bb::BoundingBox{T, 2, 4}) where T <: Real
    (bb[3], bb[4])
end

import DataFrames.combine
function combine(b1::BoundingBox{T, 1, 2}, b2::BoundingBox{T, 1, 2}) where T <: Real
   BoundingBox((min(b1[1], b2[1]), max(b1[2], b2[2])));
end

function combine(b1::BoundingBox{T, 2, 4}, b2::BoundingBox{T, 2, 4}) where T <: Real
   BoundingBox((min(b1[1], b2[1]), max(b1[2], b2[2]),
                min(b1[3], b2[3]), max(b1[4], b2[4])));
end

function combine(n::Nothing, b::BoundingBox)
    b;
end
function combine(b::BoundingBox, n::Nothing)
    b;
end
function combine(n1::Nothing, n2::Nothing)
    n1;
end

import Base.in
function in(x::AVec{Float64}, bb::BoundingBox{T, 2}) where T
    x[1] < bb[1] && return false;
    x[1] > bb[2] && return false;
    x[2] < bb[3] && return false;
    x[2] > bb[4] && return false;
    true;
end
