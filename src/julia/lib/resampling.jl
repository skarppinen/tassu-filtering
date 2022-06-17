#############################################
### This file contains resampling algorithms.
###
include("../../../config.jl");
include(joinpath(LIB_PATH, "wolfpack-types.jl"));
using Random
using LinearAlgebra

abstract type Resampling end
abstract type StandardResampling <: Resampling end

"""
Resamplings that require no arguments.
"""
struct MultinomialResampling <: StandardResampling end
struct StratifiedResampling <: StandardResampling end
struct SystematicResampling <: StandardResampling end

"""
Struct representing killing resampling where the kill constant
can be computed with a univariate function.
"""
struct KillingResampling{KillConstFun <: Function} <: StandardResampling
  kf::KillConstFun # The function used to compute the killing constant.
  function KillingResampling(kf::Function)
    new{typeof(kf)}(kf);
  end
end

"""
"Default" constructor. Use 1.0 / max(p) as the killing constant.
"""
function KillingResampling()
  KillingResampling(p -> 1.0 / maximum(p));
end

"""
Shorthand for computing the killing constant.
"""
function (k::KillingResampling)(p::AbstractVector{T})::T where {T <: Real}
  k.kf(p);
end

"""
    resample!(ind, p, kr)
Sample from a discrete probability distribution p using killing resampling.

# Arguments:
- `ind`: Integer vector where output is stored.
- `p`: Vector that determines a probability mass function (assumed normalised).
- `kr`: A KillingResampling object determining how the killing resampling is performed.

Killing resampling is only possible when ``length(ind) = length(p)``,
and ``ind[i] ≂̸ i`` with probability ``p[i] * killconst``. Largest valid ``killconst = 1.0 / maximum(p)``
(not checked).
"""
function resample!(ind::AVec{<: Integer}, p::AVec{<: Real},
                   kr::KillingResampling)
    m = length(p);
    n = length(ind);
    @assert m == n "Killing resampling possible only if m = n";

    K = 1; @inbounds S = p[1]; U_ = 1.0; n_ = 0;
    killconst = kr(p);
    for j in 1:n
      if rand() < (@inbounds p[j] * killconst)
        @inbounds ind[j] = j # The ones that "succeed" in the first sampling.
      else
        @inbounds ind[j] = 0; n_ += 1; # The ones that did not, n_ is their amount.
      end
    end

    # Sample those which did not succeed.
    for j in 1:n
      (@inbounds ind[j] != 0) && continue; # Skip those which did.
      # Order statistics in reverse order:
      U_ = rand() ^ (1.0 / n_) * U_;
      n_ -= 1;
      # ...same as forward order by symmetry...
      U = 1.0 - U_;

      # Find K such that F(K) >= u
      while K < m && U > S
        K = K + 1       # Note that K is not reset!
        @inbounds S = S + p[K] # S is the partial sum up to K
      end
      @inbounds ind[j] = K
    end
    nothing
end

"""
Efficient multinomial resampling by sampling order statistics.
The time complexity is O(n).
"""
function resample!(ind::AVec{<: Integer},
                   p::AbstractVector{<: AFloat},
                   mr::MultinomialResampling)
    m = length(p);
    n = length(ind);
    K = 1; @inbounds S = p[1];
    U_ = 1.0; n_ = n;
    for j in 1:n
      # Order statistics in reverse order:
      U_ = rand() ^ (1.0 / n_) * U_;
      n_ -= 1;
      # ...same as forward order by symmetry...
      U = 1.0 - U_;
      # Find K such that F(K) >= u
      while K < m && U > S
        K = K + 1;       # Note that K is not reset!
        @inbounds S = S + p[K]; # S is the partial sum up to K
      end
      @inbounds ind[j] = K;
    end
    nothing;
end

"""
Stratified resampling in O(N).
"""
function resample!(ind::AVec{<: Integer}, p::AVec{<: AFloat},
                   sr::StratifiedResampling)
    N = length(ind); M = length(p);
    u = 0.0; K = 1; S = @inbounds p[1];
    for j in 1:N
        # Sample u ~ U((j-1) / N, j / N). Note that all the u's come ordered.
        u = (j - 1 + rand()) / N;

        # Find index K such that u <= sum(p[1:K]).
        while K < M && u > S
            K += 1;
            @inbounds S = S + p[K];
        end
        @inbounds ind[j] = K;
    end
    nothing;

end

function resample!(ind::AVec{<: Integer}, p::AVec{<: AFloat},
                   sr::SystematicResampling)
    N = length(ind); M = length(p);
    u_ = rand(); K = 1; S = @inbounds p[1];
    for j in 1:N
        # Set u. Note that all the u's come ordered.
        u = (j - 1 + u_) / N;

        # Find index K such that u <= sum(p[1:K]).
        while K < M && u > S
            K += 1;
            @inbounds S = S + p[K];
        end
        @inbounds ind[j] = K;
    end
    nothing;
end

include(joinpath(LIB_PATH, "optimal-resampling.jl"));
