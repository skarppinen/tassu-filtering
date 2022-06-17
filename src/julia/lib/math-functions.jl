include("../../../config.jl");
include(joinpath(LIB_PATH, "type-aliases.jl"));

"""
flip(i, first, second)

If `i` == `first` return `second`, otherwise return `first`.
"""
function flip(i::Integer, first::Integer, second::Integer)
    i == first ? second : first;
end

"""
Function checks if a matrix is constant diagonal, e.g can be written
as c * I, with a constant c, and the identity matrix I.
"""
@inline function isconstdiag(C::AMat{<: Real})
    isdiag(C) || (return false;)
    res = true;
    for i in 1:(size(C, 1) - 1)
        res = res ? isapprox(C[i, i], C[i + 1, i + 1]) : false;
    end
    res;
end

"""
Function checks if a 2D covariance matrix is constant diagonal, e.g can be written
as c * I, with a constant c, and the identity matrix I.
"""
@inline function isconstdiag(C::Cov2D{<: Real})
    isdiag(C) && isapprox(C[1, 1], C[2, 2]);
end

"""
Normalise a vector of weight logarithms, `log_weights`, in place.
After normalisation, the weights are in the linear scale.
Additionally, the logarithm of the linear scale mean weight is returned.
"""
@inline function normalise_logweights!(log_weights::AbstractArray{<: Real})
  m = maximum(log_weights);
  if isapprox(m, -Inf) # To avoid NaN in case that all values are -Inf.
    log_weights .= zero(eltype(log_weights));
    return -Inf;
  end
  log_weights .= exp.(log_weights .- m);
  log_mean_weight = m + log(mean(log_weights));
  normalize!(log_weights, 1);
  log_mean_weight;
end

"""
Compute log(sum(exp.(`x`))) in a numerically stable way.
"""
@inline function logsumexp(x::AbstractArray{<: Real})
  m = maximum(x);
  isapprox(m, -Inf) && (return -Inf;) # If m is -Inf, without this we would return NaN.
  s = 0.0;
  for i in eachindex(x)
    s += exp(x[i] - m);
  end
  m + log(s);
end

function threaded_mapreduce(f, op, x)
    @assert length(x) % Threads.nthreads() == 0
    results = zeros(eltype(x), Threads.nthreads()) # Allocation, but still faster.
    Threads.@threads for tid in 1:Threads.nthreads()
        # split work
        acc = zero(eltype(x))
        len = div(length(x), Threads.nthreads())
        domain = ((tid-1)*len +1):tid*len
        acc = op(acc, mapreduce(f, op, view(x, domain)))
        results[tid] = acc
    end
    foldl(op, results)
end

@inline function threaded_logsumexp(x::AbstractArray{<: Real})
  m = maximum(x);
  isapprox(m, -Inf) && (return -Inf;) # If m is -Inf, without this we would return NaN.
  s = threaded_mapreduce(y -> exp(y - m), +, x);
  m + log(s);
end

"""
Sample one index from 1:length(x) proportional on the weights in `x`.
It is assumed that the weights are normalised to 1.
"""
@inline function wsample_one(rng::AbstractRNG, x::AArr{<: AFloat})
  u = rand(rng);
  s = zero(eltype(x));
  for i in eachindex(x)
    s += x[i];
    if u <= s
      return i;
    end
  end
  length(x);
end

midpoint(x, y) = (x + y) / 2.0;
