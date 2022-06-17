using StaticArrays
using Distributions

"""
A type alias for a fixed size vector of length N.
For example,
MVector{2, T} <: Vec2D{T} and SVector{2, T} <: Vec2D{T}.
"""
const VecND{N, T} = StaticArray{Tuple{N}, T, 1};
const Vec2D{T} = VecND{2, T};
const Vec1D{T} = VecND{1, T};

"""
A type alias for a covariance of dimension N.
For example,
MMatrix{2, 2, T, 4} <: Cov2D{T} and SMatrix{2, 2, T, 4} <: Cov2D{T}.
"""
const CovND{N, T} = StaticArray{Tuple{N, N}, T, 2};
const Cov2D{T} = CovND{2, T};
const Cov1D{T} = CovND{1, T};

"""
An abstract type to describe discrete truncated distributions.
"""
const DiscreteTruncated = Truncated{T, Discrete, S} where {T, S};

"""
An abstract type for describing distributions that are discrete and have finite
support.
"""
const DiscreteFiniteSupport = Union{DiscreteTruncated, DiscreteNonParametric,
                                    DiscreteUniform, Bernoulli, BetaBinomial,
                                    Binomial, Categorical, Hypergeometric,
                                    PoissonBinomial, Skellam};
