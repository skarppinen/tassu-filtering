include("../../../config.jl");
using Random
using Distributions
using LinearAlgebra
using DataFrames
using CSV
using StaticArrays
using Distributions

"""
A struct keeping parameters of the wolftracking model.
"""
struct WolftrackParameters{N, L, B}
    λ_obs::Float64
    λ_clutter::Float64
    λ_birth::Float64
    λ_birth0::Float64
    λ_death::Float64
    Σ_obs::SMatrix{N, N, Float64, L}
    inv_area::Float64
    λ_obs_spat_I::Float64 # The integral over the spatial lambda_obs.

    # Precomputed upon construction.
    _tildeC::Float64
    _log_inv_area::Float64 # The log of inverse area.
    function WolftrackParameters(λ_obs::AFloat,
                                 λ_clutter::AFloat,
                                 λ_birth::AFloat,
                                 λ_birth0::AFloat,
                                 λ_death::AFloat,
                                 Σ_obs::AMat{<: AFloat},
                                 inv_area::AFloat,
                                 λ_obs_spat_I::AFloat)
        N = size(Σ_obs, 1);
        @assert N == size(Σ_obs, 2) "different column and row dimension in Σ_obs";
        @assert λ_obs >= 0 "all λ's must be positive";
        @assert λ_clutter >= 0 "all λ's must be positive";
        @assert λ_birth >= 0 "all λ's must be positive";
        @assert λ_birth0 >= 0 "all λ's must be positive";
        @assert λ_death >= 0 "all λ's must be positive";
        @assert inv_area > 0 "`inv_area` must be positive";
        @assert λ_obs_spat_I > 0 "`λ_obs_spat_I` must be positive";

        constdiag = isconstdiag(Σ_obs);
        _log_inv_area = log(inv_area);
        _tildeC = inv_area * λ_obs_spat_I;
        new{N, N * N, constdiag}(λ_obs, λ_clutter, λ_birth,
                                 λ_birth0, λ_death, Σ_obs,
                                 inv_area, λ_obs_spat_I, _tildeC, _log_inv_area);
    end
end

function WolftrackParameters(;λ_obs::AFloat,
                             λ_clutter::AFloat,
                             λ_birth::AFloat,
                             λ_birth0::AFloat,
                             λ_death::AFloat,
                             Σ_obs::AMat{<: AFloat},
                             inv_area::AFloat,
                             λ_obs_spat_I::AFloat = inv(inv_area))
    WolftrackParameters(λ_obs, λ_clutter, λ_birth, λ_birth0,
                        λ_death, Σ_obs, inv_area, λ_obs_spat_I);

end

import Base.show
"""
Pretty printing for `WolftrackParameters` objects.
"""
function Base.show(io::IO, θ::WolftrackParameters)
    println(io, "A WolftrackParameters object with values:");
    for field in fieldnames(WolftrackParameters)
        field_str = string(field);
        startswith(field_str, "_") && continue;
        v = getfield(θ, field);
        if startswith(field_str, "log_")
            println(io, string(field_str[5:end], " => ", exp(v)));
        else
            println(io, string(field_str, " => ", v));
        end
    end
    nothing;
end

"""
An observation type.
Fields:
- `time`: the time of the observation.
- `dt`: the time interval associated with the observation.
- `location`: the location of the observation.
"""
struct WolfpackObs{N}
    time::Float64
    dt::Float64
    location::SVector{N, Float64}
    function WolfpackObs(location::AVec{<: AFloat}, time::AFloat, dt::AFloat)
        @assert length(location) > 0 && length(location) <= 2 "`length(location)` must be 1 or 2."
        @assert dt >= 0 "dt must be non-zero"
        @assert time >= 0.0 "`time` must be non-negative";
        new{length(location)}(time, dt, location);
    end
end

"""
An observation type with no location.
"""
struct WolfpackMisObs
    time::Float64
    dt::Float64
    function WolfpackMisObs(time::Real, dt::Real)
        @assert time >= 0.0 "`time` must be non_zero";
        @assert dt >= 0 "dt must be non-negative";
        new(time, dt);
    end
end

"""
An observation type alias.
"""
const Observation{N} = Union{WolfpackMisObs, WolfpackObs{N}};

function timeof(o::Observation)
    o.time;
end

"""
An abstract type representing different representations of wolfpacks.
"""
abstract type AbstractWolfpack{N} end

"""
A type representing the knowledge about the location of a wolfpack.
If newborn = true, then the location is uniformly distributed, and otherwise
normally with mean `mean` and covariance `cov`.
"""
struct Wolfpack{N, L} <: AbstractWolfpack{N}
    mean::MVector{N, Float64}
    cov::MMatrix{N, N, Float64, L}
    newborn::Base.RefValue{Bool}
    function Wolfpack(mean::AVec{<: AFloat}, cov::AMat{<: AFloat})
        N = length(mean);
        L = N * N;
        E = eltype(mean);
        @assert N == size(cov, 1) == size(cov, 2) "dimensions do not match.";
        new{N, L}(convert(MVector{N, Float64}, mean),
                  convert(MMatrix{N, N, Float64, L}, cov),
                  Base.RefValue(true));
    end
end
@inline function Wolfpack{N, L}() where {N, L}
    l = L; n = N;
    @assert l == n * n "invalid type parameters N` and `L`";
    Wolfpack(zeros(MVector{N, Float64}),
             zeros(MMatrix{N, N, Float64, L}));
end
@inline function Wolfpack{N}() where N
    Wolfpack(zeros(MVector{N, Float64}),
             zeros(MMatrix{N, N, Float64, N * N}));
end
@inline function dimension(::Type{<: Wolfpack{N}}) where N
    N;
end
@inline function dimension(pack::Wolfpack{N}) where N
    dimension(typeof(pack));
end
import Base.copy!
@inline function copy!(dest::Wolfpack{1}, src::Wolfpack{1})
    @inbounds dest.mean[1] = src.mean[1];
    @inbounds dest.cov[1] = src.cov[1];
    @inbounds dest.newborn[] = src.newborn[];
    nothing;
end
@inline function copy!(dest::Wolfpack{2}, src::Wolfpack{2})
    @inbounds dest.mean[1] = src.mean[1];
    @inbounds dest.mean[2] = src.mean[2];
    @inbounds dest.cov[1] = src.cov[1];
    @inbounds dest.cov[2] = src.cov[2];
    @inbounds dest.cov[3] = src.cov[3];
    @inbounds dest.cov[4] = src.cov[4];
    @inbounds dest.newborn[] = src.newborn[];
    nothing;
end

import Distributions.cdf
function cdf(pack::Wolfpack{1}, l::AFloat, u::AFloat, θ::WolftrackParameters, obslevel::Bool)
    if pack.newborn[]
        # This is approximate at the borders.
        return (u - l) * θ.inv_area;
    else
        if obslevel
            sd = sqrt(pack.cov[1, 1] + θ.Σ_obs[1, 1]);
        else
            sd = sqrt(pack.cov[1, 1]);
        end
        return exp(logdiffcdf(Normal(pack.mean[1], sd), u, l));
    end
end

import Distributions.pdf
function pdf(pack::Wolfpack{1}, θ::WolftrackParameters, x::Real)
    if pack.newborn[]
        return θ.inv_area;
    else
        return @inbounds pdf(Normal(pack.mean[1], sqrt(pack.cov[1, 1])), x);
    end
end

"""
Return a reference to the mean vector of the pack.
"""
@inline function ref_to_mean(pack::Wolfpack{N}) where N
    pack.mean;
end

"""
A wolfpack type optimised for constant diagonal covariance matrices.
If the location covariance of a 2D wolfpack is guaranteed to be
constant diagonal, only 3 floats are needed to describe all the information
about the pack. In the 1D case the memory requirement is the same as with
the type Wolfpack, so construction is possible only for N = 2.
"""
struct WolfpackConstDiag{N, L} <: AbstractWolfpack{N}
    data::MVector{L, Float64}
    newborn::Base.RefValue{Bool}
    function WolfpackConstDiag(mean::AVec{<: AFloat}, covdiag::AFloat)
        @assert covdiag >= 0.0 "`covdiag` must be >= 0.0";
        @assert length(mean) == 2 "`length(mean)` must be 2.";
        data = MVector{3, Float64}(mean[1], mean[2], covdiag);
        new{2, 3}(data, Ref(true));
    end
end
@inline function WolfpackConstDiag{2, 3}()
    WolfpackConstDiag(zeros(MVector{2, Float64}), 0.0);
end
@inline function WolfpackConstDiag{2}()
    WolfpackConstDiag{2, 3}();
end
@inline function copy!(dest::WolfpackConstDiag{2}, src::WolfpackConstDiag{2})
    copy!(dest.data, src.data);
    dest.newborn[] = src.newborn[];
    nothing;
end
@inline function dimension(::Type{<: WolfpackConstDiag{2}})
    2;
end
@inline function dimension(pack::WolfpackConstDiag{2})
    dimension(typeof(pack));
end

function cdf(pack::WolfpackConstDiag{2}, x_l::AFloat, x_u::AFloat,
             y_l::AFloat, y_u::AFloat, θ::WolftrackParameters, obslevel::Bool)
    if pack.newborn[]
        return (x_u - x_l) * (y_u - y_l) * θ.inv_area;
    else
        if obslevel
            sd = sqrt(pack.data[3] + θ.Σ_obs[1, 1]);
        else
            sd = sqrt(pack.data[3]);
        end
        return exp(logdiffcdf(Normal(pack.data[1], sd), x_u, x_l) +
                   logdiffcdf(Normal(pack.data[2], sd), y_u, y_l));
    end
end

"""
Return a reference to the mean vector of the pack.
"""
@inline function ref_to_mean(pack::WolfpackConstDiag{N}) where N
    view(pack.data, 1:N);
end

"""
Any kind of 2D wolfpack.
"""
const Wolfpack2D = Union{WolfpackConstDiag{2, 3}, Wolfpack{2, 4}};


"""
A type for wolf pack particles.

Type parameters:
-`WPType`: The type used to represent the wolfpacks.

Fields:
-`N`: The number of packs in the particle. This value is a reference to an
Int.
-`packs`: Vector of wolf packs. The order of the packs holds no meaning.
"""
struct WolfpackParticle{WPType <: AbstractWolfpack}
    N::Base.RefValue{Int}
    packs::Vector{WPType}
    function WolfpackParticle(wptype::Type{<: AbstractWolfpack},
                              npacks::Integer; extra::Integer = 10)
        packs = [wptype() for i in 1:npacks];
        new{wptype}(Ref(npacks), packs);
    end
end

function Base.show(io::IO, wp::WolfpackParticle{WPType}) where WPType
    print(io, string(dimension(WPType), "D WolfpackParticle (N = ",
                     string(wp.N[]), ", capacity = ", string(length(wp.packs)), ")"));
end

"""
Get the dimension of a particle.
"""
dimension(::Type{WolfpackParticle{WPType}}) where WPType = dimension(WPType);
dimension(wp::WolfpackParticle) = dimension(typeof(wp));

"""
Get the element type of a particle.
"""
Base.eltype(::Type{<: WolfpackParticle}) = Float64;

@inline function _copy_no_dim_check!(dest::WolfpackParticle, src::WolfpackParticle)
    for i in 1:src.N[]
        @inbounds copy!(dest.packs[i], src.packs[i]);
    end
    @inbounds dest.N[] = src.N[];
    nothing;
end

"""
A function that transfers the information of a source particle to
a destination particle.
Arguments:
- `dest`: the particle on which information is overwritten.
- `src`: the particle from which we extract information.
"""
function copy!(dest::WolfpackParticle{WPType},
               src::WolfpackParticle{WPType}) where WPType
    # Compare the length of the destination particle means vector
    # with the new number of particles and decide whether it is
    # necessary to create new slots.
    if length(dest.packs) < src.N[]
        for i in 1:(src.N[] - length(dest.packs))
            push!(dest.packs, WPType());
        end
    end
    _copy_no_dim_check!(dest, src);
    nothing
end

"""
Function drops unused "preallocated pack storage" from particle.
"""
function drop_unused!(wp::WolfpackParticle)
    n_unused = length(wp.packs) - wp.N[];
    drop_index = wp.N[] + 1;
    for i in 1:n_unused
        splice!(wp.packs, drop_index);
    end
    nothing;
end

"""
A struct keeping preallocated memory required internally for weight computations.

Fields:
`size`: Dimensions of preallocated arrays.
`lltmp`: Temporary vector keeping likelihood values related to observation association
for each pack.
`λ_obs_x_pre_tmp`: Temporary vector for storing λ_obs values varying in space.
`λ_obs_x_assoc_tmp`: Temporary vector for storing "association correction terms"
related to spatially varying λ_obs.
`prob_table`: A preallocated table used when drawing samples from the one step
optimal proposal distribution.
"""
struct ProposalDistStorage{N}
    size::Int
    lltmp::Vector{Float64}
    λ_obs_x_pre_tmp::Vector{Float64}
    λ_obs_x_assoc_tmp::Vector{Float64}
    meantmp::Vector{MVector{N, Float64}}
    prob_table::Matrix{Float64}
    function ProposalDistStorage(dim::Int, max_npacks::Integer)
        @assert max_npacks > 0 "`max_npacks` must be positive";
        lltmp = zeros(Float64, max_npacks + 2);
        λ_obs_x_pre_tmp = zeros(Float64, max_npacks + 2);
        λ_obs_x_assoc_tmp = zeros(Float64, max_npacks + 2);
        meantmp = [zero(MVector{dim, Float64}) for i in 1:(max_npacks + 2)];
        prob_table = zeros(Float64, max_npacks + 2, max_npacks + 2);
        new{dim}(max_npacks + 2, lltmp, λ_obs_x_pre_tmp, λ_obs_x_assoc_tmp,
                 meantmp, prob_table);
    end
end

@inline function get_max_npacks(pds::ProposalDistStorage)
    pds.size - 2;
end


"""
A state object used in the wolfpack filter for storing the latest filtering
distribution. The distribution is represented by Npar number of
particles of type WolfpackParticle{WPType} and their weights.
The storage object also keeps other preallocated memory used in the resampling
step as well as in building and sampling the optimal proposal distribution.

Type parameters:
`WPType`: The type used to represent wolf packs.
`NT`: The number of threads used in the computations.

Fields:
`X`: A vector of particles keeping the current filtering distribution.
`W`: A vector storing the latest weights.
`A`: A vector of resampling indices. These are used in the resampling step to
select the particles that get to live to the next time index.
`pds`: A vector of storage objects used internally for sampling from the one step
optimal proposal distribution. The length of this vector corresponds to the number
of threads used, since each thread needs its own memory for the computations.
`V`: A vector used in the computation of the loglikelihood approximation of the
particle filter.
`fi`: (Filtered index) A pointer to an Int which specifies which column of `X` currently
holds the filtered particles (i.e X[:, fi] and the vector W make up the approximate
filtering distribution)
"""
struct WolfpackFilterState{WPType <: AbstractWolfpack, N, NT}
    X::Matrix{WolfpackParticle{WPType}}
    W::Vector{Float64}
    A::Vector{Int}
    pds::Vector{ProposalDistStorage{N}}
    V::Vector{Float64}
    fi::Base.RefValue{Int}
    function WolfpackFilterState(v::AVec{WolfpackParticle{WPType}},
                                 W::AVec{<: AFloat};
                                 max_npacks::Int = 100,
                                 n_threads::Integer = -1) where WPType
        @assert !isempty(v) "`v` must not be empty.";
        @assert !isempty(W) "`W` must not be empty.";
        @assert max_npacks > 0 "`max_npacks` must be > 0.";
        @assert isapprox(sum(W), 1.0) "`W` must be normalised.";

        npar = length(v);
        X = hcat(v, deepcopy(v)); # Ensure proper copy.
        W = copy(W);
        A = zeros(Int, npar);
        V = zeros(Float64, npar);
        fi = Ref(1);
        dim = dimension(eltype(v));
        if n_threads <= 1
            # No multithreading.
            pds = [ProposalDistStorage(dim, max_npacks)];
        else
            # Multithreading enabled.
            n_threads = min(Threads.nthreads(), n_threads);
            for i in 2:n_threads
                Random.default_rng(i);
            end
            pds = [ProposalDistStorage(dim, max_npacks) for i in 1:n_threads];
        end
        new{WPType, dim, length(pds)}(X, W, A, pds, V, fi);
    end
end

function wolfpacktype(fs::WolfpackFilterState{WPType}) where WPType
    WPType;
end

@inline function get_npar(fs::WolfpackFilterState)
    size(fs.X, 1);
end
@inline function get_max_npacks(fs::WolfpackFilterState)
    get_max_npacks(fs.pds[1]);
end

function WolfpackFilterState(Npacks::AVec{<: Integer}, W::AVec{<: AFloat},
                             θ::WolftrackParameters{N, L, B};
                             n_threads::Int = 1, max_npacks::Int = 100) where {N, L, B}
    npar = length(Npacks);
    @assert npar == length(W) "the length of `N` and `W` must match";
    WPType = determine_wolfpacktype(N, B);
    pv = Vector{WolfpackParticle{WPType}}(undef, npar);
    for i in 1:npar
        pv[i] = WolfpackParticle(WPType, Npacks[i]);
    end
    WolfpackFilterState(pv, W; n_threads = n_threads,
                        max_npacks = max_npacks);
end

"""
A container object for keeping the filtering distributions for time indices
1, ..., T. The approximation to the filtering distribution consists of the
particles and weights at time index k.

Type parameters:
`WPType`: The type used to represent wolfpacks.

Fields:
`time`: A vector of times which gives the time instant of each filtering distribution.
`X`: A matrix of WolfpackParticles. The particles at time index k are found at
`X[:, k]`.
`W`: A matrix of weights. The weights at time index k are found at `W[:, k]`.
`A`: A matrix of ancestor indices.
"""
struct WolfpackFilterOutput{WPType <: AbstractWolfpack}
    time::Vector{Float64}
    X::Matrix{WolfpackParticle{WPType}}
    W::Matrix{Float64}
    A::Matrix{Int}
    function WolfpackFilterOutput(time::AVec{<: AFloat},
                                  X::Matrix{WolfpackParticle{WPType}},
                                  W::Matrix{<: AFloat},
                                  A::Matrix{<: Integer}) where WPType
        @assert length(time) == size(X, 2) "length of `time` must match the column dimension of `X`";
        @assert size(X) == size(W) "dimensions of `X` and `W` must match.";
        @assert size(A, 1) == size(W, 1) "invalid row dimension for matrix `A`";
        @assert size(A, 2) == size(W, 2) - 1 "invalid column dimension for matrix `A`";
        new{WPType}(time, X, W, A);
    end
end

function WolfpackFilterOutput{WPType}(npar::Int, time::AVec{<: AFloat}; n_packs::Int = 10) where WPType
    @assert npar > 1 "`npar` must be > 1.";
    @assert n_packs > 0 "`n_packs` must be > 0.";
    @assert issorted(time) "`time` must be an increasing sequence.";

    X = Matrix{WolfpackParticle{WPType}}(undef, npar, length(time));
    W = zeros(Float64, npar, length(time));
    for i in eachindex(X)
        X[i] = WolfpackParticle(WPType, n_packs; extra = 0);
    end
    A = zeros(Int, npar, length(time) - 1);
    WolfpackFilterOutput(time, X, W, A);
end

@inline function get_npar(fo::WolfpackFilterOutput)
    size(fo.X, 1);
end

@inline function dimension(fo::WolfpackFilterOutput{WPType}) where WPType
    dimension(WPType);
end

Base.size(fo::WolfpackFilterOutput) = size(fo.X);
Base.size(fo::WolfpackFilterOutput, dim::Integer) = size(fo.X, dim);

"""
Drop preallocated, unused pack storage.
"""
function drop_unused!(fo::WolfpackFilterOutput)
    for particle in fo.X
        drop_unused!(particle);
    end
    nothing;
end

include(joinpath(LIB_PATH, "simulation-types.jl"));
