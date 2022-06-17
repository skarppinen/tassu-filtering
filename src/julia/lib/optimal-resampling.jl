include(joinpath("../../../config.jl"));
include(joinpath(LIB_PATH, "wolfpack-types.jl"));

"""
Struct for representing ranges in a vector.
"""
struct WriteRanges
    lo::Vector{Int}
    hi::Vector{Int}
    function WriteRanges(N::Integer)
        @assert N > 0 "`N` must be > 0.";
        lo = zeros(Int, N);
        hi = zeros(Int, N);
        new(lo, hi);
    end
end

"""
Initialise the ranges in the WriteRanges object based on a vector of (positive)
integers in the case of a non-missing observation.
No bounds checking is done, it is assumed that the WriteRanges object
`wr` has enough space. Passing in the vector `[1, 2, 3]` as the second argument
would set the first three ranges in `wr` to 1:9, 10:25, 26:50.
"""
function set_ranges!(wr::WriteRanges, npacksvec::AVec{<: Integer}, obstype::Type{<: Observation})
    l = u = 0;
    for i in 1:length(npacksvec)
        wr.lo[i] = l = u + 1;
        wr.hi[i] = u = u + prob_table_elems(npacksvec[i], obstype);
    end
    nothing;
end

function set_ranges!(wr::WriteRanges, fs::WolfpackFilterState,
                     obstype::Type{<: Observation})
    l = u = 0;
    fi = fs.fi[];
    for i in 1:get_npar(fs)
        wr.lo[i] = l = u + 1;
        wr.hi[i] = u = u + prob_table_elems(fs.X[i, fi], obstype);
    end
    nothing;
end

import Base.getindex;
function getindex(wr::WriteRanges, i::Integer)
    wr.lo[i]:wr.hi[i];
end

import Base.length;
function length(wr::WriteRanges, i::Integer)
    length(wr[i]);
end

function get_range_and_elem_index(wr::WriteRanges, i::Integer)
    i > wr.hi[end] && return (-1, i);
    range_i = 1; passed = 0;
    while !(wr.lo[range_i] <= i <= wr.hi[range_i])
        passed += wr.hi[range_i] - wr.lo[range_i] + 1;
        range_i += 1;
    end
    range_i, i - passed;
end

"""
Optimal resampling object for the discrete PF.

Fields:
`q_max`: A preallocated vector for storing all weights associated with particles
and the discrete possibilites that can occur to them. The vector has length R * N,
where R is the maximal amount (given a maximal value for the number of packs) of
discrete possibilites that can occur to a single particle. (note that the number
of discrete possibilities is a deterministic function of the pack count)

`qA_max`: A preallocated vector for storing indices that map a weight with the
same index to a particular outcome (within some particle's probability table).

`wr`: A WriteRanges object that specifies which indices in `q_max` and `qA_max`
are associated with which particle. For example the range `wr.lo[i]:wr.hi[i]`
gives the range for the weights associated with particle i.
"""
struct OptimalResampling <: Resampling
    q_max::Vector{Float64}
    qA_max::Vector{Int}
    wr::WriteRanges
    function OptimalResampling(npar::Integer, max_npacks::Integer)
        @assert max_npacks > 0 && npar > 0 "strictly positive inputs requested.";
        R = (max_npacks + 2) * (max_npacks + 2);
        M = R * npar;
        q_max = zeros(M);
        qA_max = zeros(Int, M);
        wr = WriteRanges(npar);
        new(q_max, qA_max, wr);
    end
end

"""
Finish optimal resampling assuming that the cutoff value 1/c and the number
of weights >= 1/c, `L` has been found.

The input is the discrete probability distribution (`qA`, `q`) (support of length M).
It is assumed that `q` and `qA` are ordered such that the `L` weights >= 1/c are at the end.
Here `qA` gives the outcomes (integers) and `q` the weights of each outcome.

The output is the discrete probability distribution (`wA`, `w`) (support of length N < M).
Here, `wA` gives the outcomes (integers), and `w` the weights of each outcome.

Arguments:
`w`: Output vector of weights (length N).
`wA`: Output vector of outcomes (integers, length N).
`q`: Input vector of weights M (> N).
`qA`: Input vector of outcomes.

`inv_c`: The cutoff value for the weights in `q`.
`L`: Integer stating how many weights in `q` are >= 1/c. (these are kept in the resampling)
"""
function _optimal_resample!(w::AVec{<: AFloat}, wA::AVec{<: Integer},
                            q::AVec{<: AFloat}, qA::AVec{<: Integer},
                            inv_c::AFloat, L::Integer)
    N = length(w); M = length(q);
    K = 1;
    u_ = rand();
    i_keep_start = M - L + 1; # The index in `q` where the weights >= inv_c start.
    i_sys_end = M - L; #i_keep_start - 1; # The index in `q` where the stratified resampling partition ends.
    n_sys = N - L; # The amount of values to sample using systematic resampling.

    # Normalise the first `i_sys_end` weights of `q` (that are less than `inv_c`).
    # Then sample `n_sys` from `qA` with systematic resampling.
    view_sys = view(q, 1:i_sys_end);
    normalize!(view_sys, 1);
    S = @inbounds view_sys[1];
    for j in 1:n_sys
        u = (j - 1 + u_) / n_sys;
        while K < i_sys_end && u > S
            K += 1;
            @inbounds S += view_sys[K];
        end
        @inbounds wA[j] = qA[K]; # Note: here we have to pick qA[K], not K!
        @inbounds w[j] = inv_c; # Set `inv_c` as the new weight.
    end

    # The remaining particles (with weights >= inv_c) keep their weights and
    # live on.
    i = n_sys + 1;
    for j in i_keep_start:M
        @inbounds wA[i] = qA[j];
        @inbounds w[i] = q[j];
        i += 1;
    end
    nothing;
end

function swap!(A::AVec, l::Integer, u::Integer)
    @inbounds tmp = A[l];
    @inbounds A[l] = A[u];
    @inbounds A[u] = tmp;
    nothing;
end

function partition!(X::AVec, A::AVec{Int}, l::Integer = 1, u::Integer = length(X))
    x = @inbounds X[u];
    i = l - 1;
    for j in l:(u-1)
        if @inbounds X[j] <= x
            i += 1;
            swap!(X, i, j); swap!(A, i, j);
        end
    end
    out_i = i + 1;
    swap!(X, out_i, u); swap!(A, out_i, u);
    out_i;
end

function random_partition!(X::AVec, A::AVec{Int}, l::Integer = 1, u::Integer = length(X))
    i = rand(l:u);
    swap!(X, u, i); swap!(A, u, i);
    partition!(X, A, l, u);
end

"""
Helper function to construct a vector of weights such that
there is at least one weight which satisfies the function `condition` within
this function. This means that the value for c in the optimal resampling algorithm
is equal to (N - a_κ) / b_κ for some κ among the sampled weights this function returns.
"""
function sim_weights_with_valid_kappa(M::Integer, N::Integer)
    @assert M > N "bad M and N";
    function condition(κ::Float64, q::Vector, N::Int)
        sum(map(x -> min(1.0, x / κ), q)) <= N
    end
    found = false;
    num = 0;
    q = zeros(M)
    while !found
        rand!(q); normalize!(q, 1);
        for κ in q
            if condition(κ, q, N)
                found = true;
                num += 1;
            end
        end
    end
    q, num;
end

"""
Find the value of 1/c for the discrete particle filter given normalised
weights `q`. Prior to running this function, `A` should be initialised to
the equivalent of `collect(1:length(q))`.
`N` is the amount of particles we use to approximate a discrete distribution
with M (> N) outcomes. Alongside 1/c, the number of weights >= 1/c, L,
is returned.
"""
function find_inv_c!(q::AVec{<: AFloat}, A::AVec{<: Integer}, N::Integer)
    M = length(q); i = zero(typeof(N)); l = typeof(N)(1); u = M;
    a_κ = b_κ = b_κ_lm1 = zero(Float64); # b_κ_lm1 is the sum of weights up to l - 1.
    while l < u || i != u
        # Get trial value for κ.
        i = random_partition!(q, A, l, u);
        κ = @inbounds q[i];

        # Test condition.
        b_κ = b_κ_lm1 + sum(view(q, l:(i - 1)));
        a_κ = M - i + 1.0;
        if inv(κ) * b_κ + a_κ <= N
            # κ is in the set we want to find the minimum of,
            # so decrease search range from above.
            u = i;

            # Handle cases where there are multiple same values.
            while @inbounds isapprox(q[u], q[u - 1])
                u -= 1;
            end
            #u - 1 == l && isapprox(q[u], q[l]) && break;
        else
            # κ is not in the set, need to increase κ, so decrease search
            # range from below. Accumulate also b_κ_lm1 to reduce computation
            # in future iterations.
            l = i + 1;
            b_κ_lm1 = b_κ + κ;
        end
    end
    # Return inv(c) and number of elements that have weights >= inv(c).
    # Note that these L elements are found in the end of the vector `q`, i.e
    # from indices (M - L + 1):M. This is because `q` is mutated in place.
    out = if l == u
        inv((N - a_κ) / b_κ), typeof(N)(a_κ)
    else
        inv(N), typeof(N)(0);
    end
    out;
end

"""
Allocating version of optimal resampling for testing purposes.
"""
function optimal_resample(q::AVec{<: AFloat}, N::Integer)
    @assert isapprox(sum(q), 1.0) "`q` should be normalised.";
    q = copy(q);
    qA = collect(1:length(q));
    inv_c, L = find_inv_c!(q, qA, N);
    w = zeros(N);
    wA = zeros(Int, N);
    _optimal_resample!(w, wA, q, qA, inv_c, L);
    w, wA, inv_c, L
end
