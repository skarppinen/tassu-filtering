using DataFrames
using Random

"""
A structure keeping four "function types" `birth`, `death`, `obs` and `spat`.
The first three can be used to specify a timevarying intensity for the birthrate,
deathrate and observation rate, respectively. The fourth field `spat` can be
used to specify a spatially varying observation rate.

It is assumed that each of the fields can be "called like a function",
i.e the call `birth(t)` would work, for example.
All fields should return a scalar value (the intensity) when called.
The first three should accept a scalar (time) as input, and the
last one an abstract vector of length one or two depending on the dimension
of the model.
"""
struct IntensityCovariates{BFun, DFun, OFun, SpatFun}
   birth::BFun
   death::DFun
   obs_t::OFun
   obs_x::SpatFun
end

"""
Constructor for IntensityCovariates. By default, each covariate will be constant
one, which corresponds to no covariates.
"""
function IntensityCovariates(;birth = build_constant(1.0),
                              death = build_constant(1.0),
                              obs_t = build_constant(1.0),
                              obs_x = build_constant(1.0))
   IntensityCovariates(birth, death, obs_t, obs_x);
end

"""
Build a constant function that returns the argument `v` with all inputs.
"""
function build_constant(v::Real)
   f = let v = v
      function(x)
         v;
      end
   end
   f;
end

"""
A struct representing a piecewise constant function on [0, `T`), where `T = t[end]`.
The function is zero outside this region. The function takes the value `y[i]` on the
interval `[t[i], t[i + 1])`.
"""
struct PiecewiseConstant <: Function
    t::Vector{Float64}
    y::Vector{Float64}
    function PiecewiseConstant(t::AVec{<: AFloat}, y::AVec{<: Real})
        @assert issorted(t) "`t` must be sorted.";
        @assert allunique(t) "all elements in `t` must be unique";
        @assert length(t) >= 2 "`length(t)` must be >= 2.";
        @assert length(t) - 1 == length(y) "`length(y)` must equal `length(t) - 1`";
        new(t, y);
    end
end

const PWC = PiecewiseConstant;

"""
Evaluate the piecewise constant function at the point `t`.
This function makes it possible to "call" the object "like a function", i.e
with code such as `pc(5.0)`, where `pc` is a `PiecewiseConstant` object.
"""
function (pc::PiecewiseConstant)(t::Real)
    i = searchsortedlast(pc.t, t); # Note: t's are unique, so sslast is ok.
    if i <= 0 || t >= pc.t[end]
        return zero(t);
    end
    typeof(t)(pc.y[i]);
end

"""
Return a piecewise constant function that has one interval `range` at which
the function takes the value `v`. If the value is omitted, the function is constant
one on `range`.
"""
function PiecewiseConstant(range::NTuple{2, Float64}, v::Real = 1.0)
    @assert range[1] < range[2] "invalid range, lower bound >= upper bound";
    PiecewiseConstant([range[1], range[2]], [Float64(v)]);
end

function domain(pc::PiecewiseConstant)
    (pc.t[1], pc.t[end])
end
function indomain(pc::PiecewiseConstant, x::Real)
    d = domain(pc);
    within(d, x);
end
function within(r::Tuple{<: Real, <: Real}, x::Real)
    r[1] <= x < r[2];
end

"""
Function returns the sorted vector of times `t` such that both piecewise
constant functions `pc1` and `pc2` are constant on each interval
`[t[i], t[i + 1])`.
"""
function constant_intervals(pc1::PiecewiseConstant, pc2::PiecewiseConstant)
    sort!(unique(vcat(pc1.t, pc2.t)));
end

# Define + and * operations for PiecewiseConstant objects.
import Base.*, Base.+
for op in (:+, :*)
    let
        fs = :(begin
            function $op(pc1::PiecewiseConstant, pc2::PiecewiseConstant)
                t = constant_intervals(pc1, pc2);
                y = zeros(length(t) - 1);
                for i in 1:length(y)
                    y[i] = $op(pc1(t[i]), pc2(t[i]));
                end
                PiecewiseConstant(t, y);
            end
            function $op(λ::Real, pc::PiecewiseConstant)
                PiecewiseConstant(copy(pc.t), broadcast($op, λ, pc.y));
            end
            function $op(pc::PiecewiseConstant, λ::Real)
                $op(λ, pc);
            end
        end)
        eval(fs);
    end
end

import Base.maximum;
maximum(pc::PiecewiseConstant) = maximum(pc.y);

"""
Approximate the function `f` with a piecewise constant function on the domain
`range` using `n` intervals. The return value is an object of type PiecewiseConstant
implementing the approximation.
"""
function piecewise_constant_approximator(f, range::NTuple{2, Float64}, n::Integer)
    @assert range[1] < range[2] "invalid range, lower bound >= upper bound";
    @assert n >= 2 "`n` must be at least 2";
    t = collect(LinRange(range[1], range[2], n));
    y = zeros(length(t) - 1);

    fprev = f(t[1]); fnext = f(t[2]);
    for i in 1:length(y)
        y[i] = (fprev + fnext) / 2.0;
        fprev = fnext;
        fnext = f(t[i + 1]);
    end
    PiecewiseConstant(t, y);
end


"""
Build a piecewise function from a DataFrame `d`.
`d` should contain exactly two columns, where the first column
contains the increasing timestamps at which the value of the function changes,
and the second column contains the value the piecewise function gets at each
interval. The last value of the first column should specify the upper bound of
the piecewise function's domain.

For example, the call:
``
d = DataFrame(t = [0.0, 1.0, 10.0], value = [1, 2])
f = build_piecewise(d);
``
will produce a function object of type PiecewiseConstant.
This represents a function that takes the value 1 on the interval [0.0, 1.0) and
the value 2 on the interval [1.0, 10.0). The function takes the value 0 outside
[0.0, 10.0).
"""
function build_piecewise_constant(d::DataFrame)
   check_xy_dataframe(d, colnames = (:time, :y));
   t = d[:, :time]; # Time column.
   ymis = d[:, :y]; # Value column with last value missing.
   PiecewiseConstant(t, y);
end

"""
Validate a DataFrame containing two columns of xy data. The other column
is not checked for type since it might contain missing data.
"""
function check_xy_dataframe(d::DataFrame; colnames::NTuple{2, Symbol} = (:time, :y))
   nm = Symbol.(names(d));
   @assert size(d, 2) == 2 "the input DataFrame must have exactly two columns.";
   @assert size(d, 1) >= 1 "the input DataFrame must not be empty.";
   @assert colnames[1] in nm "the input data must contain the column `$(string(colnames[1]))`.";
   @assert colnames[2] in nm "the input data must contain the column `$(string(colnames[2]))`.";
   @assert issorted(d[!, colnames[1]]) "the column `$(string(colnames[1]))` should be increasing.";
   for i in 1:2
       @assert eltype(d[!, colnames[i]]) <: Real "the column `$(string(colnames[i]))` must be real valued.";
   end
end

function sim_given_packs(wph, θ, covariates, bs)

    Nu = evolution_of_N(wph);
    @assert domain(Nu) == domain(covariates.obs_t) "invalid domains";
    λ_obs_x_max = maximum(covariates.obs_x.r);
    λ_obs = θ.λ_obs;
    λ_clutter = θ.λ_clutter;
    densmax = 4.0 * pi * pi * det(θ.Σ_obs);
    inv_area = θ.inv_area;
    t_bounds = bs[1]; t = t_bounds[1]; T = t_bounds[2];
    x_bounds = bs[2];
    y_bounds = bs[3];
    location_dist = product_distribution([Uniform(x_bounds[1], x_bounds[2]),
                                          Uniform(y_bounds[1], y_bounds[2])]);
    σobs = sqrt(θ.Σ_obs[1, 1]);

    # Construct majoring piecewise constant function to simulate from.
    area = (x_bounds[2] - x_bounds[1]) * (y_bounds[2] - y_bounds[1]);
    cNu = λ_obs * (λ_obs_x_max / densmax);
    inner = (cNu * Nu) + λ_clutter * inv_area * λ_obs_x_max;
    sim_pc = (area * covariates.obs_t) * inner;

    t_knots = copy(sim_pc.t); # Points at which intensity changes.
    next_i_knot = 2; # Index of next "knot" changepoint.

    out = Vector{NTuple{3, Float64}}(undef, 0);
    y = zeros(2); # Temp for locations.
    while t < T
        next_t_knot = t_knots[next_i_knot];
        λ = sim_pc(t);
        tnew = draw_event(λ, t);

        if tnew < next_t_knot

            # Simulate location and check acceptance (thinning).
            rand!(location_dist, y);

            p = λ_obs_tot(tnew, y[1], y[2], wph, θ, covariates) / λ;
            if rand() <= p
                push!(out, (tnew, y[1], y[2]));
            end
            t = tnew;

        else
            t = next_t_knot;
            next_i_knot += 1;
        end
    end
    out;
end
