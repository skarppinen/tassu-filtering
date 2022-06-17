using StaticArrays
include(joinpath(LIB_PATH, "type-aliases.jl"))

"""
A type representing the lifetime of a single wolfpack.

Fields:
- `birth_time`: a Float64 indicating the birth time of the pack.
- `death_time`: a Float6464 indicating the death of the pack.
- `location` : a MVector storing the location of the pack.
"""
mutable struct WolfpackLifetime{N}
    birth_time::Float64
    death_time::Float64
    location::MVector{N, Float64}
    function WolfpackLifetime(birth_time::Real, death_time::Real, location::AVec{<: AFloat})
        N = length(location);
        @assert birth_time < death_time "death cannot occur before birth.";
        @assert 0 < N < 3 "N must be 1 or 2.";
        new{N}(birth_time, death_time, convert(MVector{N, Float64}, location));
    end
end

function WolfpackLifetime(birth_time::Real, death_time::Real, N::Int64)
    location = zeros(MVector{N, Float64});
    @assert birth_time < death_time "Birth and death dates are inconsistent. Death cannot occur before birth."
    @assert 0 < N < 3 "N must be 1 or 2."
    WolfpackLifetime(birth_time, death_time, location);
end


"""
Get the dimension of a WolfpackLifetime object.
"""
dimension(::WolfpackLifetime{N}) where {N} = N;


"""
The "history" of several wolfpacks that lived during the time interval
[`t_start`, `t_end`].
"""
struct WolfpackHistory{N}
    t_start::Float64
    t_end::Float64
    lifetimes::Vector{WolfpackLifetime{N}}
    function WolfpackHistory(t_start::Real, t_end::Real,
                             lt::Vector{WolfpackLifetime{N}}) where {N}
        @assert t_start < t_end "the end has come before the beginning.";
        new{N}(t_start, t_end, lt);
    end
end

function dimension(wph::WolfpackHistory{N}) where N
    N;
end


"""
The function returns the evolution of the pack size N as a piecewise constant
function, given a history of wolfpacks.
The output of the function is an object of type PiecewiseConstant.

Arguments:
`wph`: A WolfpackHistory object.
"""
function evolution_of_N(wph::WolfpackHistory{N}) where N
   t_start = wph.t_start; t_end = wph.t_end; lt = wph.lifetimes;
   length(lt) == 0 && (return PiecewiseConstant([t_start, t_end], [0]));

   # Build the time vector at which changes occured.
   times::Vector{Float64} = sort!(unique(vcat(map(x -> [x.birth_time, x.death_time], lt)...)));
   !(t_end in times) && push!(times, t_end);
   !(t_start in times) && pushfirst!(times, t_start);
   @assert issorted(times) "something happened when computing evolution of N, times not sorted!";

   # Build the number of N at each time point.
   n_packs = zeros(Int, length(times) - 1);
   for (i, t) in enumerate(times[1:(end - 1)])
      n_packs[i] = n_packs_alive(wph, t);
   end
   PiecewiseConstant(times, n_packs);
end

"""
Return the value of N at the time points defined by `grid`.
The function does a lookup to `pack_evolution` and `event_times` to get the
values.

Arguments:
`grid`: A vector giving the time points at which N should be returned.
`wph`: A WolfPackHistory object.
"""
function n_grid(grid::AVec{<: Real}, wph::WolfpackHistory)
   event_times, N = evolution_of_N(wph);
   out = zeros(Int, length(grid));
   for (i, p) in enumerate(grid)
      j = searchsortedlast(event_times, p);
      out[i] = N[j];
   end
   out;
end

"""
Find out whether the pack represented by a `WolfpackLifetime` is alive at time
`t`.
"""
function is_alive(lt::WolfpackLifetime, t::Real)
    lt.birth_time <= t < lt.death_time
end

"""
Return the number of packs alive at time `t`.
"""
function n_packs_alive(truth::AVec{<: WolfpackLifetime}, t::Real)
   n = 0;
   for pack in truth
       is_alive(pack, t) && (n += 1;)
   end
   n;
end

"""
Return the number of packs alive at time `t`.
"""
function n_packs_alive(wp_hist::WolfpackHistory, t::Real)
    n_packs_alive(wp_hist.lifetimes, t);
end

"""
The function returns the list of packs that are alive at a time `t`.

Arguments :
`pack_list`: A vector of WolfpackLifetime objects.
"""
function packs_alive(packs::Vector{<: WolfpackLifetime}, t::Real)
    alive = eltype(packs)[];
    for pack in packs
        is_alive(pack, t) && (push!(alive, pack);)
    end
    alive;
end
function packs_alive(wph::WolfpackHistory, t::Real)
    packs_alive(wph.lifetimes, t);
end
