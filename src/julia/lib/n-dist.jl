include("../../../config.jl");
include("wolfpack-types.jl");
using Distributions

"""
Function computes the discrete distribution of the number of packs at a
particular time instant.

Arguments:
- `weights` : The normalised weights.
- `particles`: The vector of particles.

Returns a DiscreteNonParametric object with the following attributes:
- `support`: the range of possible pack counts (across all particles)
- `p`: the probabilities of each pack count
"""
function get_discrete_distribution(weights::AVec{<: AFloat},
                                   particles::AVec{<: WolfpackParticle})

    n_particles = length(particles);
    ns = map(x -> x.N[], particles); # N values found in particles.
    uniq_ns = unique(ns);

    prob_vec_len = maximum(uniq_ns);
    prob_vec = zeros(prob_vec_len + 1); # + 1 for zero!

    # Populate probability table.
    for value in uniq_ns
        for (i, n) in enumerate(ns)
            if n == value
                prob_vec[value + 1] += weights[i];
            end
        end
    end
    DiscreteNonParametric(collect(0:prob_vec_len), prob_vec);
end

function get_discrete_distribution(weights::AMat{<: AFloat},
                                   particles::AMat{<: WolfpackParticle})
    map(1:size(particles, 2)) do i
        get_discrete_distribution(view(weights, :, i), view(particles, :, i));
    end
end

function get_discrete_distribution(fo::WolfpackFilterOutput{N}) where N
    get_discrete_distribution(fo.W, fo.X);
end

"""
Return the probability table of the distribution of N at each time point.
Rows = N goes from 0, 1, .. (top to bottom)
Columns = timepoints
"""
function dist_of_N(fo::WolfpackFilterOutput)
   dist_of_N(get_discrete_distribution(fo));
end

function dist_of_N(weights::AMat{<: AFloat},
                   particles::AMat{<: WolfpackParticle})
    dist_of_N(get_discrete_distribution(weights, particles));
end

function dist_of_N(v::AVec{<: DiscreteNonParametric})
    dimens = maximum(map(x -> length(x.support), v));
    z = zeros(dimens, length(v));
    for j in 1:size(z, 2)
       dist = v[j];
       for i in 1:length(dist.support)
          z[i, j] = dist.p[i];
       end
    end
    z;
end
