using LinearAlgebra
using Distributions

"""
An abstract type marking different possibilities of gating, i.e
reducing computation by cutting off very unlikely possibilities.
"""
abstract type Gating end

"""
No gating, i.e compute observation updates for every pack.
"""
struct NoGating <: Gating end

"""
Gating by reducing associating calculations by computing
the squared Mahalanobis distance (chi-square distributed) of the new observation
relative to N(mean location of wolf pack, Sigma_obs). If at least
one observation has been associated with a wolf pack, it is guaranteed that
The covariance of the location of the wolf pack is <= Sigma_obs. Hence,
Sigma_obs serves as a good candidate for gating. Here, one should set
K such that Sigma_obs = K * I.
"""
struct ChiSquareGating <: Gating
    dim::Int
    q::Float64 # The quantile.
    K::Float64 # Multiplier. `gating covariance` is K * I, where I identity.

    _cutoff::Float64 # The actual value to compare against.
    function ChiSquareGating(d::Integer, p::AFloat, K::AFloat)
        @assert 0 < p < 1 "`p` must be in (0, 1)";
        @assert K > 0 "`K` must be positive.";
        @assert d in (1, 2) "`dimension` must be 1 or 2.";
        q = quantile(Chisq(d), 1.0 - p);
        new(d, q, K, q * K);
    end
end

function ChiSquareGating(p::AFloat, θ::WolftrackParameters)
    ChiSquareGating(size(θ.Σ_obs, 1), p, maximum(θ.Σ_obs));
end

@inline function gates(csg::ChiSquareGating, mu::AVec{<: AFloat}, o::WolfpackObs{1})
    d = o.location[1] - mu[1];
    d * d > csg._cutoff;
end
@inline function gates(csg::ChiSquareGating, mu::AVec{<: AFloat}, o::WolfpackObs{2})
    d1 = o.location[1] - mu[1];
    d2 = o.location[2] - mu[2];
    d1 * d1 + d2 * d2 > csg._cutoff;
end
@inline function gates(ng::NoGating, mu::AVec{<: AFloat}, o::WolfpackObs{N}) where N
    false;
end

"""
Compute the gated loglikelihood, where the gating criterion is first checked.
If the gating criterion is true (i.e true likelihood is neglible), -Inf is returned.
Else the true likelihood is returned.
"""
@inline function gated_loglik!(mu::MVector{1, Float64}, pack::Wolfpack{1},
                               o::WolfpackObs{1},
                               gating::Gating,
                               θ::WolftrackParameters{1})
    gates(gating, pack.mean, o) ? -Inf : kalman_loglik_and_mean!(mu, pack, o, θ);
end
@inline function gated_loglik!(mu::MVector{2, Float64}, pack::WolfpackConstDiag{2}, o::WolfpackObs{2},
                               gating::Gating, θ::WolftrackParameters{2})
    gates(gating, ref_to_mean(pack), o) ? -Inf : kalman_loglik_and_mean!(mu, pack, o, θ);
end

## Kalman update routines.
@inline function kalman_loglik_and_mean!(mu::MVector{2, Float64}, pack::WolfpackConstDiag{2},
                                         obs::WolfpackObs{2}, θ::WolftrackParameters{2, 4, true})
    _kalman_loglik_and_mean_nomov_constdiag_2d!(mu, view(pack.data, 1:2), pack.data[3],
                                                          obs.location, θ.Σ_obs[1, 1]);
end
@inline function _kalman_loglik_and_mean_nomov_constdiag_2d!(mu::MVector{2, Float64},
                                                             mean::AVec{<: AFloat},
                                                             σ2_diag::AFloat,
                                                             obs::StaticArray{Tuple{2}, <: AFloat, 1},
                                                             σ2_obs::AFloat)
    σ2_sum = σ2_diag + σ2_obs;
    σ2_diag_times_inv_σ2_sum = σ2_diag * inv(σ2_sum);
    om1 = obs[1] - mean[1];
    om2 = obs[2] - mean[2];
    mu[1] = mean[1] + σ2_diag_times_inv_σ2_sum * om1;
    mu[2] = mean[2] + σ2_diag_times_inv_σ2_sum * om2;

    # Independence of Gaussians used here.
    dist = Normal(0.0, sqrt(σ2_sum));
    logpdf(dist, om1) + logpdf(dist, om2); # OK.
end
@inline function kalman_update!(pack::WolfpackConstDiag{2}, obs::WolfpackObs{2},
                                θ::WolftrackParameters{2, 4, true})
    cpred = pack.data[3];
    cpred_inv_s = cpred * inv(cpred + θ.Σ_obs[1, 1]);
    _constdiag_mu_filt_nomov_2d!(view(pack.data, 1:2), obs.location, cpred_inv_s);
    pack.data[3] = _constdiag_cov_filt_diagelem(cpred, cpred_inv_s);
    nothing;
end
@inline function _constdiag_mu_filt_nomov_2d!(mu::AVec{<: AFloat},
                                              obs::StaticArray{Tuple{2}, <: AFloat, 1},
                                              cpred_inv_s::AFloat)
    mu[1] = mu[1] + cpred_inv_s * (obs[1] - mu[1]);
    mu[2] = mu[2] + cpred_inv_s * (obs[2] - mu[2]);
    nothing;
end
@inline function _constdiag_cov_filt_diagelem(cpred::AFloat, cpred_inv_s::AFloat)
    cpred - cpred * cpred_inv_s;
end

"""
1D Kalman loglik without movement model.
"""
@inline function kalman_loglik_and_mean!(mu::MVector{1, Float64}, pack::Wolfpack{1}, obs::WolfpackObs{1},
                               θ::WolftrackParameters{1, 1, B}) where B
    _kalman_loglik_and_mean_nomov_1d!(mu, pack.mean[1], pack.cov[1, 1],
                                      obs.location[1], θ.Σ_obs[1, 1]);
end
@inline function _kalman_loglik_and_mean_nomov_1d!(mu::MVector{1, Float64}, mean::AFloat,
                                          var::AFloat, obs::AFloat, σ2_obs::AFloat)
    mu[1] = mean + var * inv(var + σ2_obs) * (obs - mean);
    logpdf(Normal(0.0, sqrt(var + σ2_obs)), obs - mean); # OK.
end

"""
In place 1D Kalman update with no movement model.
"""
@inline function kalman_update!(pack::Wolfpack{1}, obs::WolfpackObs{1},
                                θ::WolftrackParameters{1, 1, B}) where B
    _kalman_update_nomov_1d!(pack.mean, pack.cov, obs.location[1],
                                       θ.Σ_obs[1, 1]);
    nothing;
end
function _kalman_update_nomov_1d!(mu::MVector{1, <: AFloat},
                                  cov::MMatrix{1, 1, <: AFloat, 1},
                                  obs::AFloat, σ2_obs::AFloat)
    cpred = cov[1, 1];
    s = cpred + σ2_obs;
    cpred_inv_s = cpred * inv(s);
    mu[1] = mu[1] + cpred_inv_s * (obs - mu[1]);
    cov[1] = cpred - cpred * cpred_inv_s;
    nothing; # OK.
end


"""
In place 2D Kalman update with constant diagonal Sigma_obs and
no movement model.
"""
@inline function kalman_update!(pack::Wolfpack{2}, obs::WolfpackObs{2},
                                θ::WolftrackParameters{2, 4, true})
    _kalman_update_nomov_constdiag_2d!(pack.mean, pack.cov, θ.Σ_obs[1, 1],
                                                 obs.location);
    nothing;
end

function _kalman_update_nomov_constdiag_2d!(mu::MVector{2, <: AFloat},
                                            cov::MMatrix{2, 2, <: AFloat, 4},
                                            σ2_obs::AFloat,
                                            obs::StaticArray{Tuple{2}, <: AFloat, 1})
    cpred = cov[1, 1];
    cpred_inv_s = cpred * inv(cpred + σ2_obs);
    _constdiag_mu_filt_nomov_2d!(mu, obs, cpred_inv_s);
    filt_cov_diag_elem = _constdiag_cov_filt_diagelem(cpred, cpred_inv_s);
    cov[1] = filt_cov_diag_elem;
    cov[2] = zero(eltype(cov));
    cov[3] = zero(eltype(cov));
    cov[4] = filt_cov_diag_elem;
    nothing; # OK.
end
