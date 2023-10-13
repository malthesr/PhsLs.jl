module Posterior

export clusterpost!, clusterpost, clusterexpect!, clusterexpect, 
    clusteralleleexpect!, clusteralleleexpect,
    clusterancestryjumpexpect!, clusterancestryjumpexpect

using ..Utils
using ..Types
using ..Parameters
using ..Emission
using ..Posterior
using ..ForwardBackward
using ..ForwardBackward: FwdSums, forwardsums!

function clusteralleleexpect!(e::Mat, gl::Gl, ab::FwdBwdSite, par::ParSite)
    C = clusters(par)
    @inbounds for (z1, z2) in zzs(C)
        k = ab.fwd[z1, z2] * ab.bwd[z1, z2] / emission(gl, Z(z1, z2), par.P)
        p1, p2 = (par.P[z1], par.P[z2])
        e[z1, 1] += k * (1 - p1) * (gl[1] * (1 - p2) + gl[2] .* p2)
        e[z1, 2] += k * p1 * (gl[2] * (1 .- p2) + gl[3] .* p2)
    end
end

function clusteralleleexpect!(e::Arr3, gl::GlVec, ab::FwdBwd, par::ParInd)
    (S, C) = size(par)
    @inbounds for s in 1:S
        curr = view(e, s, :, :)
        clusteralleleexpect!(curr, gl[s], ab[s], par[s])
    end
end

function clusteralleleexpect(gl::GlVec, ab::FwdBwd, par::ParInd)
    (S, C) = size(par)
    e = zeros(S, C, 2)
    clusteralleleexpect!(e, gl, ab, par)
    e
end

function clusterancestryexpect!(e::Mat, ab::FwdBwdSite, par::ParSite)
    C, K = size(par)
    @inbounds for z in zs(C)
        tmp = 0.0
        @inbounds for zp in zs(C)
            tmp += ab.fwd[z, zp] * ab.bwd[z, zp]
        end
        @inbounds for y in ys(K)
            e[z, y] = tmp * par.Q[y]
        end
    end
end

function clusterancestryjumpexpect!(e::Mat, gl::Gl, aprevsums::FwdSums, b::Mat, c::Float64, h::Vec, par::ParSite)
    C, K = size(par)
    (; P, F, Q, er) = par
    @inbounds for z in zs(C)
        tmp = 0.0
        @inbounds for zp in zs(C)
            emit = emission(gl, Z(z, zp), P)
            tmp += emit * b[z, zp] * (
                (1 - er) * er * aprevsums.z[zp] +
                (1 - er)^2 * h[zp]
            )
        end
        @inbounds for y in ys(K)
            e[z, y] = tmp[z] * Q[y] * F[z, y] / c
        end
    end
end

function clusterancestryjumpexpect!(e::Arr3, gl::GlVec, ab::FwdBwd, h::Mat, par::ParInd)
    S, C, K = size(par)
    clusterancestryexpect!(view(e, 1, :, :), ab[1], par[1])
    aprevsums = zeros(FwdSums, C)
    @inbounds for s in 2:S
        curr = view(e, s, :, :)
        (aprev, b, c) = (ab[s - 1].fwd, ab[s].bwd, ab[s].scaling);
        forwardsums!(aprevsums, aprev)
        clusterancestryjumpexpect!(
            curr, gl[s], aprevsums, b, c, view(h, s, :), par[s]
        )
    end
end

function clusterancestryjumpexpect(gl::GlVec, ab::FwdBwd, h::Mat, par::ParInd)
    S, C, K = size(par)
    e = zeros(S, C, K)
    clusterancestryjumpexpect!(e, gl, ab, h, par)
    e
end

end
