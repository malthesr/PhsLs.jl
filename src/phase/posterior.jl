module Posterior

export clusterpost!, clusterpost, clusterexpect!, clusterexpect, 
    clusteralleleexpect!, clusteralleleexpect,
    jumpclusterexpect!, jumpclusterexpect

using ..Utils
using ..Types
using ..Parameters
using ..Emission
using ..Posterior
using ..ForwardBackward
using ..ForwardBackward: FwdSums, forwardsums!

function clusterpost!(p::Mat, ab::FwdBwdSite)
    p .= fwd(ab) .* bwd(ab)
end

function clusterpost!(p::Arr3, ab::FwdBwd)
    (S, C, C) = size(ab)
    p = zeros(S, C, C)
    @inbounds for s in 1:S
        curr = view(p, s, :, :)
        clusterpost!(curr, ab[s])
    end
end

clusterpost(ab::FwdBwdSite) = fwd(ab) .* bwd(ab)
clusterpost(ab::FwdBwd) = fwd(ab) .* bwd(ab)

function clusterexpect!(e::Vec, ab::FwdBwdSite)
    a, b = (fwd(ab), bwd(ab))
    @inbounds for (z1, z2) in zzs(clusters(ab))
        e[z1] += a[z1, z2] * b[z1, z2]
    end
end

clusterexpect(ab::FwdBwdSite) = colsum(clusterpost(ab))

function clusteralleleexpect!(e::Mat, gl::Gl, ab::FwdBwdSite, par::ParSite)
    C = clusters(par)
    @inbounds for (z1, z2) in zzs(C)
        k = fwd(ab)[z1, z2] * bwd(ab)[z1, z2] / emission(gl, Z(z1, z2), par.P)
        p1, p2 = (par.P[z1], par.P[z2])
        e[z1, 1] += k * (1 - p1) * (gl[1] * (1 - p2) + gl[2] .* p2)
        e[z1, 2] += k * p1 * (gl[2] * (1 .- p2) + gl[3] .* p2)
    end
end

function clusteralleleexpect!(e::Arr3, gl::GlVec, ab::FwdBwd, par::Par)
    (S, C) = size(par)
    @inbounds for s in 1:S
        curr = view(e, s, :, :)
        clusteralleleexpect!(curr, gl[s], ab[s], par[s])
    end
end

function clusteralleleexpect(gl::GlVec, ab::FwdBwd, par::Par)
    (S, C) = size(par)
    e = zeros(S, C, 2)
    clusteralleleexpect!(e, gl, ab, par)
    e
end

function jumpclusterexpect!(e::Vec, gl::Gl, aprevsums::FwdSums, b::Mat, c::Float64, par::ParSite)
    C = clusters(par)
    (; P, F, er) = par
    @inbounds for (z1, z2) in zzs(C)
        emit = emission(gl, Z(z1, z2), P)
        e[z1] += emit * b[z1, z2] * (
            (1 - er) * er * aprevsums.z[z2] +
            (1 - er)^2 * F[z2]
        )
    end
    e[:] .*= F ./ c
end

function jumpclusterexpect!(e::Mat, gl::GlVec, ab::FwdBwd, par::Par)
    (S, C) = size(par)
    clusterexpect!(view(e, 1, :), ab[1])
    prevsums = zeros(FwdSums, C)
    for s in 2:S
        curr = view(e, s, :)
        forwardsums!(prevsums, fwd(ab[s - 1]))
        jumpclusterexpect!(
            curr,
            gl[s],
            prevsums,
            bwd(ab[s]),
            scaling(ab[s]),
            par[s]
        )
    end
end

function jumpclusterexpect(gl::GlVec, ab::FwdBwd, par::Par)
    (S, C) = size(par)
    e = zeros(S, C)
    jumpclusterexpect!(e, gl, ab, par)
    e
end

end
