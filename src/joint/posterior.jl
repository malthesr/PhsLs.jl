module Posterior

export ancestrypost, clusterpost, clusterancestrypost,
    clusteralleleexpect, clusteralleleexpect!,
    clusterancestryexpects, clusterancestryexpects!

using ..Utils
using ..Types
using ..Parameters
using ..Emission
using ..ForwardBackward
using ..ForwardBackward: FwdSums, forwardsums!

function clusteralleleexpect!(e::Arr3, gl::GlVec, ab::FwdBwd, par::ParInd)
    (S, C, K) = size(par)
    @inbounds for s in 1:S
        P = par[s].P
        (a, b) = (ab[s].fwd, ab[s].bwd)
        l = gl[s]
        @inbounds for (z1, z2) in zzs(C)
            emit = emission(l, Z(z1, z2), P)
            zpost = 0
            @inbounds for (y1, y2) in yys(K)
                zpost += a[z1, z2, y1, y2] * b[z1, z2, y1, y2]
            end
            e[s, z1, 1] += (1 - P[z1]) * 
                (l[1] * (1 - P[z2]) + l[2] * P[z2]) *
                zpost / emit
            e[s, z1, 2] += P[z1] * 
                (l[2] * (1 - P[z2]) + l[3] * P[z2]) *
                zpost / emit
        end
    end
end

function clusteralleleexpect(gl::GlVec, ab::FwdBwd, par::ParInd)
    (S, C, K) = size(par)
    e = zeros(S, C, 2)
    clusteralleleexpect!(e, gl, ab, par)
    e
end

function clusterancestryexpects!(ezy::Mat, ey::Vec, gl::Gl, aprevsums::FwdSums, b::Arr4, c::Float64, par::ParSite)
    (C, K) = size(par)
    (; P, F, Q, er, et) = par
    @inbounds for (z1, z2) in zzs(C)
        emit = emission(gl, Z(z1, z2), P)
        @inbounds for (y1, y2) in yys(K)
            q1, q2, f1, f2 = (Q[y1], Q[y2], F[z1, y1], F[z2, y2])
            ebc = emit * b[z1, z2, y1, y2] / c
            ey[y1] += ebc * (
                (1 - et)^2 * q1 * q2 * f1 * f2 +
                et * (1 - et) * q1 * f1 * (
                    er * aprevsums.zy[z2, y2] +
                    (1 - er) * f2 * aprevsums.zzy[y2]
                )
            )
            ezy[z1, y1] += ebc * (
                (1 - et) * q1 * f1 * (
                    (1 - et) * q2 * f2 +
                    et * (1 - er) * f2 * aprevsums.zzy[y2] +
                    et * er * aprevsums.zy[z2, y2]
                ) +
                et * (1 - er) * f1 * (
                    (1 - et) * q2 * f2 * aprevsums.zzy[y1] +
                    et * (1 - er) * f2 * aprevsums.zz[y1, y2] +
                    et * er * aprevsums.z[z2, y1, y2] 
                )
            )
        end
    end
end

function clusterancestryexpectsinit!(ezy::Mat, ey::Vec, ab::FwdBwdSite)
    C, K = size(ezy)
    a, b = (ab.fwd, ab.bwd)
    @inbounds for (z1, z2) in zzs(C)
        @inbounds for (y1, y2) in yys(K)
            v = a[z1, z2, y1, y2] * b[z1, z2, y1, y2] 
            ezy[z1, y1] += v
            ey[y1] += v
        end
    end
end

function clusterancestryexpects!(ezy::Arr3, ey::Mat, gl::GlVec, ab::FwdBwd, par::ParInd)
    (S, C, K) = size(par)
    clusterancestryexpectsinit!(view(ezy, 1, :, :), view(ey, 1, :), ab[1])
    aprevsums = zeros(FwdSums, C, K)
    @inbounds for s in 2:S
        forwardsums!(aprevsums, ab[s - 1].fwd)
        b = ab[s].bwd
        c = ab[s].scaling
        clusterancestryexpects!(
            view(ezy, s, :, :),
            view(ey, s, :),
            gl[s],
            aprevsums,
            b,
            c, 
            par[s]
        )
    end
end

function clusterancestryexpects(gl::GlVec, ab::FwdBwd, par::ParInd)
    (S, C, K) = size(par)
    ezy = zeros(S, C, K)
    ey = zeros(S, K)
    clusterancestryexpects!(ezy, ey, gl, ab, par)
    (ezy, ey)
end

end
