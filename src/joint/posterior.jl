module Posterior

export ancestrypost, clusterpost, clusterancestrypost,
    clusteralleleexpect, clusterancestryexpects

using ..Utils
using ..Types
using ..Parameters
using ..Emission
using ..ForwardBackward
using ..ForwardBackward: FwdSums, forwardsums!

clusterancestrypost(ab::FwdBwd) = fwd(ab) .* bwd(ab)
clusterancestrypost(ab::FwdBwdSite) = fwd(ab) .* bwd(ab)

clusterpost(zypost::Arr5) = sumdrop(zypost, dims=(4, 5))
clusterpost(ab::FwdBwd) = clusterpost(clusterancestrypost(ab))
clusterpost(zypost::Arr4) = sumdrop(zypost, dims=(3, 4))
clusterpost(ab::FwdBwdSite) = clusterpost(clusterancestrypost(ab))

ancestrypost(zypost::Arr5) = sumdrop(zypost, dims=(2, 3))
ancestrypost(ab::FwdBwd) = ancestrypost(clusterancestrypost(ab))
ancestrypost(zypost::Arr4) = sumdrop(zypost, dims=(1, 2))
ancestrypost(ab::FwdBwdSite) = ancestrypost(clusterancestrypost(ab))

function clusteralleleexpect(gl::GlVec, zpost::Arr3, par::ParInd)
    (S, C, K) = size(par)
    e = zeros(S, C, 2)
    @inbounds for (s, (gl, par)) in enumerate(eachsite(gl, par))
        P = par.P
        for (z1, z2) in zzs(C)
            emit = emission(gl, Z(z1, z2), P)
            post = zpost[s, z1, z2]
            e[s, z1, 1] += (1 - P[z1]) * 
                (gl[1] * (1 - P[z2]) + gl[2] * P[z2]) *
                post / emit
            e[s, z1, 2] += P[z1] * 
                (gl[2] * (1 - P[z2]) + gl[3] * P[z2]) *
                post / emit
        end
    end
    e
end

function clusterancestryexpects!(ezy::Mat, ey::Vec, gl::Gl, aprevsums::FwdSums, b::Arr4, c::Float64, par::ParSite)
    (C, K) = size(par)
    (; P, F, Q, er, et) = par
    for (z1, z2) in zzs(C)
        emit = emission(gl, Z(z1, z2), P)
        for (y1, y2) in yys(K)
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

function clusterancestryexpects(gl::GlVec, ab::FwdBwd, par::ParInd)
    (S, C, K) = size(par)
    ezy = zeros(S, C, K)
    ezy[1, :, :] = sumdrop(clusterancestrypost(ab[1]), dims=(1, 3))
    ey = zeros(S, K)
    ey[1, :] = sumdrop(ancestrypost(ab[1]), dims=1)
    aprevsums = zeros(FwdSums, C, K)
    @inbounds for s in 2:S
        forwardsums!(aprevsums, fwd(ab[s - 1]))
        b = bwd(ab[s])
        c = scaling(ab[s])
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
    (ezy, ey)
end

end
