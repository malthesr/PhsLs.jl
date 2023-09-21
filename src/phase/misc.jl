module Misc

export clusterliks, clusterliks!, clusterfreqs

using ..Utils
using ..Types
using ..Parameters
using ..ForwardBackward
using ..ForwardBackward: FwdSums, forwardsums!
using ..Posterior

function clusterliks!(cl::Mat, ab::FwdBwdSite, cf::Vec; mincl=1e-12)
    C = clusters(ab)
    @inbounds for (z1, z2) in zzs(C)
        cl[z1, z2] = fwd(ab)[z1, z2] * bwd(ab)[z1, z2] / (cf[z1] * cf[z2])
    end
    clamp!(cl, mincl, 1 - mincl)
    norm!(cl)
end

function clusterliks!(cl::Arr3, ab::FwdBwd, cf::Mat)
    (S, C, C) = size(ab)
    for s in 1:S
        curr = view(cl, s, :, :)
        clusterliks!(curr, ab[s], cf[s, :])
    end
end

function clusterliks(ab::FwdBwd, cf::Mat)
    (S, C, C) = size(ab)
    cl = zeros(S, C, C)
    clusterliks!(cl, ab, cf)
    cl
end

clusterliks(gl::GlVec, cf::Mat, par::Par) =
    clusterliks(forwardbackward(gl, par), cf)

function clusterfreqs(par::Par)
    (S, C) = size(par)
    cf = zeros(Float64, S, C, C)
    @inbounds for (z1, z2) in zzs(C)
        f = par[1].F
        cf[1, z1, z2] = f[z1] * f[z2]
    end
    prevsums = zeros(FwdSums, C)
    @inbounds for s in 2:S
        (; P, F, er) = par[s]
        forwardsums!(prevsums, view(cf, s - 1, :, :))
        for (z1, z2) in zzs(C)
            f1, f2 = (F[z1], F[z2])
            cf[s, z1, z2] = er^2 * cf[s - 1, z1, z2] +
                er * (1 - er) * (f1 * prevsums.z[z2] + f2 * prevsums.z[z1]) +
                (1 - er)^2 * f1 * f2
        end
    end
    sumdrop(cf, dims=3)
end

end
