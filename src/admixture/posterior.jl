module Posterior

export clusterfreqs, dataclusterjoint, clusterpoppost,
    clusterpopexpect, clusterpopexpect!

using Base.Iterators: product

using ..Types
using ..Utils
using ..Parameters
using ..Posterior

function clusterfreqs!(cf::Vec, par::ParIndSite)
    (C, K) = size(par)
    (; F, Q) = par
    @inbounds for z in zs(C)
        tmp = 0
        @inbounds for y in ys(K)
            tmp += F[z, y] * Q[y]
        end
        cf[z] = tmp
    end
end

function dataclusterjoint!(j::Vec, cl::Mat, cf::Vec)
    C = length(cf)
    fill!(j, 0.0)
    @inbounds for (z1, z2) in zzs(C)
        j[z1] += cl[z1, z2] * cf[z1] * cf[z2]
    end
end

function clusterpoppost!(p::Mat, cf::Vec, j::Vec, par::ParIndSite)
    (C, K) = size(par)
    (; F, Q) = par
    @inbounds for z in zs(C)
        @inbounds for y in ys(K)
            p[z, y] += Q[y] * F[z, y] / cf[z] * j[z]
        end
    end
    norm!(p)
    nothing
end

function clusterpopexpect!(e::Arr3, cl::Arr3, par::ParInd)
    (S, C, K) = size(par)
    loglik = 0
    cf = zeros(C)
    j = zeros(C)
    for s in 1:S
        clusterfreqs!(cf, par[s])
        dataclusterjoint!(j, cl[s, :, :], cf)
        clusterpoppost!(view(e, s, :, :), cf, j, par[s])
        loglik += log(sum(j))
    end
    loglik
end

function clusterpopexpect(cl::Arr3, par::ParInd)
    (S, C, K) = size(par)
    e = zeros(S, C, K)
    loglik = clusterpopexpect!(e, cl, par)
    (loglik, e)
end

end
