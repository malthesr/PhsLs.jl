module Posterior

export clusterpost, genotypepost

using StaticArrays: MMatrix, SMatrix

using ..Utils
using ..Types
using ..Parameters
using ..ForwardBackward

clusterpost(ab::FwdBwdSite) = fwd(ab) .* bwd(ab)
clusterpost(ab::FwdBwd) = fwd(ab) .* bwd(ab)

function genotypepost(gl::Gl, ab::FwdBwdSite, par::ParSite)
    post = zeros(MMatrix{2, 2, Float64})
    cpost = clusterpost(ab)
    af = allelefreqs(par)
    for z in zs(clusters(par))
        k = emission(gl, z, af)
        for g in gs
            post[g[1], g[2]] += 2 * cpost[z...] * gl[g] * emission(g, z, af) / k
        end
    end
    SMatrix(post)
end

function genotypepost(gl::Vec{Gl}, ab::FwdBwd, par::Par)
    map(
        ((gl, ab, par),) -> genotypepost(gl, ab, par),
        gl, eachsite(ab), eachsite(par)
    )
end

end
