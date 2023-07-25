module Posterior

export clusterpost, clusterpoppost, jumpclusterpoppost

using Base.Iterators: product

using ..Utils
using ..Types
using ..Parameters
using ..Emission
using ..Transition
using ..ForwardBackward

clusterpost(ab::FwdBwdSite) = fwd(ab) .* bwd(ab)
clusterpost(ab::FwdBwd) = fwd(ab) .* bwd(ab)

function clusterpoppost(gl::Gl, z::Z, zprev::Z, y::Y, aprev::Float64, b::Float64, c::Float64, par::ParSite)
    emit = emission(gl, z, P(par)) 
    trans = sum(j -> transition(z, zprev, y, j, par), js)
    aprev * b / c * emit * trans
end

function clusterpoppost(gl::Gl, aprev::Mat, b::Mat, c::Float64, par::ParSite)
    (C, K) = (clusters(par), populations(par))
    it = product(zs(C), zs(C), ys(K))
    fn = ((z, zprev, y),) -> clusterpoppost(
        gl,
        z, zprev, y,
        aprev[zprev...], b[z...], c,
        par
    )
    map(fn, it)
end

function jumpclusterpoppost(gl::Gl, z::Z, zprev::Z, y::Y, j::J, aprev::Float64, b::Float64, c::Float64, par::ParSite)
    emit = emission(gl, z, P(par)) 
    trans = transition(z, zprev, y, j, par)
    aprev * b / c * emit * trans
end

function jumpclusterpoppost(gl::Gl, aprev::Mat, b::Mat, c::Float64, par::ParSite)
    (C, K) = (clusters(par), populations(par))
    post = zeros(Float64, C, C, C, C, K, K, 2, 2)
    for z in zs(C)
        for zprev in zs(C)
            for y in ys(K)
                for j in js
                    post[z..., zprev..., y..., (Int.(j) .+ 1)...] = 
                        jumpclusterpoppost(
                            gl, z, zprev, y, j, aprev[zprev...], b[z...], c, par
                        )
                end
            end
        end
    end
    post
end

end
