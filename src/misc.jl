module Misc

export clusterliks, clusterfreqs, call, mlclusters

using ..Utils
using ..Types
using ..Parameters
using ..ForwardBackward
using ..Posterior
using ..Expectation

function clusterliks(ab::FwdBwd, clusterfreqs::Mat)
    (S, C, C) = size(ab)
    cl = clusterpost(ab)
    for s in 1:S
        cl[s, :, :] ./= outer(clusterfreqs[s, :])
        norm!(view(cl, s, :, :))
    end
    cl
end

clusterliks(gl::Vec{Gl}, clusterfreqs::Mat, par::Par) =
    clusterliks(FwdBwd(gl, par), clusterfreqs)

function mlclusters(clusterliks::Mat)
    sortml = ml -> ml[1] > ml[2] ? (ml[2], ml[1]) : ml
    ml = sortml(Tuple(argmax(clusterliks)))
    Z(ml[1], ml[2])
end

mlclusters(clusterliks::Arr) = map(mlclusters, eachslice(clusterliks, dims=1))

mlclusters(gl::Vec{Gl}, clusterfreqs::Mat, par::Par) = 
    mlclusters(clusterliks(gl, clusterfreqs, par))

mlclusters(gl::Mat{Gl}, clusterfreqs::Mat, par::Par) =
    reduce(hcat, map(gl -> mlclusters(gl, clusterfreqs, par), eachind(gl)))

function clusterfreqs(gl::Mat{Gl}, par::Par)
    I = inds(gl)
    S, C = size(par)
    cf = zeros(Float64, S, C)
    for i in 1:I
        ab = FwdBwd(ind(gl, i), par)
        cf += clusterexpect(ab)
    end
    norm(cf, dims=2)
end

function call(gl::Gl, ab::FwdBwdSite, par::ParSite)
    C = length(par)
    post = genotypepost(gl, ab, par)
    G(Tuple(argmax(post)) .- 1)
end

function call(gl::Vec{Gl}, ab::FwdBwd, par::Par)
    map(
        ((gl, ab, par),) -> genotypepost(gl, ab, par),
        gl, eachsite(ab), eachsite(par)
    )
end

call(gl::Vec{Gl}, par::Par) = call(gl, FwdBwd(gl, par), par)

end
