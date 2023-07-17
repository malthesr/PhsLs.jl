module Expectation

export clusterexpect, clusteralleleexpect, jumpclusterexpect

using Base.Iterators: product

using ..Utils
using ..Types
using ..Parameters
using ..Emission
using ..ForwardBackward

function clusteralleleexpect(gl::Gl, ab::FwdBwdSite, par::ParSite)
    af = allelefreqs(par)
    k = fwd(ab) .* bwd(ab) ./ emission(gl, af)
    refexpect = z -> sum(k[z, :] .* (1 - af[z]) .* (gl[1] .* (1 .- af) .+ gl[2] .* af))
    altexpect = z -> sum(k[z, :] .* af[z] .* (gl[2] .* (1 .- af) + gl[3] .* af))
    fns = [refexpect, altexpect]
    map(((z, f),) -> f(z), product(clusters(length(af)), fns))
end

function clusteralleleexpect(gl::Vec{Gl}, ab::FwdBwd, par::Par)
    (S, C) = size(par)
    expect = zeros(Float64, S, C, 2)
    for s in 1:S
        expect[s, :, :] = clusteralleleexpect(gl[s], ab[s], par[s])
    end
    expect
end

function clusterexpect(ab::FwdBwdSite)
    colsum(fwd(ab) .* bwd(ab))
end

function clusterexpect(ab::FwdBwd)
    map(clusterexpect, eachsite(ab))
end

function jumpclusterexpect(gl::Gl, aprev::Mat{Float64}, b::Mat{Float64}, c::Float64, par::ParSite)
    e = stayfreq(par)
    colsums = (1 - e) * e .* colsum(aprev) .+ 
        (1 - e)^2 .* jumpclusterfreqs(par)
    rowsum(
        emission(gl, par) .* 
        b .* 
        outer(jumpclusterfreqs(par), colsums)
    ) ./ c
end

function jumpclusterexpect(gl::Vec{Gl}, ab::FwdBwd, par::Par)
    (S, C) = size(par)
    expect = zeros(Float64, S, C)
    expect[1, :] = clusterexpect(ab[1])
    for s in 2:S
        expect[s, :] = jumpclusterexpect(
            gl[s],
            fwd(ab[s - 1]),
            bwd(ab[s]),
            scaling(ab[s]),
            par[s]
        )
    end
    expect
end

end
