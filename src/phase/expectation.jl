module Expectation

export clusterexpect, clusteralleleexpect, jumpclusterexpect

using Base.Iterators: product

using ..Utils
using ..Types
using ..Parameters
using ..Emission
using ..Posterior
using ..ForwardBackward

function clusteralleleexpect(gl::GlVec, ab::FwdBwd, par::Par)
    (S, C) = size(par)
    expect = zeros(Float64, S, C, 2)
    for s in 1:S
        af = P(par[s])
        k = clusterpost(ab[s]) ./ emission(gl[s], af)
        for z in cs(clusters(par))
            expect[s, z, 1] = sum(
                k[z, :] .* (1 - af[z]) .* (gl[s][1] .* (1 .- af) .+ gl[s][2] .* af)
            )
            expect[s, z, 2] = sum(
                k[z, :] .* af[z] .* (gl[s][2] .* (1 .- af) + gl[s][3] .* af)
            )
        end
    end
    expect
end

function clusterexpect(ab::FwdBwdSite)
    colsum(clusterpost(ab))
end

function clusterexpect(ab::FwdBwd)
    (S, C, C) = size(ab)
    expect = zeros(Float64, S, C)
    for s in 1:S
        expect[s, :] = clusterexpect(ab[s])
    end
    expect
end

function jumpclusterexpect(gl::Gl, aprev::Mat, b::Mat, c::Float64, par::ParSite)
    e = stayfreq(par)
    colsums = (1 - e) * e .* colsum(aprev) .+ 
        (1 - e)^2 .* jumpclusterfreqs(par)
    rowsum(
        emission(gl, P(par)) .* 
        b .* 
        outer(jumpclusterfreqs(par), colsums)
    ) ./ c
end

function jumpclusterexpect(gl::GlVec, ab::FwdBwd, par::Par)
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
