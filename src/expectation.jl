module Expectation

export ClusterAlleleExpect, JumpClusterExpect,
    clusterexpect, clusteralleleexpect, jumpclusterexpect

using Base.Iterators: product
using StaticArrays: SMatrix, SVector

using ..Utils
using ..Types
using ..Parameters
using ..Emission
using ..ForwardBackward

const ClusterAlleleExpect{C} = SMatrix{C, 2, Float64}
const JumpClusterExpect{C} = SVector{C, Float64}

function clusteralleleexpect(gl::Gl, ab::FwdBwdSite{C}, par::ParSite{C}) where {C}
    p = allelefreqs(par)
    k = fwd(ab) .* bwd(ab) ./ emission(gl, p)
    refexpect = z -> sum(k[z, :] .* (1 - p[z]) .* (gl[1] .* (1 .- p) .+ gl[2] .* p))
    altexpect = z -> sum(k[z, :] .* p[z] .* (gl[2] .* (1 .- p) + gl[3] .* p ))
    fns = SVector(refexpect, altexpect)
    map(((z, f),) -> f(z), product(clusters(Val(C)), fns))
end

function clusteralleleexpect(gl::Vec{Gl}, ab::FwdBwd{C}, par::Par{C}) where {C}
    map(
        ((gl, ab, par),) -> clusteralleleexpect(gl, ab, par),
        zip(gl, eachsite(ab), eachsite(par))
    )
end

function clusterexpect(ab::FwdBwdSite{C}) where {C} 
    colsum(fwd(ab) .* bwd(ab))
end

function clusterexpect(ab::FwdBwd{C}) where {C}
    map(clusterexpect, eachsite(ab))
end

function jumpclusterexpect(gl::Gl, aprev::Fwd{C}, b::Bwd{C}, c::Float64, par::ParSite{C}) where {C}
    e = stayfreq(par)
    colsums = (1 - e) * e .* colsum(aprev) .+ 
        (1 - e)^2 .* jumpclusterfreqs(par)
    rowsum(
        emission(gl, par) .* 
        b .* 
        outer(jumpclusterfreqs(par), colsums)
    ) ./ c
end

function jumpclusterexpect(gl::Vec{Gl}, ab::FwdBwd{C}, par::Par{C}) where {C}
    S = sites(par)
    expect = zeros(JumpClusterExpect{C}, S)
    expect[1] = clusterexpect(ab[1])
    for s in 2:S
        expect[s] = jumpclusterexpect(
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
