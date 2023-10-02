module Em

using Random

using ..Utils
using ..Input
using ..Types
using ..Parameters
using ..ForwardBackward
using ..Posterior
using ..EmCore

struct Expect{A<:Arr3, M<:Mat}
    clusterallele::A
    jumpcluster::M
end

Base.:+(x::Expect{A, M}, y::Expect{A, M}) where {A, M} =
    Expect(x.clusterallele .+ y.clusterallele, x.jumpcluster .+ y.jumpcluster)

function EmCore.estep(gl::GlVec, par::Par; oldpi=false)
    ab = forwardbackward(gl, par)
    clusterallele = clusteralleleexpect(gl, ab, par)
    if oldpi
        jumpcluster = clusterexpect(ab)
    else
        jumpcluster = jumpclusterexpect(gl, ab, par)
    end
    loglik = loglikelihood(ab)
    EStep(Expect(clusterallele, jumpcluster), loglik)
end

EmCore.estep(gl::GlMat, par::Par; kwargs...) =
    parmapreduce(gl -> Sum(estep(gl, par; kwargs...)), +, eachind(gl))

function EmCore.mstep(sum::Sum{EStep{Expect{A, M}}}, par::Par; fixedrecomb=false) where {A, M}
    I = sum.n
    expect = sum.total.expect
    P = expect.clusterallele[:, :, 2] ./ sumdrop(expect.clusterallele, dims=3)
    if !fixedrecomb
        er = [0.; 1 .- rowsum(expect.jumpcluster[2:end, :]) ./ I]
    else
        er = par.er
    end
    F = expect.jumpcluster
    norm!(F, dims=1)
    newpar = Par(P, F, er)
    protect!(newpar)
    (sum.total.loglik, newpar)
end

function EmCore.em(input::Beagle; C::Integer, seed=nothing, fixedrecomb=false, oldpi=false, kwargs...)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    par = parinit(C, input.pos)

    ekwargs = Dict(:oldpi=>oldpi)
    mkwargs = Dict(:fixedrecomb=>fixedrecomb)
    EmCore.em(input.gl, par; ekwargs=ekwargs, mkwargs=mkwargs, kwargs...)
end

end
