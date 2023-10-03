module Em

using Random

using ..Types
using ..Utils
using ..Input
using ..Parameters
using ..Posterior
using ..EmCore

import ...Phase

struct Expect{A<:Arr3, M<:Mat}
    clusterpop::A
    pop::M
end

Base.:+(x::Expect{A, M}, y::Expect{A, M}) where {A, M} =
    Expect(x.clusterpop .+ y.clusterpop, vcat(x.pop, y.pop))

function EmCore.estep(cl::Arr3, par::ParInd)
    (loglik, clusterpop) = clusterpopexpect(cl, par)
    pop = sumdrop(clusterpop, dims=(1, 2))
    pop = reshape(pop, (1, length(pop)))
    EStep(Expect(clusterpop, pop), loglik)
end

function EmCore.estep(gl::GlMat, par::Par; clfn::Function)
    it = enumerate(zip(eachind(gl), eachind(par)))
    fn = ((i, (gl, par),),) -> Sum(estep(clfn(gl), par))
    parmapreduce(fn, +, it)
end

function EmCore.mstep(sum::Sum{EStep{Expect{A, M}}}, par::Par) where {A, M}
    (S, C, K) = size(par)
    expect = sum.total.expect
    F = expect.clusterpop
    norm!(F, dims=1)
    Q = expect.pop ./ S
    newpar = Par(F, Q)
    protect!(newpar)
    (sum.total.loglik, newpar)
end

function EmCore.em(input::Beagle, phasepar::Phase.Par; K::Integer, seed=nothing, kwargs...)
    (I, S) = size(input.gl)
    (S, C) = size(phasepar)

    if !isnothing(seed)
        Random.seed!(seed)
    end
    par = parinit(I, S, C, K)

    cf = Phase.clusterfreqs(phasepar)
    clfn = (gl::GlVec) -> Phase.clusterliks(gl, cf, phasepar)
    ekwargs = Dict(:clfn=>clfn)

    EmCore.em(input.gl, par; ekwargs=ekwargs, kwargs...)
end

function EmCore.accelerate(par0::Par, par1::Par, par2::Par; minalpha=1, maxalpha=4)
    (F0, Q0) = (par0.F, par0.Q)
    (F1, Q1) = (par1.F, par1.Q)
    (F2, Q2) = (par2.F, par2.Q)
    ss(x) = sum(x.^2)
    rss(x, y) = ss(x .- y)
    alpha = (rss(F1, F0) + rss(Q1, Q0)) / 
        (ss(F2 - 2 * F1 + F0) + ss(Q2 - 2 * Q1 + Q0))
    alpha = max(minalpha, sqrt(alpha))
    alpha = min(maxalpha, sqrt(alpha))
    @info("Acceleration has α: $(alpha)")
    accel(p0, p1, p2) = p0 + 2 * alpha * (p1 - p0) + alpha^2 * (p2 - 2 * p1 + p0)
    paraccel = Par(accel(F0, F1, F2), accel(Q0, Q1, Q2))
    protect!(paraccel)
    (alpha, paraccel)
end

end
