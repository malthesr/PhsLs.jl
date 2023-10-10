module Em

export em

using Random

using ..Types
using ..Utils
using ..Input
using ..Parameters
using ..Posterior
using ..EmCore

import ...Phase

EmCore.workspacetype(::Type{Par{M, V}}) where {M, V} = Workspace{Expect, Buf}

struct Expect
    clusterpop::Array{Float64, 3}
    pop::Matrix{Float64}
end

function EmCore.create(::Type{Expect}, par::Par) 
    (I, S, C, K) = size(par)
    Expect(zeros(S, C, K), zeros(I, K))
end

function EmCore.clear!(e::Expect)
    e.clusterpop .= 0.0
    e.pop .= 0.0
end

struct ExpectInd
    clusterpop::Array{Float64, 3}
    pop::Vector{Float64}
end

function EmCore.create(::Type{ExpectInd}, par::Par) 
    (I, S, C, K) = size(par)
    ExpectInd(zeros(S, C, K), zeros(K))
end

function EmCore.clear!(e::ExpectInd)
    e.clusterpop .= 0.0
    e.pop .= 0.0
end

struct Buf
    cl::Array{Float64, 3}
    ab::Phase.FwdBwd
    expect::ExpectInd
end

function EmCore.create(::Type{Buf}, par::Par)
    (I, S, C, K) = size(par)
    Buf(zeros(S, C, C), zeros(Phase.FwdBwd, S, C), create(ExpectInd, par))
end

function EmCore.clear!(b::Buf)
    clear!(b.expect)
end

function EmCore.estep!(buf::Buf, par::ParInd)
    loglik = clusterpopexpect!(buf.expect.clusterpop, buf.cl, par)
    buf.expect.pop .= sumdrop(buf.expect.clusterpop, dims=(1, 2))
    loglik
end

function add!(sum::Sum{Expect}, expect::ExpectInd, i::Integer)
    sum.expect.clusterpop .+= expect.clusterpop
    sum.expect.pop[i, :] .= expect.pop
    sum.n += 1
end

function EmCore.estep!(ws::Workspace, gl::GlMat, par::Par; clfn!::Function, kwargs...)
    (I, S, C, K) = size(par)
    loglik = 0.0
    lck = ReentrantLock()
    @assert(length(ws.bufs) == Threads.nthreads())
    Threads.@threads :static for i = 1:I
        buf = ws.bufs[Threads.threadid()]
        clear!(buf)
        clfn!(buf, ind(gl, i))
        ll = estep!(buf, par[i]; kwargs...)
        lock(lck) do
            loglik += ll
            add!(ws.sum, buf.expect, i)
        end
    end
    loglik
end

function EmCore.mstep!(par::Par, sum::Sum{Expect}; fixedQ=false)
    (I, S, C, K) = size(par)
    par.F .= sum.expect.clusterpop
    norm!(par.F, dims=1)
    if !fixedQ
        par.Q .= sum.expect.pop ./ S
    end
    protect!(par)
end

function em(beagle::Beagle, phasepar::Phase.Par; K::Integer, initpar=nothing, seed=nothing, fixedQ=false, kwargs...)
    gl = joingl(beagle)
    (I, S) = size(gl)
    (S2, C) = size(phasepar)
    @assert(S == S2)

    if !isnothing(initpar)
        par = initpar
    elseif !isnothing(seed)
        Random.seed!(seed)
        par = parinit(I, S, C, K)
    end

    cf = Phase.clusterfreqs(phasepar)
    clfn! = (buf::Buf, gl::GlVec) -> begin
        Phase.forwardbackward!(buf.ab, gl, phasepar)
        Phase.clusterliks!(buf.cl, buf.ab, cf)
    end
    ekwargs = Dict(:clfn! => clfn!)
    mkwargs = Dict(:fixedQ=>fixedQ)

    embase(gl, par; ekwargs=ekwargs, mkwargs=mkwargs, kwargs...)
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
