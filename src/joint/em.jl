module Em

export em

using Logging
using Random

using ..Utils
using ..Input
using ..Types
using ..Parameters
using ..ForwardBackward
using ..Posterior
using ..EmCore

EmCore.workspacetype(::Type{Par{A, M, V}}) where {A, M, V} = Workspace{Expect, Buf}

struct Expect
    clusterallele::Array{Float64, 3}
    clusterancestryjump::Array{Float64, 3}
    ancestryjump::Matrix{Float64}
end

function EmCore.create(::Type{Expect}, par::Par) 
    (I, S, C, K) = size(par)
    Expect(zeros(S, C, 2), zeros(S, C, K), zeros(I, K))
end

function EmCore.clear!(e::Expect)
    e.clusterallele .= 0.0
    e.clusterancestryjump .= 0.0
    e.ancestryjump .= 0.0
end

struct ExpectInd
    clusterallele::Array{Float64, 3}
    clusterancestryjump::Array{Float64, 3}
    ancestryjump::Matrix{Float64}
end

function EmCore.create(::Type{ExpectInd}, par::Par) 
    (I, S, C, K) = size(par)
    ExpectInd(zeros(S, C, 2), zeros(S, C, K), zeros(S, K))
end

function EmCore.clear!(e::ExpectInd)
    e.clusterallele .= 0.0
    e.clusterancestryjump .= 0.0
    e.ancestryjump .= 0.0
end

struct Buf
    ab::FwdBwd
    expect::ExpectInd
end

function EmCore.create(::Type{Buf}, par::Par)
    (I, S, C, K) = size(par)
    Buf(zeros(FwdBwd, S, C, K), create(ExpectInd, par))
end

function EmCore.clear!(b::Buf)
    clear!(b.expect)
end

function EmCore.estep!(buf::Buf, gl::GlVec, par::ParInd; oldpi=false)
    forwardbackward!(buf.ab, gl, par)
    clusteralleleexpect!(buf.expect.clusterallele, gl, buf.ab, par);
    clusterancestryexpects!(
        buf.expect.clusterancestryjump, buf.expect.ancestryjump, gl, buf.ab, par
    );
    loglikelihood(buf.ab)
end

function add!(sum::Sum{Expect}, expect::ExpectInd, i::Integer)
    sum.expect.clusterallele .+= expect.clusterallele
    sum.expect.clusterancestryjump .+= expect.clusterancestryjump
    sum.expect.ancestryjump[i, :] = sumdrop(expect.ancestryjump, dims=1)
    sum.n += 1
end

function EmCore.estep!(ws::Workspace, gl::GlMat, par::Par; kwargs...)
    (I, S, C, K) = size(par)
    loglik = 0.0
    lck = ReentrantLock()
    @assert(length(ws.bufs) == Threads.nthreads())
    Threads.@threads :static for i = 1:I
        buf = ws.bufs[Threads.threadid()]
        clear!(buf)
        ll = estep!(buf, ind(gl, i), par[i]; kwargs...)
        lock(lck) do
            loglik += ll
            add!(ws.sum, buf.expect, i)
        end
    end
    loglik
end

function EmCore.mstep!(par::Par, sum::Sum{Expect}; fixedQ=false)
    (I, S, C, K) = size(par)
    @assert(isapprox(Base.sum(sum.expect.clusterallele), I * S))
    par.P .= sum.expect.clusterallele[:, :, 2] ./ 
        sumdrop(sum.expect.clusterallele, dims=3)
    par.F .= sum.expect.clusterancestryjump
    norm!(par.F, dims=(1, 3))
    if !fixedQ
        par.Q .= sum.expect.ancestryjump
        norm!(par.Q, dims=1)
    end
    protect!(par)
end

function em(beagle::Beagle; C::Integer, K::Integer, initpar=nothing, seed=nothing, fixedQ=false, kwargs...)
    (I, S) = size(beagle)
    gl = joingl(beagle)

    if !isnothing(initpar)
        par = initpar
    elseif !isnothing(seed)
        Random.seed!(seed)
        pos = map(chr -> chr.pos, beagle.chrs)
        par = parinit(I, S, C, K, pos)
    end

    ekwargs = Dict()
    mkwargs = Dict(:fixedQ=>fixedQ)

    embase(
        gl,
        par;
        ekwargs=ekwargs,
        mkwargs=mkwargs,
        kwargs...
    )
end

function EmCore.accelerate(par0::Par, par1::Par, par2::Par; minalpha=1, maxalpha=4)
    (P0, F0, Q0) = (par0.P, par0.F, par0.Q)
    (P1, F1, Q1) = (par1.P, par1.F, par1.Q)
    (P2, F2, Q2) = (par2.P, par2.F, par2.Q)
    ss(x) = sum(x.^2)
    rss(x, y) = ss(x .- y)
    alpha = (rss(P1, P0) + rss(F1, F0) + rss(Q1, Q0)) / 
        (ss(P2 - 2 * P1 + P0) + ss(F2 - 2 * F1 + F0) + ss(Q2 - 2 * Q1 + Q0))
    alpha = max(minalpha, sqrt(alpha))
    alpha = min(maxalpha, sqrt(alpha))
    @info("Acceleration has α: $(alpha)")
    accel(p0, p1, p2) = p0 + 2 * alpha * (p1 - p0) + alpha^2 * (p2 - 2 * p1 + p0)
    paraccel = Par(accel(P0, P1, P2), accel(F0, F1, F2), accel(Q0, Q1, Q2), par2.er, par2.et)
    protect!(paraccel)
    (alpha, paraccel)
end
end

