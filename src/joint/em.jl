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

EmCore.workspacetype(::Type{<:Par}) = Workspace{Expect, Buf}

struct Expect
    clusterallele::Array{Float64, 3}
    clusterancestryjump::Array{Float64, 3}
    ancestry::Matrix{Float64}
end

function EmCore.create(::Type{Expect}, par::Par) 
    (I, S, C, K) = size(par)
    Expect(zeros(S, C, 2), zeros(S, C, K), zeros(I, K))
end

function EmCore.clear!(e::Expect)
    e.clusterallele .= 0.0
    e.clusterancestryjump .= 0.0
    e.ancestry .= 0.0
end

struct ExpectInd
    clusterallele::Array{Float64, 3}
    clusterancestryjump::Array{Float64, 3}
    ancestry::Vector{Float64}
end

function EmCore.create(::Type{ExpectInd}, par::Par) 
    (I, S, C, K) = size(par)
    ExpectInd(zeros(S, C, 2), zeros(S, C, K), zeros(K))
end

function EmCore.clear!(e::ExpectInd)
    e.clusterallele .= 0.0
    e.clusterancestryjump .= 0.0
    e.ancestry .= 0.0
end

struct Buf
    ab::FwdBwd
    h::Mat
    expect::ExpectInd
end

function EmCore.create(::Type{Buf}, par::Par)
    (I, S, C, K) = size(par)
    Buf(zeros(FwdBwd, S, C), zeros(S, C), create(ExpectInd, par))
end

function EmCore.clear!(b::Buf)
    clear!(b.expect)
end

function EmCore.estep!(buf::Buf, gl::GlVec, par::ParInd)
    jumpclusterfreq!(buf.h, par)
    forwardbackward!(buf.ab, gl, buf.h, par)
    clusteralleleexpect!(buf.expect.clusterallele, gl, buf.ab, par);
    clusterancestryjumpexpect!(
        buf.expect.clusterancestryjump, gl, buf.ab, buf.h, par
    );
    ancestryexpect!(buf.expect.ancestry, gl, buf.ab, buf.h, par);
    loglikelihood(buf.ab)
end

function add!(sum::Sum{Expect}, expect::ExpectInd, i::Integer)
    sum.expect.clusterallele .+= expect.clusterallele
    sum.expect.clusterancestryjump .+= expect.clusterancestryjump
    sum.expect.ancestry[i, :] = expect.ancestry
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

function EmCore.mstep!(par::Par, sum::Sum{Expect}; fix::Set{Symbol}=Set())
    (I, S, C, K) = size(par)
    @assert(isapprox(Base.sum(sum.expect.clusterallele), I * S))
    @assert(isapprox(Base.sum(sum.expect.ancestry), I * S))
    if !(:P in fix)
        par.P .= sum.expect.clusterallele[:, :, 2] ./ 
            sumdrop(sum.expect.clusterallele, dims=3)
    end
    if !(:er in fix)
        par.er .= [
            0.; 
            1 .- sumdrop(sum.expect.clusterancestryjump, dims=(2, 3))[2:end] ./ I
        ]
    end
    if !(:F in fix)
        par.F .= sum.expect.clusterancestryjump
        norm!(par.F, dims=(1, 3))
    end
    if !(:Q in fix)
        par.Q .= sum.expect.ancestry
        norm!(par.Q, dims=1)
    end
    protect!(par)
end

const Fix = Option{Union{Symbol, Set{Symbol}}}
const Init = Option{Union{Par, Int}}

function em(beagle::Beagle; C::Integer, K::Integer, init::Init=nothing, fix::Fix=nothing, kwargs...)
    (I, S) = size(beagle)
    gl = joingl(beagle)

    if typeof(init) == Par
        par = init
    else 
        if !isnothing(init)
            Random.seed!(init)
        end
        par = parinit(beagle, C=C, K=K)
    end

    ekwargs = Dict()

    if isnothing(fix)
        fix = Set{Symbol}()
    elseif typeof(fix) == Symbol
        fix = Set([fix])
    end
    mkwargs = Dict(:fix=>fix)

    embase(
        gl,
        par;
        ekwargs=ekwargs,
        mkwargs=mkwargs,
        kwargs...
    )
end

end

