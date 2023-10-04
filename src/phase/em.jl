module Em

using Logging
using Random

using ..Utils
using ..Input
using ..Types
using ..Parameters
using ..ForwardBackward
using ..Posterior
using ..EmCore

EmCore.workspacetype(::Type{Par{M, V}}) where {M, V} = Workspace{Expect, Buf}

struct Expect
    clusterallele::Array{Float64, 3}
    jumpcluster::Matrix{Float64}
end

function EmCore.create(::Type{Expect}, par::Par) 
    (S, C) = size(par)
    Expect(zeros(S, C, 2), zeros(S, C))
end

function EmCore.clear!(e::Expect)
    e.clusterallele .= 0.0
    e.jumpcluster .= 0.0
end

struct Buf
    ab::FwdBwd
    expect::Expect
end

function EmCore.create(::Type{Buf}, par::Par)
    (S, C) = size(par)
    Buf(zeros(FwdBwd, S, C), create(Expect, par))
end

function EmCore.clear!(b::Buf)
    clear!(b.expect)
end

function EmCore.estep!(buf::Buf, gl::GlVec, par::Par; oldpi=false)
    forwardbackward!(buf.ab, gl, par)
    clusteralleleexpect!(buf.expect.clusterallele, gl, buf.ab, par)
    if oldpi
        clusterexpect!(buf.expect.jumpcluster, buf.ab)
    else
        jumpclusterexpect!(buf.expect.jumpcluster, gl, buf.ab, par)
    end
    loglikelihood(buf.ab)
end

function EmCore.add!(x::Expect, y::Expect)
    x.clusterallele .+= y.clusterallele
    x.jumpcluster .+= y.jumpcluster
end

function EmCore.estep!(ws::Workspace, gl::GlMat, par::Par; kwargs...)
    I = inds(gl)
    (S, C) = size(par)
    loglik = 0.0
    lck = ReentrantLock()
    @assert(length(ws.bufs) == Threads.nthreads())
    Threads.@threads :static for i = 1:I
        buf = ws.bufs[Threads.threadid()]
        clear!(buf)
        ll = estep!(buf, ind(gl, i), par; kwargs...)
        lock(lck) do
            loglik += ll
            add!(ws.sum, buf.expect)
        end
    end
    loglik
end

function EmCore.mstep!(par::Par, sum::Sum{Expect}; fixedrecomb=false)
    I = sum.n
    par.P .= sum.expect.clusterallele[:, :, 2] ./ 
        sumdrop(sum.expect.clusterallele, dims=3)
    if !fixedrecomb
        par.er .= [0.; 1 .- rowsum(sum.expect.jumpcluster[2:end, :]) ./ I]
    end
    par.F .= sum.expect.jumpcluster
    norm!(par.F, dims=1)
    protect!(par)
end

function EmCore.em(beagle::Beagle; C::Integer, seed=nothing, fixedrecomb=false, oldpi=false, kwargs...)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    ekwargs = Dict(:oldpi=>oldpi)
    mkwargs = Dict(:fixedrecomb=>fixedrecomb)

    chrlogliks = Vector{Float64}[]
    chrpars = Par[]
    for chr in beagle.chrs
        @info("Running chromosome $(chr.chr)")

        init = parinit(C, chr.pos)
        (chrloglik, chrpar) = EmCore.em(
            chr.gl,
            init;
            accel=false,
            ekwargs=ekwargs,
            mkwargs=mkwargs,
            kwargs...
        )
        push!(chrlogliks, chrloglik)
        push!(chrpars, chrpar)
    end

    logliks = reduce(hcat, chrlogliks)
    P = reduce(vcat, getfield.(chrpars, :P))
    F = reduce(vcat, getfield.(chrpars, :F))
    er = reduce(vcat, getfield.(chrpars, :er))
    par = Par(P, F, er)

    (logliks, par)
end

end
