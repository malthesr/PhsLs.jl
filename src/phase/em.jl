module Em

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

function EmCore.estep!(ws::Workspace, gl::GlMat, par::Par; kwargs...)
    I = inds(gl)
    (S, C) = size(par)
    lk = ReentrantLock()
    @assert(length(ws.bufs) == Threads.nthreads())
    Threads.@threads :static for i = 1:I
        buf = ws.bufs[Threads.threadid()]
        clear!(buf)
        loglik = estep!(buf, ind(gl, i), par; kwargs...)
        lock(lk) do
            ws.estep.sum.clusterallele .+= buf.expect.clusterallele
            ws.estep.sum.jumpcluster .+= buf.expect.jumpcluster
            ws.estep.n += 1
            ws.estep.loglik += loglik
        end
    end
end

function EmCore.mstep!(par::Par, estep::EStep{Expect}; fixedrecomb=false)
    I = estep.n
    expect = estep.sum
    par.P .= expect.clusterallele[:, :, 2] ./ sumdrop(expect.clusterallele, dims=3)
    if !fixedrecomb
        par.er .= [0.; 1 .- rowsum(expect.jumpcluster[2:end, :]) ./ I]
    end
    par.F .= expect.jumpcluster
    norm!(par.F, dims=1)
    protect!(par)
    estep.loglik
end

function EmCore.em(input::Beagle; C::Integer, seed=nothing, fixedrecomb=false, oldpi=false, kwargs...)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    par = parinit(C, input.pos)

    ekwargs = Dict(:oldpi=>oldpi)
    mkwargs = Dict(:fixedrecomb=>fixedrecomb)
    EmCore.em(input.gl, par; accel=false, ekwargs=ekwargs, mkwargs=mkwargs, kwargs...)
end

end
