module EmCore

using Logging

using ..Utils

export Sum, Workspace, create, clear!, add!, workspacetype,
    estep!, mstep, mstep!, emstep, emstep!, em, accelerate

function create end
function clear end

mutable struct Sum{T}
    expect::T
    n::Int
end

function add!(sum::Sum{T}, expect::T) where {T}
    add!(sum.expect, expect)
    sum.n += 1
end

create(::Type{Sum{T}}, par) where {T} = Sum(create(T, par), 0)

function clear!(s::Sum)
    clear!(s.expect)
    s.n = 0
end

mutable struct Workspace{E, B}
    sum::Sum{E}
    bufs::Vector{B}
end

function create(::Type{Workspace{E, B}}, par) where {E, B} 
    sum = create(Sum{E}, par)
    bufs = [create(B, par) for _ in 1:Threads.nthreads()]
    Workspace(sum, bufs)
end

function workspacetype end

create(::Type{Workspace}, par) = create(workspacetype(typeof(par)), par)

function clear!(ws::Workspace) 
    clear!(ws.sum)
    for i in 1:(length(ws.bufs))
        clear!(ws.bufs[i])
    end

end

function estep! end
function mstep end
function mstep! end

function emstep(input, par; kwargs...)
    newpar = deepcopy(par)
    loglik = emstep!(newpar, input; kwargs...)
    (loglik, newpar)
end

function emstep!(par, input; ws::Option{Workspace}=nothing, ekwargs=Dict(), mkwargs=Dict())
    if isnothing(ws)
        ws = create(Workspace, par)
    end
    loglik = estep!(ws, input, par; ekwargs...)
    mstep!(par, ws.sum; mkwargs...)
    loglik
end

function em(input,
            par;
            tol=1e-4,
            maxiter=100,
            accel=true,
            parcallback=nothing,
            ekwargs=Dict(),
            mkwargs=Dict(),
            accelargs=Dict())
    oldloglik = -Inf
    change = Inf
    iter = 0
    loglik = 0.0
    logliks = Float64[]
    ws = create(Workspace, par)
    while change > tol && iter < maxiter
        if accel
            (; loglik, par, iters) = 
                acceleratedemstep(input, par, ws=ws, ekwargs=ekwargs, mkwargs=mkwargs, accelargs...)
            iter += iters
        else
            loglik = emstep!(par, input, ws=ws, ekwargs=ekwargs, mkwargs=mkwargs)
            iter += 1
        end
        change = abs(loglik - oldloglik)
        @info("Finished EM iteration $(iter): logℓ=$(loglik) (Δ=$(change))")
        if loglik < oldloglik
            @warn("logℓ is not monotonically non-decreasing")
        end
        oldloglik = loglik
        push!(logliks, loglik)
        if !isnothing(parcallback)
            parcallback(par)
        end
        clear!(ws)
    end
    (logliks, par)
end

function accelerate end

function acceleratedemstep(input, par; ws::Workspace, ekwargs, mkwargs, kwargs...) 
    (_, par1) = emstep(input, par, ws=ws, ekwargs=ekwargs, mkwargs=mkwargs)
    (loglik2, par2) = emstep(input, par1, ws=ws, ekwargs=ekwargs, mkwargs=mkwargs)
    (alphas, accelpar) = accelerate(par, par1, par2, kwargs...)
    if all(isapprox.(alphas, -1))
        @info("Skipping acceleration")
        return Acceleration(loglik2, par2, 2)
    end
    (accelloglik, accelpar) = emstep(input, accelpar, ws=ws, ekwargs=ekwargs, mkwargs=mkwargs)
    if accelloglik > loglik2
        Acceleration(accelloglik, accelpar, 3)
    else
        @info(
            "Accelerated log-likelihood worse after stabilisation " *
            "($(accelloglik)<$(loglik2))), falling back to previous parameters"
        )
        Acceleration(loglik2, par2, 2)
    end
end
       
struct Acceleration{T}
    loglik::Float64
    par::T
    iters::Int
end

end
