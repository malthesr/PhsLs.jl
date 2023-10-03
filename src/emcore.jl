module EmCore

export EStep, Workspace, create, clear!, workspacetype,
    estep, estep!, mstep!, emstep, emstep!, em, accelerate

using Logging

function create end
function clear end

mutable struct EStep{T}
    sum::T
    n::Int
    loglik::Float64    
end

create(::Type{EStep{T}}, par) where {T} = EStep(create(T, par), 0, 0.0)
function clear!(e::EStep)
    clear!(e.sum)
    e.n = 0
    e.loglik = 0.0
end

mutable struct Workspace{E, B}
    estep::EStep{E}
    bufs::Vector{B}
end

function create(::Type{Workspace{E, B}}, par) where {E, B} 
    sum = create(EStep{E}, par)
    bufs = [create(B, par) for _ in 1:Threads.nthreads()]
    Workspace(sum, bufs)
end

function workspacetype end

create(::Type{Workspace}, par) = create(workspacetype(typeof(par)), par)

function clear!(ws::Workspace) 
    clear!(ws.estep)
    for i in 1:(length(ws.bufs))
        clear!(ws.bufs[i])
    end

end

function estep! end

function estep(input, par; kwargs...)
    ws = create(Workspace, par)
    estep!(ws, input, par, kwargs...)
    ws.estep
end

function mstep! end

function emstep!(par, ws::Workspace, input; ekwargs=Dict(), mkwargs=Dict())
    estep!(ws, input, par; ekwargs...)
    mstep!(par, ws.estep; mkwargs...)
end

function emstep(input, par; kwargs...)
    ws = create(Workspace, par)
    emstep!(par, ws, input, kwargs...)
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
        loglik = emstep!(par, ws, input, ekwargs=ekwargs, mkwargs=mkwargs)
        iter += 1
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

function acceleratedemstep(input, par; ekwargs, mkwargs, kwargs...) 
    (_, par1) = emstep(input, par, ekwargs=ekwargs, mkwargs=mkwargs)
    (loglik2, par2) = emstep(input, par1, ekwargs=ekwargs, mkwargs=mkwargs)
    (alphas, accelpar) = accelerate(par, par1, par2, kwargs...)
    if all(isapprox.(alphas, -1))
        @info("Skipping acceleration")
        return Acceleration(loglik2, par2, 2)
    end
    (accelloglik, accelpar) = emstep(input, accelpar, ekwargs=ekwargs, mkwargs=mkwargs)
    if accelloglik > loglik2
        Acceleration(accelloglik, accelpar, 3)
    else
        @warn(
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
