module EmCore

export Sum, EStep, estep, mstep, emstep, em, accelerate

using Logging

struct Sum{T}
    total::T
    n::UInt
end

Sum(item::T) where {T} = Sum{T}(item, 1)

Base.:+(sum::Sum{T}, item::T) where {T} =
    Sum(sum.total + item, sum.n + 1)
Base.:+(lhs::Sum{T}, rhs::Sum{T}) where {T} =
    Sum(lhs.total + rhs.total, lhs.n + rhs.n)

struct EStep{E}
    expect::E
    loglik::Float64
end

Base.:+(x::EStep{T}, y::EStep{T}) where {T} =
    EStep{T}(x.expect + y.expect, x.loglik + y.loglik)

function estep end
function mstep end
emstep(input, par; ekwargs=Dict(), mkwargs=Dict()) =
    mstep(estep(input, par; ekwargs...), par; mkwargs...)

function em(input,
            par;
            tol=1e-4,
            maxiter=100,
            accel=true,
            ekwargs=Dict(),
            mkwargs=Dict(),
            accelargs=Dict())
    oldloglik = -Inf
    change = Inf
    iter = 0
    loglik = 0.0
    logliks = Float64[]
    pars = typeof(par)[]
    while change > tol && iter < maxiter
        if accel
            (; loglik, par, iters) = 
                acceleratedemstep(input, par; ekwargs=ekwargs, mkwargs=mkwargs, accelargs...)
            iter += iters
        else
            (loglik, par) =
                emstep(input, par; ekwargs=ekwargs, mkwargs=mkwargs)
            iter += 1
        end
        change = abs(loglik - oldloglik)
        @info("Finished EM iteration $(iter): logℓ=$(loglik) (Δ=$(change))")
        if loglik < oldloglik
            @warn("logℓ is not monotonically non-decreasing")
        end
        oldloglik = loglik
        push!(pars, par)
        push!(logliks, loglik)
    end
    (logliks, pars)
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
