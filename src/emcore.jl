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
emstep(input, par; kwargs...) = mstep(estep(input, par; kwargs...), par)

function accelerate end

function accelerate(par0, par1, par2; minalpha=1, maxalpha=4)
    r = par1 .- par0
    v = (par2 .- par1) .- r
    alpha = sqrt(sum(r.^2)) / sqrt(sum(v.^2))
    alpha = -min(max(alpha, minalpha), maxalpha)
    paraccel = par0 .- 2 .* alpha .* r .+ alpha^2 .* v
    (alpha, paraccel)
end

function acceleratedemstep(input, par; kwargs...) 
    (_, par1) = emstep(input, par; kwargs...)
    (loglik2, par2) = emstep(input, par1; kwargs...)
    (alpha, accelpar) = accelerate(par, par1, par2)
    @info("Acceleration has α: $(alpha)")
    if isapprox(alpha, -1)
        @info("Skipping accelleration")
        return (2, loglik2, par2)
    end
    (accelloglik, accelpar) = emstep(input, accelpar; kwargs...)
    if accelloglik > loglik2
        (3, accelloglik, accelpar)
    else
        @warn(
            "Accelerated log-likelihood worse after stabilisation \
            ($(accelloglik)<$(loglik2))), falling back to previous parameters"
        )
        (2, loglik2, par2)
    end
end

function em(input, par; tol=1e-4, maxiter=100, noaccelerate=false, kwargs...)
    oldloglik = -Inf
    change = Inf
    iter = 0
    loglik = 0.0
    logliks = Float64[]
    pars = typeof(par)[]
    while change > tol && iter < maxiter
        if noaccelerate
            (loglik, par) = emstep(input, par; kwargs...)
            iter += 1
        else
            (iters, loglik, par) = acceleratedemstep(input, par; kwargs...)
            iter += iters
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

end
