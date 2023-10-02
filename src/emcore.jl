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

function em(input, par; tol=1e-4, maxiter=100, ekwargs=Dict(), mkwargs=Dict())
    oldloglik = -Inf
    change = Inf
    iter = 0
    loglik = 0.0
    logliks = Float64[]
    pars = typeof(par)[]
    while change > tol && iter < maxiter
        (loglik, par) = emstep(input, par; ekwargs=ekwargs, mkwargs=mkwargs)
        iter += 1
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
