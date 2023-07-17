module Em

export estep, mstep, emstep, em

using Logging

using ..Utils
using ..Types
using ..Parameters
using ..ForwardBackward
using ..Expectation

struct Sum{T}
    total::T
    n::UInt
end

Sum(item::T) where {T} = Sum{T}(item, 1)

function Base.:+(sum::Sum{T}, item::T) where {T}
    Sum(sum.total + item, sum.n + 1)
end

function Base.:+(lhs::Sum{T}, rhs::Sum{T}) where {T}
    Sum(lhs.total + rhs.total, lhs.n + rhs.n)
end

n(sum::Sum{T}) where {T} = sum.n
total(sum::Sum{T}) where {T} = sum.total

struct EStep{A<:Arr{Float64}, M<:Mat{Float64}}
    clusterallele::A
    jumpcluster::M
    loglik::Float64
end

function Base.:+(x::EStep{A, M}, y::EStep{A, M}) where {A, M}
    @assert(length(x.clusterallele) == length(y.clusterallele))
    @assert(length(x.jumpcluster) == length(y.jumpcluster))
    clusterallele = x.clusterallele .+ y.clusterallele
    jumpcluster = x.jumpcluster .+ y.jumpcluster
    loglik = x.loglik + y.loglik
    EStep(clusterallele, jumpcluster, loglik)
end

function estep(gl::Vec{Gl}, par::Par)
    ab = FwdBwd(gl, par)
    clusterallele = clusteralleleexpect(gl, ab, par)
    jumpcluster = jumpclusterexpect(gl, ab, par)
    loglik = loglikelihood(ab)
    EStep(clusterallele, jumpcluster, loglik)
end

function estep(gl::Mat{Gl}, par::Par)
    parmapreduce(gl -> Sum(estep(gl, par)), +, eachind(gl))
end

function mstep(expect::Sum{EStep{A, M}}) where {A, M}
    I = n(expect)
    expect = total(expect)
    allelefreqs = expect.clusterallele[:, :, 2] ./ sumdrop(expect.clusterallele, dims=3)
    jumpclusterfreqs = norm(expect.jumpcluster; dims=2)
    stayfreqs = [0.; 1 .- rowsum(expect.jumpcluster[2:end, :]) ./ I]
    newpar = Par(allelefreqs, jumpclusterfreqs, stayfreqs)
    protect!(newpar)
    (expect.loglik, newpar)
end

emstep(gl::Mat{Gl}, par::Par) = mstep(estep(gl, par))

function em(gl::Mat{Gl}, par::Par; tol=1e-4, maxiter=100)
    oldloglik = -Inf
    change = Inf
    iter = 0
    logliks = Float64[]
    while change > tol && iter < maxiter
        (loglik, par) = emstep(gl, par)
        iter += 1
        change = loglik - oldloglik
        @info("Finished iteration $(iter): logℓ=$(loglik) (Δ=$(change))")
        oldloglik = loglik
        push!(logliks, loglik)
    end
    (logliks, par)
end

end
