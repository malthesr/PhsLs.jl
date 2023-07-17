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

struct EStep{C}
    clusterallele::Vector{ClusterAlleleExpect{C}}
    jumpcluster::Vector{JumpClusterExpect{C}}
    loglik::Float64
end

function Base.:+(x::EStep{C}, y::EStep{C})  where {C}
    @assert(length(x.clusterallele) == length(y.clusterallele))
    @assert(length(x.jumpcluster) == length(y.jumpcluster))
    clusterallele = x.clusterallele .+ y.clusterallele
    jumpcluster = x.jumpcluster .+ y.jumpcluster
    loglik = x.loglik + y.loglik
    EStep{C}(clusterallele, jumpcluster, loglik)
end

function estep(gl::Vec{Gl}, par::Par{C}) where {C}
    ab = FwdBwd(gl, par)
    clusterallele = clusteralleleexpect(gl, ab, par)
    jumpcluster = jumpclusterexpect(gl, ab, par)
    loglik = loglikelihood(ab)
    EStep{C}(clusterallele, jumpcluster, loglik)
end

function estep(gl::Mat{Gl}, par::Par{C}) where {C}
    parmapreduce(gl -> Sum(estep(gl, par)), +, eachind(gl))
end

function mstep(expect::Sum{EStep{C}}) where {C}
    I = n(expect)
    expect = total(expect)
    allelefreqs = map(e -> e[:, 2] ./ rowsum(e), expect.clusterallele)
    jumpclusterfreqs = map(norm, expect.jumpcluster)
    stayfreqs = [0.; 1 .- map(sum, expect.jumpcluster[2:end]) ./ I]
    newpar = Par{C}(allelefreqs, jumpclusterfreqs, stayfreqs)
    protect!(newpar)
    (expect.loglik, newpar)
end

emstep(gl::Mat{Gl}, par::Par{C}) where {C} = mstep(estep(gl, par))

function em(gl::Mat{Gl}, par::Par{C}; tol=1e-4, maxiter=100) where {C}
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
