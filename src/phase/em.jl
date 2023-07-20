module Em

using Logging

using ..Utils
using ..Types
using ..Parameters
using ..ForwardBackward
using ..Expectation
using ..EmCore

struct Expect{A<:Arr{Float64}, M<:Mat{Float64}}
    clusterallele::A
    jumpcluster::M
end

Base.:+(x::Expect{A, M}, y::Expect{A, M}) where {A, M} =
    Expect(x.clusterallele .+ y.clusterallele, x.jumpcluster .+ y.jumpcluster)

function EmCore.estep(gl::Vec{Gl}, par::Par)
    ab = forwardbackward(gl, par)
    clusterallele = clusteralleleexpect(gl, ab, par)
    jumpcluster = jumpclusterexpect(gl, ab, par)
    loglik = loglikelihood(ab)
    EStep(Expect(clusterallele, jumpcluster), loglik)
end

EmCore.estep(gl::Mat{Gl}, par::Par) =
    parmapreduce(gl -> Sum(estep(gl, par)), +, eachind(gl))

function EmCore.mstep(sum::Sum{EStep{Expect{A, M}}}) where {A, M}
    I = sum.n
    expect = sum.total.expect
    allelefreqs = expect.clusterallele[:, :, 2] ./ sumdrop(expect.clusterallele, dims=3)
    jumpclusterfreqs = norm(expect.jumpcluster; dims=2)
    stayfreqs = [0.; 1 .- rowsum(expect.jumpcluster[2:end, :]) ./ I]
    newpar = Par(allelefreqs, jumpclusterfreqs, stayfreqs)
    protect!(newpar)
    (sum.total.loglik, newpar)
end

end
