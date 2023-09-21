module Em

using ..Utils
using ..Types
using ..Parameters
using ..ForwardBackward
using ..Posterior
using ..EmCore

struct Expect{A<:Arr3, M<:Mat}
    clusterallele::A
    jumpcluster::M
end

Base.:+(x::Expect{A, M}, y::Expect{A, M}) where {A, M} =
    Expect(x.clusterallele .+ y.clusterallele, x.jumpcluster .+ y.jumpcluster)

function EmCore.estep(gl::GlVec, par::Par)
    ab = forwardbackward(gl, par)
    clusterallele = clusteralleleexpect(gl, ab, par)
    jumpcluster = jumpclusterexpect(gl, ab, par)
    loglik = loglikelihood(ab)
    EStep(Expect(clusterallele, jumpcluster), loglik)
end

EmCore.estep(gl::GlMat, par::Par) =
    parmapreduce(gl -> Sum(estep(gl, par)), +, eachind(gl))

function EmCore.mstep(sum::Sum{EStep{Expect{A, M}}}, par::Par) where {A, M}
    I = sum.n
    expect = sum.total.expect
    P = expect.clusterallele[:, :, 2] ./ sumdrop(expect.clusterallele, dims=3)
    er = [0.; 1 .- rowsum(expect.jumpcluster[2:end, :]) ./ I]
    F = expect.jumpcluster
    norm!(F, dims=1)
    newpar = Par(P, F, er)
    protect!(newpar)
    (sum.total.loglik, newpar)
end

end
