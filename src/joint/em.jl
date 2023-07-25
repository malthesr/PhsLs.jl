module Em

using ..Utils
using ..Types
using ..Parameters
using ..ForwardBackward
using ..Expectation
using ..EmCore

struct Expect{A<:Arr, M<:Mat}
    clusterallele::A
    jumpclusterpop::A
    pop::M
end

Base.:+(x::Expect{A, M}, y::Expect{A, M}) where {A, M} =
    Expect(
        x.clusterallele .+ y.clusterallele,
        x.jumpclusterpop .+ y.jumpclusterpop,
        vcat(x.pop, y.pop)
    )

function EmCore.estep(gl::GlVec, par::ParInd)
    ab = forwardbackward(gl, par)
    clusterallele = clusteralleleexpect(gl, ab, par)
    jumpclusterpop = jumpclusterpopexpect(gl, ab, par)
    pop = colsum(popexpect(gl, ab, par))
    pop = reshape(pop, (1, populations(par)))
    loglik = loglikelihood(ab)
    EStep(Expect(clusterallele, jumpclusterpop, pop), loglik)
end

function EmCore.estep(gl::GlMat, par::Par)
    it = zip(eachind(gl), eachind(par))
    fn = ((gl, par),) -> Sum(estep(gl, par))
    parmapreduce(fn, +, it)
end

function EmCore.mstep(sum::Sum{EStep{Expect{A, M}}}) where {A, M}
    I = sum.n
    expect = sum.total.expect
    allelefreqs = expect.clusterallele[:, :, 2] ./ sumdrop(expect.clusterallele, dims=3)
    jumpclusterpopfreqs = norm(expect.jumpclusterpop, dims=2)
    stayfreqs = [0.; 1 .- sumdrop(expect.jumpclusterpop[2:end, :, :], dims=(2, 3)) ./ I]
    S = size(allelefreqs, 1)
    admixtureprops = expect.pop ./ S
    newpar = Par(allelefreqs, jumpclusterpopfreqs, stayfreqs, admixtureprops)
    protect!(newpar)
    (sum.total.loglik, newpar)
end

end
