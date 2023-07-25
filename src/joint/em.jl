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

function accelerate(par0, par1, par2)
    r = par1 .- par0
    v = (par2 .- par1) .- r
    alpha = -sqrt(sum(r.^2)) / sqrt(sum(v.^2))
    paraccel = par0 .- r .* 2 .* alpha .+ v .* alpha^2
    paraccel
end

function EmCore.emstep(gl::GlMat, par::Par)
    (_loglik1, par1) = mstep(estep(gl, par))
    (loglik2, par2) = mstep(estep(gl, par1))
    qaccel = accelerate(Q(par), Q(par1), Q(par2))
    accelpar = Par(
        P(par2), 
        F(par2),
        stayfreqs(par2),
        qaccel
    )
    (loglik2, accelpar)
end

end
