module Em

using ..Utils
using ..Types
using ..Parameters
using ..ForwardBackward
using ..Posterior
using ..EmCore

struct Expect{A<:Arr3, M<:Mat}
    clusterallele::A
    clusterancestryjump::A
    ancestryjump::M
end

Base.:+(x::Expect{A, M}, y::Expect{A, M}) where {A, M} =
    Expect(
        x.clusterallele .+ y.clusterallele,
        x.clusterancestryjump .+ y.clusterancestryjump,
        vcat(x.ancestryjump, y.ancestryjump)
    )

function EmCore.estep(gl::GlVec, par::ParInd)
    (S, C, K) = size(par)
    ab = forwardbackward(gl, par);
    zypost = clusterancestrypost(ab);
    @assert(all(isapprox.(map(sum, eachslice(zypost, dims=1)), 1)))
    zpost = clusterpost(zypost);
    @assert(all(isapprox.(map(sum, eachslice(zpost, dims=1)), 1)))
    clusterallele = clusteralleleexpect(gl, zpost, par);
    @assert(isapprox(sum(clusterallele), S))
    (clusterancestryjump, ancestryjump) = clusterancestryexpects(gl, ab, par);
    ancestryjump = reshape(sumdrop(ancestryjump, dims=1), (1, K))
    loglik = loglikelihood(ab)
    EStep(Expect(clusterallele, clusterancestryjump, ancestryjump), loglik)
end

function EmCore.estep(gl::GlMat, par::Par)
    it = eachind(gl, par)
    fn = ((gl, par),) -> Sum(estep(gl, par))
    parmapreduce(fn, +, it)
end

function EmCore.mstep(sum::Sum{EStep{Expect{A, M}}}, par::Par) where {A, M}
    I = sum.n
    expect = sum.total.expect
    allelefreqs = expect.clusterallele[:, :, 2] ./ 
        sumdrop(expect.clusterallele, dims=3)
    norm!(expect.clusterancestryjump, dims=(1, 3))
    norm!(expect.ancestryjump, dims=1)
    newpar = Par(
        allelefreqs,
        expect.clusterancestryjump,
        expect.ancestryjump,
        par.er,
        par.et
    )
    protect!(newpar)
    (sum.total.loglik, newpar)
end

end
