module Em

using ..Utils
using ..Types
using ..Parameters
using ..ForwardBackward
using ..Posterior
using ..EmCore

struct Expect{A<:Arr3, M<:Mat, V<:Vec}
    clusterallele::A
    clusterancestryjump::A
    ancestryjump::M
    ancestryjumpsum::V
end

Base.:+(x::Expect{A, M, V}, y::Expect{A, M, V}) where {A, M, V} =
    Expect(
        x.clusterallele .+ y.clusterallele,
        x.clusterancestryjump .+ y.clusterancestryjump,
        vcat(x.ancestryjump, y.ancestryjump),
        x.ancestryjumpsum .+ y.ancestryjumpsum,
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
    ancestryjumpsum = sumdrop(ancestryjump, dims=2)
    ancestryjump = reshape(sumdrop(ancestryjump, dims=1), (1, K))
    loglik = loglikelihood(ab)
    EStep(Expect(clusterallele, clusterancestryjump, ancestryjump, ancestryjumpsum), loglik)
end

function EmCore.estep(gl::GlMat, par::Par)
    it = eachind(gl, par)
    fn = ((gl, par),) -> Sum(estep(gl, par))
    parmapreduce(fn, +, it)
end

function EmCore.mstep(sum::Sum{EStep{Expect{A, M, V}}}, par::Par) where {A, M, V}
    I = sum.n
    expect = sum.total.expect
     # Must do these
    clusterjumpfreqs = sumdrop(expect.clusterancestryjump, dims=(2, 3))[2:end] ./ I
    clusterstayfreqs = [0.;  1.0 .- clusterjumpfreqs]
    ancestrystayfreqs = [0.;  1.0 .- expect.ancestryjumpsum[2:end] ./ I] 
    allelefreqs = expect.clusterallele[:, :, 2] ./ 
        sumdrop(expect.clusterallele, dims=3)
    norm!(expect.clusterancestryjump, dims=(1, 3))
    norm!(expect.ancestryjump, dims=1)
    newpar = Par(
        allelefreqs,
        expect.clusterancestryjump,
        expect.ancestryjump,
        clusterstayfreqs,
        ancestrystayfreqs,
    )
    protect!(newpar)
    (sum.total.loglik, newpar)
end

function EmCore.accelerate(par0::Par, par1::Par, par2::Par)
    (alpha, accelQ) = accelerate(par0.Q, par1.Q, par2.Q)
    accelpar = Par(par2.P, par2.F, accelQ, par2.er, par2.et)
    protect!(accelpar)
    (alpha, accelpar)
end

end
