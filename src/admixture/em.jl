module Em

using ..Types
using ..Utils
using ..Parameters
using ..Expectation
using ..EmCore

struct Expect{A<:Arr, M<:Mat}
    clusterpop::A
    pop::M
end

Base.:+(x::Expect{A, M}, y::Expect{A, M}) where {A, M} =
    Expect(x.clusterpop .+ y.clusterpop, vcat(x.pop, y.pop))

function EmCore.estep(clusterliks::Arr, par::ParInd)
    (loglik, clusterpop) = loglikelihoodclusterpopexpect(clusterliks, par)
    pop = sumdrop(clusterpop, dims=(1, 2))
    pop = reshape(pop, (1, length(pop)))
    EStep(Expect(clusterpop, pop), loglik)
end

function EmCore.estep(gl::GlMat, par::Par; clusterlikfn::Function)
    it = enumerate(zip(eachind(gl), eachind(par)))
    fn = ((i, (gl, par),),) -> Sum(estep(clusterlikfn(i, gl), par))
    parmapreduce(fn, +, it)
end

function EmCore.mstep(sum::Sum{EStep{Expect{A, M}}}) where {A, M}
    expect = sum.total.expect
    clusterallelefreqs = norm(expect.clusterpop, dims=2)
    S = size(clusterallelefreqs, 1)
    admixtureprops = expect.pop ./ (2 * S)
    newpar = Par(clusterallelefreqs, admixtureprops)
    protect!(newpar)
    (sum.total.loglik, newpar)
end

end
