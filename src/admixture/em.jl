module Em

using ..Types
using ..Utils
using ..Parameters
using ..Posterior
using ..EmCore

struct Expect{A<:Arr3, M<:Mat}
    clusterpop::A
    pop::M
end

Base.:+(x::Expect{A, M}, y::Expect{A, M}) where {A, M} =
    Expect(x.clusterpop .+ y.clusterpop, vcat(x.pop, y.pop))

function EmCore.estep(cl::Arr3, par::ParInd)
    (loglik, clusterpop) = clusterpopexpect(cl, par)
    pop = sumdrop(clusterpop, dims=(1, 2))
    pop = reshape(pop, (1, length(pop)))
    EStep(Expect(clusterpop, pop), loglik)
end

function EmCore.estep(gl::GlMat, par::Par; clfn::Function)
    it = enumerate(zip(eachind(gl), eachind(par)))
    fn = ((i, (gl, par),),) -> Sum(estep(clfn(gl), par))
    parmapreduce(fn, +, it)
end

function EmCore.mstep(sum::Sum{EStep{Expect{A, M}}}, par::Par) where {A, M}
    (S, C, K) = size(par)
    expect = sum.total.expect
    F = expect.clusterpop
    norm!(F, dims=1)
    Q = expect.pop ./ S
    newpar = Par(F, Q)
    protect!(newpar)
    (sum.total.loglik, newpar)
end

end
