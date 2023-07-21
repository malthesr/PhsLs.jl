module Expectation

export loglikelihoodclusterpopexpect

using ..Types
using ..Utils
using ..Parameters
using ..Posterior
using ..Misc

function loglikelihoodclusterpopexpect(clusterliks::Arr, par::ParInd)
    S = sites(par)
    C = clusters(par)
    K = populations(par)
    expect = zeros(Float64, S, C, K)
    loglik = 0
    for s in 1:S
        cf = clusterfreqs(par[s])
        joint = dataclusterjoint(clusterliks[s, :, :], cf)
        expect[s, :, :] = 2 .* clusterpoppost(cf, joint, par[s])
        loglik += log(sum(joint))
    end
    (loglik, expect)
end

end
