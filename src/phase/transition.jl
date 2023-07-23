module Transition

export transition

using ..Types
using ..Parameters

transition(z::Cluster, zprev::Cluster, j::Jump, par::ParSite) =
    j == jump ? jumpfreq(par) * H(par)[z] : stayfreq(par) *  Float64(z == zprev)

function transition(z::Z, zprev::Z, j::J, par::ParSite)
    it = zip(z, zprev, j)
    prod(((z, zprev, j),) -> transition(z, zprev, j, par), it)
end

transition(z::Z, zprev::Z, par::ParSite) = 
    sum(j -> transition(z, zprev, j, par), js)

transition(zprev::Z, par::ParSite) = 
    map(z -> transition(z, zprev, par), zs(clusters(par)))

end
