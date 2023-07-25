module Transition

using Base.Iterators: product

export transition

using ..Types
using ..Parameters

function transition(z::Cluster, zprev::Cluster, y::Pop, j::Jump, par::ParSite)
    Q(par)[y] * (
        j == jump ? jumpfreq(par) * F(par)[z, y] : stayfreq(par) * Float64(z == zprev)
    )
end
 
function transition(z::Z, zprev::Z, y::Y, j::J, par::ParSite)
    it = zip(z, zprev, y, j)
    prod(((z, zprev, y, j),) -> transition(z, zprev, y, j, par), it)
end

function transition(z::Z, zprev::Z, par::ParSite)
    K = populations(par)
    it = product(ys(K), js)
    sum(((y, j),) -> transition(z, zprev, y, j, par), it)
end

transition(zprev::Z, par::ParSite) = 
    map(z -> transition(z, zprev, par), zs(clusters(par)))

end
