module Posterior

export clusterpoppost

using Base.Iterators: product

using ..Utils
using ..Types
using ..Parameters

unnormclusterpoppost(z::Cluster, k::Pop, h::Vec, l::Mat, par::ParIndSite) = 
    Q(par)[k] * F(par)[z, k] / h[z] * sum(l[z, :])

function clusterpoppost(h::Vec, l::Mat, par::ParIndSite)
    it = product(cs(clusters(par)), ks(populations(par)))
    norm(map(((z, k),) -> unnormclusterpoppost(z, k, h, l, par), it))
end

end
