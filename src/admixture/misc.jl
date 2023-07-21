module Misc

export clusterfreqs, dataclusterjoint

using ..Types
using ..Parameters

clusterfreqs(z::Cluster, par::ParIndSite) = sum(Q(par) .* F(par)[z, :])
clusterfreqs(par::ParIndSite) = map(z -> clusterfreqs(z, par), cs(clusters(par)))

dataclusterjoint(clusterliks::Mat, z::Z, h::Vec) =
    clusterliks[z...] * h[z[1]] * h[z[2]]
dataclusterjoint(clusterliks::Mat, h::Vec) = 
    map(z -> dataclusterjoint(clusterliks, z, h), zs(length(h)))


end
