module Parameters

export Par, ParInd, ParIndSite, clusterallelefreqs, admixtureprops, F, Q, 
    parinit, protect!

using ..Utils
using ..Types

struct Par{A<:Arr, M<:Mat}
    clusterallelefreqs::A
    admixtureprops::M
end

struct ParInd{A<:Arr, V<:Vec}
    clusterallelefreqs::A
    admixtureprops::V
end

struct ParIndSite{M<:Mat, V<:Vec}
    clusterallelefreqs::M
    admixtureprops::V
end

clusterallelefreqs(par) = par.clusterallelefreqs
admixtureprops(par) = par.admixtureprops

F(par) = clusterallelefreqs(par)
Q(par) = admixtureprops(par)

Base.getindex(par::Par, i::Integer) = 
    ParInd(view(F(par), :, :, :), view(Q(par), i, :))
Base.getindex(par::ParInd, s::Integer) = 
    ParIndSite(view(F(par), s, :, :), view(Q(par), :))

Types.sites(par::Par) = size(F(par), 1)
Types.inds(par::Par) = size(Q(par), 1)
Types.clusters(par::Par) = size(F(par), 2)
Types.populations(par::Par) = size(Q(par), 2)

Types.sites(par::ParInd) = size(F(par), 1)
Types.clusters(par::ParInd) = size(F(par), 2)
Types.populations(par::ParInd) = length(Q(par))

Types.clusters(par::ParIndSite) = size(F(par), 1)
Types.populations(par::ParIndSite) = length(Q(par))

Types.eachind(par::Par) = map(i -> par[i], 1:inds(par))
Types.eachsite(par::ParInd) = map(i -> par[i], 1:sites(par))

function parinit(I::Integer, S::Integer, C::Integer, K::Integer)
    clusterallelefreqs = norm(rand(Float64, S, C, K), dims=2)
    admixtureprops = norm(rand(Float64, I, K), dims=2)
    Par(clusterallelefreqs, admixtureprops)
end

function protect!(par::Par; minF=1e-6, minQ=1e-6)
    clamp!(F(par), minF, 1.0 - minF)
    foreach(norm!, eachslice(F(par), dims=(1, 3)))
    clamp!(Q(par), minQ, 1.0 - minQ)
    foreach(norm!, eachrow(Q(par)))
end

end
