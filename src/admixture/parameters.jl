module Parameters

export Par, ParInd, ParIndSite, parinit, protect!

using ..Utils
using ..Types

struct Par{A<:Arr3, M<:Mat}
    F::A
    Q::M
end

struct ParInd{A<:Arr3, V<:Vec}
    F::A
    Q::V
end

struct ParIndSite{M<:Mat, V<:Vec}
    F::M
    Q::V
end

Base.getindex(par::Par, i::Integer) = 
    ParInd(view(par.F, :, :, :), view(par.Q, i, :))

Base.getindex(par::ParInd, s::Integer) = 
    ParIndSite(view(par.F, s, :, :), view(par.Q, :))

Types.sites(par::Par) = size(par.F, 1)
Types.inds(par::Par) = size(par.Q, 1)
Types.clusters(par::Par) = size(par.F, 2)
Types.populations(par::Par) = size(par.Q, 2)
Base.size(par::Par) = size(par.F)

Types.sites(par::ParInd) = size(par.F, 1)
Types.clusters(par::ParInd) = size(par.F, 2)
Types.populations(par::ParInd) = length(par.Q)
Base.size(par::ParInd) = size(par.F)

Types.clusters(par::ParIndSite) = size(par.F, 1)
Types.populations(par::ParIndSite) = length(par.Q)
Base.size(par::ParIndSite) = size(par.F)

Types.eachind(par::Par) = map(i -> par[i], 1:inds(par))
Types.eachsite(par::ParInd) = map(i -> par[i], 1:sites(par))

function parinit(I::Integer, S::Integer, C::Integer, K::Integer)
    F = rand(Float64, S, C, K)
    norm!(F, dims=(1, 3))
    Q = rand(Float64, I, K)
    norm!(Q, dims=1)
    Par(F, Q)
end

function protect!(par::Par; minF=1e-5, minQ=1e-5)
    clamp!(par.F, minF, 1.0 - minF)
    norm!(par.F, dims=(1, 3))
    clamp!(par.Q, minQ, 1.0 - minQ)
    norm!(par.Q, dims=1)
end

end
