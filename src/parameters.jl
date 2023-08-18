module Parameters

export Par, ParInd, ParSite, parinit, protect!

using ..Utils
using ..Types

struct Par{A<:Arr3, M<:Mat, V<:Vec}
    P::M
    F::A
    Q::M
    er::V
    et::M
end

struct ParInd{A<:Arr3, M<:Mat, V<:Vec, W<:Vec}
    P::M
    F::A
    Q::W
    er::V
    et::W
end

struct ParSite{M<:Mat, V<:Vec}
    P::V
    F::M
    Q::V
    er::Float64
    et::Float64
end

function Base.getindex(par::Par, i::Integer)
    ParInd(
        view(par.P, :, :),
        view(par.F, :, :, :),
        view(par.Q, i, :),
        view(par.er, :),
        view(par.et, i, :),
    )
end

function Base.getindex(par::ParInd, s::AbstractVector{<:Integer})
    ParInd(
        view(par.P, s, :),
        view(par.F, s, :, :),
        view(par.Q, :),
        par.er[s],
        par.et[s],
    )
end

function Base.getindex(par::ParInd, s::Integer)
    ParSite(
        view(par.P, s, :),
        view(par.F, s, :, :),
        view(par.Q, :),
        par.er[s],
        par.et[s],
    )
end

Types.inds(par::Par) = size(par.Q, 1)
Types.sites(par::Par) = size(par.P, 1)
Types.clusters(par::Par) = size(par.F, 2)
Types.populations(par::Par) = size(par.F, 3)
Base.size(par::Par) = (inds(par), sites(par), clusters(par), populations(par))

Types.sites(par::ParInd) = size(par.P, 1)
Types.clusters(par::ParInd) = size(par.F, 2)
Types.populations(par::ParInd) = size(par.F, 3)
Base.size(par::ParInd) = (sites(par), clusters(par), populations(par))

Types.clusters(par::ParSite) = length(par.P)
Types.populations(par::ParSite) = length(par.Q)
Base.size(par::ParSite) = (clusters(par), populations(par))

Types.eachind(par::Par) = map(i -> par[i], 1:inds(par))
Types.eachsite(par::ParInd) = map(s -> par[s], 1:sites(par))

function parinit(I::Integer, C::Integer, K::Integer, positions)
    S = length(positions)
    P = rand(Float64, (S, C))
    F = rand(Float64, (S, C, K))
    norm!(F, dims=(1, 3))
    Q = rand(Float64, (I, K))
    norm!(Q, dims=1)
    d = diff(positions)
    er = [0; exp.(-(d ./ 1e6))]
    et = repeat(transpose([0; exp.(0.05 .* -(d ./ 1e6))]), outer=I)
    Par(P, F, Q, er, et)
end

function protect!(par::Par; minP=1e-5, minF=1e-5, minQ=1e-5, miner=1e-10, minet=1e-10)
    clamp!(par.P, minP, 1.0 - minP)

    clamp!(par.F, minF, 1.0 - minF)
    norm!(par.F, dims=(1, 3))

    clamp!(par.Q, minQ, 1.0 - minQ)
    norm!(par.Q, dims=1)

    clamp!(par.er, miner, 1.0 - miner)
    clamp!(par.et, minet, 1.0 - minet)
end

end
