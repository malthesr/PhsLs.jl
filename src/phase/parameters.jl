module Parameters

using ..Utils
using ..Types

export Par, ParSite, parinit, protect!

struct Par{M<:Mat, V<:Vec}
    P::M
    F::M
    er::V
end

struct ParSite{V<:Vec}
    P::V
    F::V
    er::Float64
end

function Base.getindex(par::Par, s::AbstractVector{<:Integer})
    Par(
        par.P[s, :],
        par.F[s, :],
        par.er[s]
    )
end

function Base.getindex(par::Par, s::Int)
    ParSite(
        view(par.P, s, :),
        view(par.F, s, :),
        par.er[s]
    )
end


Types.sites(par::Par) = size(par.P, 1)
Types.clusters(par::Par) = size(par.P, 2)
Base.size(par::Par) = size(par.P)

Types.eachsite(par::Par) = map(s -> par[s], 1:sites(par))

Types.clusters(par::ParSite) = length(par.P)

function parinit(C::Int, positions; scaling=1e6)
    S = length(positions)
    P = rand(Float64, (S, C))
    F = rand(Float64, (S, C))
    norm!(F, dims=1)
    d = diff(positions)
    er = [1; exp.(-(d ./ 1e6))]
    Par(P, F, er)
end

function protect!(par::Par; minP=1e-5, minF=1e-5, miner=0.1, maxer=exp(-1e-9))
    clamp!(par.P, minP, 1.0 - minP)
    clamp!(par.F, minF, 1.0 - minF)
    norm!(par.F, dims=1)
    clamp!(par.er, miner, maxer)
    par.er[1] = 0
end

end
