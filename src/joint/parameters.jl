module Parameters

export Par, ParInd, ParSite, parinit, protect!, jumpclusterfreq, jumpclusterfreq!

using ..Input
using ..Utils
using ..Types

struct Par{A<:Arr3, M<:Mat, N<:Mat, V<:Vec}
    P::M
    F::A
    Q::N
    er::V
end

struct ParInd{A<:Arr3, M<:Mat, V<:Vec, W<:Vec}
    P::M
    F::A
    Q::W
    er::V
end

struct ParSite{M<:Mat, V<:Vec}
    P::V
    F::M
    Q::V
    er::Float64
end

function Base.getindex(par::Par, i::Integer)
    ParInd(
        view(par.P, :, :),
        view(par.F, :, :, :),
        view(par.Q, i, :),
        view(par.er, :),
    )
end

function Base.getindex(par::Par, i::AbstractVector{<:Integer})
    Par(
        view(par.P, :, :),
        view(par.F, :, :, :),
        view(par.Q, i, :),
        view(par.er, :),
    )
end

function Base.getindex(par::ParInd, s::AbstractVector{<:Integer})
    ParInd(
        view(par.P, s, :),
        view(par.F, s, :, :),
        view(par.Q, :),
        par.er[s],
    )
end

function Base.getindex(par::ParInd, s::Integer)
    ParSite(
        view(par.P, s, :),
        view(par.F, s, :, :),
        view(par.Q, :),
        par.er[s],
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

function jumpinit(positions::AbstractVector{<:AbstractVector{<:Integer}})
    er = Float64[]
    for pos in positions
        d = diff(pos)
        append!(er, [1; exp.(-(d ./ 1e6))])
    end
    er
end

jumpinit(positions::AbstractVector{<:Integer}) = jumpinit([positions])

function parinit(I::Integer, S::Integer, C::Integer, K::Integer, positions)
    P = rand(Float64, (S, C))
    F = rand(Float64, (S, C, K))
    norm!(F, dims=(1, 3))
    Q = rand(Float64, (I, K))
    norm!(Q, dims=1)
    er = jumpinit(positions)
    Par(P, F, Q, er)
end

function parinit(beagle::Beagle; C::Integer, K::Integer)
    I, S = size(beagle)
    positions = [chr.pos for chr in beagle.chrs]
    parinit(I, S, C, K, positions)
end

function protect!(par::Par; minP=1e-5, minF=1e-5, minQ=1e-5, miner=0.1, maxer=exp(-1e-9))
    clamp!(par.P, minP, 1.0 - minP)
    clamp!(par.F, minF, 1.0 - minF)
    norm!(par.F, dims=(1, 3))
    clamp!(par.Q, minQ, 1.0 - minQ)
    norm!(par.Q, dims=1)
    clamp!(par.er, miner, maxer)
    par.er[1] = 0
end

function jumpclusterfreq!(h::Vec, par::ParSite)
    C, K = size(par)
    (; P, F, Q, er) = par
    @inbounds for z in zs(C)
        h[z] = 0
        for y in ys(K)
            h[z] += Q[y] * F[z, y]
        end
    end
end

function jumpclusterfreq(par::ParSite)
    h = zeros(clusters(par))
    clusterfreq!(h, par)
    h
end

function jumpclusterfreq!(h::Mat, par::ParInd)
    @inbounds for s in 1:sites(par)
        jumpclusterfreq!(view(h, s, :), par[s])
    end
end

function jumpclusterfreq(par::ParInd)
    S, C, K = size(par)
    h = zeros(S, C)
    jumpclusterfreq!(h, par)
    h
end

end
