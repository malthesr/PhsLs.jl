module Parameters

using ..Utils
using ..Types

export Par, ParInd, ParSite,
    parinit, allelefreqs, jumpclusterpopfreqs, stayfreqs, stayfreq, jumpfreq,
    admixtureprops, Q, F, P, protect!

struct Par{A<:Arr, M<:Mat, V<:Vec}
    allelefreqs::M
    jumpclusterpopfreqs::A
    stayfreqs::V
    admixtureprops::M
end

struct ParInd{A<:Arr, M<:Mat, V<:Vec, W<:Vec}
    allelefreqs::M
    jumpclusterpopfreqs::A
    stayfreqs::V
    admixtureprops::W
end

struct ParSite{M<:Mat, V<:Vec, W<:Vec}
    allelefreqs::V
    jumpclusterpopfreqs::M
    stayfreq::Float64
    admixtureprops::W
end

function Base.getindex(par::Par, i::Integer)
    ParInd(
        view(P(par), :, :),
        view(F(par), :, :, :),
        view(stayfreqs(par), :),
        view(Q(par), i, :)
    )
end

function Base.getindex(par::ParInd, s::Integer)
    ParSite(
        view(P(par), s, :),
        view(F(par), s, :, :),
        stayfreqs(par)[s],
        Q(par)
    )
end

allelefreqs(par) = par.allelefreqs
jumpclusterpopfreqs(par) = par.jumpclusterpopfreqs
stayfreqs(par) = par.stayfreqs
admixtureprops(par) = par.admixtureprops
stayfreq(par::ParSite) = par.stayfreq
jumpfreq(par::ParSite) = (1.0 - par.stayfreq)

P(par) = allelefreqs(par)
F(par) = jumpclusterpopfreqs(par)
Q(par) = admixtureprops(par)

Types.inds(par::Par) = size(Q(par), 1)
Types.sites(par::Par) = size(P(par), 1)
Types.clusters(par::Par) = size(F(par), 2)
Types.populations(par::Par) = size(F(par), 3)

Types.sites(par::ParInd) = size(P(par), 1)
Types.clusters(par::ParInd) = size(F(par), 2)
Types.populations(par::ParInd) = size(F(par), 3)

Types.clusters(par::ParSite) = length(P(par))
Types.populations(par::ParSite) = length(Q(par))

Types.eachind(par::Par) = map(i -> par[i], 1:inds(par))
Types.eachsite(par::ParInd) = map(s -> par[s], 1:sites(par))

function parinit(I::Integer, C::Integer, K::Integer, positions; scaling=1e6)
    S = length(positions)
    allelefreqs = rand(Float64, (S, C))
    jumpclusterpopfreqs = norm(rand(Float64, (S, C, K)); dims=2)
    admixtureprops = norm(rand(Float64, (I, K)); dims=2)
    distances = [typemax(UInt64); diff(positions)]
    stayfreqs = exp.(-(distances ./ scaling))
    Par(allelefreqs, jumpclusterpopfreqs, stayfreqs, admixtureprops)
end

function protect!(par::Par; minP=1e-6, minF=1e-6, minQ=1e-6, minstayfreq=0.1, maxstayfreq=exp(-1e-9))
    clamp!(P(par), minP, 1.0 - minP)

    clamp!(F(par), minF, 1.0 - minF)
    foreach(norm!, eachslice(F(par), dims=(1, 3)))

    clamp!(stayfreqs(par), minstayfreq, maxstayfreq)
    par.stayfreqs[1] = 0

    clamp!(Q(par), minQ, 1.0 - minQ)
    foreach(norm!, eachrow(Q(par)))
end

end
