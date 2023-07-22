module Parameters

using ..Utils
using ..Types

export Par, ParSite,
    parinit, allelefreqs, jumpclusterfreqs, stayfreqs, stayfreq, P, H,  protect!

struct Par{M<:Mat, V<:Vec}
    allelefreqs::M
    jumpclusterfreqs::M
    stayfreqs::V
end

struct ParSite{V<:Vec}
    allelefreqs::V
    jumpclusterfreqs::V
    stayfreq::Float64
end

function Base.getindex(par::Par, s::AbstractVector{<:Integer})
    Par(
        par.allelefreqs[s, :],
        par.jumpclusterfreqs[s, :],
        par.stayfreqs[s]
    )
end
function Base.getindex(par::Par, s::Int)
    ParSite(
        view(par.allelefreqs, s, :),
        view(par.jumpclusterfreqs, s, :),
        par.stayfreqs[s]
    )
end

allelefreqs(par) = par.allelefreqs
jumpclusterfreqs(par) = par.jumpclusterfreqs
stayfreqs(par::Par) = par.stayfreqs
stayfreq(par::ParSite) = par.stayfreq

P(par) = allelefreqs(par)
H(par) = jumpclusterfreqs(par)

Base.size(par::Par) = size(P(par))
Base.length(par::ParSite{V}) where {V} = length(allelefreqs)

Types.sites(par::Par) = size(P(par), 1)
Types.clusters(par::Par) = size(par.allelefreqs, 2)
Types.eachsite(par::Par) = map(s -> par[s], 1:sites(par))

Types.clusters(par::ParSite) = length(par)

function parinit(C::Int, positions; scaling=1e6)
    S = length(positions)
    allelefreqs = rand(Float64, (S, C))
    jumpclusterfreqs = norm(rand(Float64, (S, C)); dims=2)
    distances = [typemax(UInt64); diff(positions)]
    stayfreqs = exp.(-(distances ./ scaling))
    Par(allelefreqs, jumpclusterfreqs, stayfreqs)
end

function protect!(par::Par; minP=1e-6, minH=1e-6, minstayfreq=0.1, maxstayfreq=exp(-1e-9))
    clamp!(P(par), minP, 1.0 - minP)
    clamp!(H(par), minH, 1.0 - minH)
    foreach(par -> norm!(H(par)), eachsite(par))
    clamp!(stayfreqs(par), minstayfreq, maxstayfreq)
    par.stayfreqs[1] = 0
end

end
