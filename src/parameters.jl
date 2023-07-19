module Parameters

using ..Utils
using ..Types

export Par, ParSite, AlleleFreqs, JumpClusterFreqs, StayFreq, 
    parinit, allelefreqs, jumpclusterfreqs, stayfreq, sites, protect!

struct ParameterException <: Exception end

struct Par{M<:Mat{Float64}, V<:Vec{Float64}}
    allelefreqs::M
    jumpclusterfreqs::M
    stayfreqs::V
end

function Base.getindex(par::Par, s::Int)
    ParSite(
        view(par.allelefreqs, s, :),
        view(par.jumpclusterfreqs, s, :),
        par.stayfreqs[s]
    )
end

Base.size(par::Par) = size(par.allelefreqs)
Types.sites(par::Par) = size(par.allelefreqs, 1)
Types.eachsite(par::Par) = map(s -> par[s], 1:sites(par))

struct ParSite{V<:Vec{Float64}}
    allelefreqs::V
    jumpclusterfreqs::V
    stayfreq::Float64
end

Base.length(par::ParSite{V}) where {V} = length(allelefreqs)

allelefreqs(par::ParSite) = par.allelefreqs
jumpclusterfreqs(par::ParSite) = par.jumpclusterfreqs
stayfreq(par::ParSite) = par.stayfreq

function parinit(C::Int, positions; scaling=1e6)
    S = length(positions)
    allelefreqs = rand(Float64, (S, C))
    jumpclusterfreqs = norm(rand(Float64, (S, C)); dims=2)
    distances = [typemax(UInt64); diff(positions)]
    stayfreqs = exp.(-(distances ./ scaling))
    Par(allelefreqs, jumpclusterfreqs, stayfreqs)
end

function protect!(par::Par; minallelefreq=1e-15, minjumpclusterfreq=1e-15, minstayfreq=0.9, maxstayfreq=exp(-1e-9))
    maxallelefreq = 1.0 - minallelefreq
    maxjumpclusterfreq = 1.0 - minjumpclusterfreq
    clamp!(par.allelefreqs, minallelefreq, maxallelefreq)
    clamp!(par.jumpclusterfreqs, minjumpclusterfreq, maxjumpclusterfreq)
    for s in 1:sites(par)
        norm!(par.jumpclusterfreqs[s, :])
    end
    clamp!(par.stayfreqs, minstayfreq, maxstayfreq)
    par.stayfreqs[1] = 0
end

end
