module Parameters

import StaticArrays: SVector, SMatrix, Size

using ..Utils
using ..Types

export Par, ParSite, AlleleFreqs, JumpClusterFreqs, StayFreq, 
    parinit, allelefreqs, jumpclusterfreqs, stayfreq, sites, protect!

struct ParameterException <: Exception end

const AlleleFreqs{C} = SVector{C, Float64}
const JumpClusterFreqs{C} = SVector{C, Float64}
const StayFreq = Float64

struct Par{C}
    allelefreqs::Vector{AlleleFreqs}
    jumpclusterfreqs::Vector{JumpClusterFreqs}
    stayfreqs::Vector{StayFreq}

    function (::Type{Par{C}})(af, jcf, sf) where {C} 
        if allsame(map(length, [af, jcf, sf]))
            new{C}(af, jcf, sf) 
        else
            throw(ParameterException)
        end
    end
end

Base.getindex(par::Par{C}, s::Int) where {C} = 
    ParSite{C}(par.allelefreqs[s], par.jumpclusterfreqs[s], par.stayfreqs[s])

Types.sites(par::Par{C}) where {C} = length(par.stayfreqs)
Types.eachsite(par::Par{C}) where {C} = map(s -> par[s], 1:sites(par))

struct ParSite{C}
    allelefreqs::AlleleFreqs
    jumpclusterfreqs::JumpClusterFreqs
    stayfreq::StayFreq

    function (::Type{ParSite{C}})(af, jcf, sf) where {C}
         new{C}(af, jcf, sf) 
    end
end

allelefreqs(par::ParSite{C}) where {C} = par.allelefreqs
jumpclusterfreqs(par::ParSite{C}) where {C} = par.jumpclusterfreqs
stayfreq(par::ParSite{C}) where {C} = par.stayfreq

function parinit(::Val{C}, positions; scaling=1e6) where {C}
    S = length(positions)
    allelefreqs = rand(SVector{C, Float64}, S)
    jumpclusterfreqs = map(norm, rand(SVector{C, Float64}, S))
    distances = [typemax(UInt64); diff(positions)]
    stayfreqs = exp.(-(distances ./ scaling))
    Par{C}(allelefreqs, jumpclusterfreqs, stayfreqs)
end

function protect!(par::Par{C}; minallelefreq=1e-15, minjumpclusterfreq=1e-15, minstayfreq=0.9, maxstayfreq=exp(-1e-9)) where {C}
    maxallelefreq = 1.0 - minallelefreq
    maxjumpclusterfreq = 1.0 - minjumpclusterfreq
    for s in 1:sites(par)
        par.allelefreqs[s] = 
            clamp.(par.allelefreqs[s], minallelefreq, maxallelefreq)
        par.jumpclusterfreqs[s] = 
            norm(clamp.(par.jumpclusterfreqs[s], minjumpclusterfreq, maxjumpclusterfreq))
    end
    clamp!(par.stayfreqs[2:end], minstayfreq, maxstayfreq)
end

end
