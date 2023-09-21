module Emission

export genotypeemission, emission, emission!

using ..Types
using ..Parameters

@inline genotypeemission(a::Allele, P::Float64) = a == alt ? P : 1 - P
@inline genotypeemission(g::G, z::Z, P::Vec) = 
    prod(((g, z),) -> genotypeemission(g, P[z]), zip(g, z))

@inline emission(gl::Gl, z::Z, P::Vec) = 
    sum(g -> gl[g] * genotypeemission(g, z, P), gs)

@inline function emission(gl::Gl, P::Vec)
    C = length(P)
    m = Matrix{Float64}(undef, C, C)
    emission!(m, gl, P)
    m
end
@inline emission(gl::Gl, par::ParSite) = emission(gl, par.P)

@inline function emission!(m::Mat, gl::Gl, P::Vec)
    C = length(P)
    for (z1, z2) in zzs(C)
        m[z1, z2] = emission(gl, Z(z1, z2), P)
    end
end
@inline emission!(m::Mat, gl::Gl, par::ParSite) = emission!(m, gl, par.P)


end
