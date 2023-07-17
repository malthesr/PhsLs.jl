module Emission

export emission

using ..Types
using ..Parameters

emission(a::Allele, af::Float64) = a == alt ? af : 1 - af
emission(g::G, z::Z, af::AlleleFreqs{C}) where {C} = 
    prod(((g, z),) -> emission(g, af[z]), zip(g, z))
emission(g::G, af::AlleleFreqs{C}) where {C} = 
    map(z -> emission(g, z, af), zs(Val(C)))

emission(gl::Gl, z::Z, af::AlleleFreqs{C}) where {C} = 
    sum(g -> gl[g] * emission(g, z, af), gs)
emission(gl::Gl, af::AlleleFreqs{C}) where {C} = 
    map(z -> emission(gl, z, af), zs(Val(C)))
emission(gl::Gl, par::ParSite{C}) where {C} = 
    emission(gl, allelefreqs(par))

end
