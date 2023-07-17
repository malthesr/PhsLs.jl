module Emission

export emission

using ..Types
using ..Parameters

emission(a::Allele, af::Float64) = a == alt ? af : 1 - af
emission(g::G, z::Z, af::Vec{Float64}) = 
    prod(((g, z),) -> emission(g, af[z]), zip(g, z))
emission(g::G, af::Vec{Float64}) = 
    map(z -> emission(g, z, af), zs(length(af)))

emission(gl::Gl, z::Z, af::Vec{Float64}) = 
    sum(g -> gl[g] * emission(g, z, af), gs)
emission(gl::Gl, af::Vec{Float64}) = 
    map(z -> emission(gl, z, af), zs(length(af)))
emission(gl::Gl, par::ParSite) = 
    emission(gl, allelefreqs(par))

end
