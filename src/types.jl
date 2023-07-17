module Types

import StaticArrays: SVector, SMatrix
import Base.Iterators: product

export Allele, ref, alt, G, gs, alts, Gl, Cluster, clusters, Z, zs,
    Vec, Mat, Arr, sites, inds, site, ind, eachsite, eachind

const Vec{T} = StridedVector{T}
const Mat{T} = StridedMatrix{T}
const Arr{T} = StridedArray{T, 3}

@enum Allele begin
    ref = 0
    alt = 1
end

struct G <: AbstractVector{Allele}
    alleles::SVector{2, Allele}
end

G(a::Allele, b::Allele)::G = G(SVector(a, b))
G(a::Int, b::Int)::G = G(SVector(Allele(a), Allele(b)))

Base.parent(g::G) = g.alleles
Base.size(g::G) = size(parent(g))
Base.getindex(g::G, i::Int) = getindex(parent(g), i)
Base.length(g::G) = 3
Base.iterate(g::G) = iterate(parent(g))
Base.IndexStyle(::Type{G}) = IndexLinear()

alts(g::G) = sum(Integer.(parent(g)))

const gs = SMatrix{2, 2}(G(0, 0), G(0, 1), G(1, 0), G(1, 1))

sites(g::Mat{G}) = size(g, 2)
inds(g::Mat{G}) = size(g, 1)
site(g::Mat{G}, s::Int) = view(g, :, s)
ind(g::Mat{G}, i::Int) = view(g, i, :)
eachsite(g::Mat{G}) = eachcol(g)
eachind(g::Mat{G}) = eachrow(g)

struct Gl <: AbstractVector{Float64}
    likelihoods::SVector{3, Float64}
end

Gl(x::Real, y::Real, z::Real)::Gl = Gl(SVector(x, y, z))

Base.parent(gl::Gl) = gl.likelihoods
Base.size(gl::Gl) = size(parent(gl))
Base.getindex(gl::Gl, i::Int) = getindex(parent(gl), i)
Base.length(gl::Gl) = 3
Base.iterate(gl::Gl) = iterate(parent(gl))
Base.IndexStyle(::Type{Gl}) = IndexLinear()

Base.getindex(gl::Gl, g::G) = getindex(gl, alts(g) + 1)

sites(gl::Mat{Gl}) = size(gl, 2)
inds(gl::Mat{Gl}) = size(gl, 1)
site(gl::Mat{Gl}, s::Int) = view(gl, :, s)
ind(gl::Mat{Gl}, i::Int) = view(gl, i, :)
eachsite(gl::Mat{Gl}) = eachcol(gl)
eachind(gl::Mat{Gl}) = eachrow(gl)

struct ClusterException <: Exception end

struct Cluster{C}
    cluster::UInt8

    (::Type{Cluster{C}})(cluster::UInt8) where {C} = 
        cluster > 0 && cluster <= C ? new{C}(cluster) : throw(ClusterException)
    (::Type{Cluster{C}})(cluster::Integer) where {C} = 
        Cluster{C}(UInt8(cluster))
end

Base.parent(c::Cluster{C}) where {C} = c.cluster
@generated function clusters(::Val{C}) where {C}
    :(SVector{C}(Cluster{C}.(1:C)))
end

Base.to_index(xs::AbstractArray, c::Cluster{C}) where {C} = 
    Base.to_index(xs, parent(c))

struct Z{C} <: AbstractVector{Cluster{C}}
    clusters::SVector{2, Cluster{C}}

    (::Type{Z{C}})(c::SVector{2, Cluster{C}}) where {C} = 
        new{C}(c) 
    (::Type{Z{C}})(a::Cluster{C}, b::Cluster{C}) where {C} = 
        Z{C}(SVector(a, b)) 
    (::Type{Z{C}})(x::Tuple{Cluster{C}, Cluster{C}}) where {C} = 
        Z{C}(x[1], x[2]) 
    (::Type{Z{C}})(a::Integer, b::Integer) where {C} = 
        Z{C}(Cluster{C}(a), Cluster{C}(b))
end

Base.parent(z::Z) = z.clusters
Base.size(z::Z) = size(parent(z))
Base.getindex(z::Z, i::Int) = getindex(parent(z), i)
Base.length(z::Z) = 3
Base.iterate(z::Z) = iterate(parent(z))
Base.IndexStyle(::Type{Z}) = IndexLinear()

@generated function zs(::Val{C}) where {C}
    :(
        it = clusters(Val(C));
        map(Z{C}, SMatrix(collect(product(it, it))))
    )
end

end
