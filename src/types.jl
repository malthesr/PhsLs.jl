module Types

import StaticArrays: FieldVector, SMatrix
import Base.Iterators: product

export Allele, ref, alt, G, gs, alts, Gl, Cluster, cs, Z, zs,
    Vec, Mat, Arr, sites, inds, clusters, site, ind, eachsite, eachind

const Vec{T} = StridedVector{T}
const Mat{T} = StridedMatrix{T}
const Arr{T} = StridedArray{T, 3}

function sites end
function inds end
function clusters end
function site end
function ind end
function eachsite end
function eachind end

@enum Allele begin
    ref = 0
    alt = 1
end

Base.to_index(A, a::Allele) = Base.to_index(A, Int(a) + 1)

struct G <: FieldVector{2, Allele}
    fst::Allele
    snd::Allele
end

G(a::Integer, b::Integer)::G = G(Allele(a), Allele(b))

alts(g::G) = sum(Int.(g))

const gs = SMatrix{2, 2}(G(0, 0), G(0, 1), G(1, 0), G(1, 1))

sites(g::Mat{G}) = size(g, 2)
inds(g::Mat{G}) = size(g, 1)
site(g::Mat{G}, s::Int) = view(g, :, s)
ind(g::Mat{G}, i::Int) = view(g, i, :)
eachsite(g::Mat{G}) = eachcol(g)
eachind(g::Mat{G}) = eachrow(g)

struct Gl <: FieldVector{3, Float64}
    ref::Float64
    het::Float64
    hom::Float64
end

Base.getindex(gl::Gl, g::G) = getindex(gl, alts(g) + 1)

sites(gl::Mat{Gl}) = size(gl, 2)
inds(gl::Mat{Gl}) = size(gl, 1)
site(gl::Mat{Gl}, s::Int) = view(gl, :, s)
ind(gl::Mat{Gl}, i::Int) = view(gl, i, :)
eachsite(gl::Mat{Gl}) = eachcol(gl)
eachind(gl::Mat{Gl}) = eachrow(gl)

const Cluster = UInt8

struct Z <: FieldVector{2, Cluster}
    fst::Cluster
    snd::Cluster
end

Z(a::Integer, b::Integer) = Z(Cluster(a), Cluster(b))

cs(C::Integer) = map(Cluster, 1:C)
zs(C::Integer) = map(Z, product(cs(C), cs(C)))

end
