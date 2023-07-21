module Types

import StaticArrays: FieldVector, SMatrix
import Base.Iterators: product

export Vec, Mat, Arr, Allele, ref, alt, G, GVec, GMat, gs, alts, Gl, GlVec, GlMat,
    Cluster, Z, cs, zs, Pop, Y, ks, ys, sites, inds, clusters, populations, 
    site, ind, eachsite, eachind

const Vec = StridedVector{Float64}
const Mat = StridedMatrix{Float64}
const Arr = StridedArray{Float64, 3}

function sites end
function inds end
function clusters end
function populations end
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

const GVec = StridedVector{G}
const GMat = StridedMatrix{G}

G(a::Integer, b::Integer)::G = G(Allele(a), Allele(b))

alts(g::G) = sum(Int.(g))

const gs = SMatrix{2, 2}(G(0, 0), G(0, 1), G(1, 0), G(1, 1))

sites(g::GMat) = size(g, 2)
inds(g::GMat) = size(g, 1)
site(g::GMat, s::Int) = view(g, :, s)
ind(g::GMat, i::Int) = view(g, i, :)
eachsite(g::GMat) = eachcol(g)
eachind(g::GMat) = eachrow(g)

struct Gl <: FieldVector{3, Float64}
    ref::Float64
    het::Float64
    hom::Float64
end

const GlVec = StridedVector{Gl}
const GlMat = StridedMatrix{Gl}

Base.getindex(gl::Gl, g::G) = getindex(gl, alts(g) + 1)

sites(gl::GlMat) = size(gl, 2)
inds(gl::GlMat) = size(gl, 1)
site(gl::GlMat, s::Int) = view(gl, :, s)
ind(gl::GlMat, i::Int) = view(gl, i, :)
eachsite(gl::GlMat) = eachcol(gl)
eachind(gl::GlMat) = eachrow(gl)

const Cluster = UInt8

struct Z <: FieldVector{2, Cluster}
    fst::Cluster
    snd::Cluster
end

Z(a::Integer, b::Integer) = Z(Cluster(a), Cluster(b))

cs(C::Integer) = map(Cluster, 1:C)
zs(C::Integer) = map(Z, product(cs(C), cs(C)))

const Pop = UInt8

struct Y <: FieldVector{2, Pop}
    fst::Pop
    snd::Pop
end

Y(a::Integer, b::Integer) = Y(Pop(a), Pop(b))

ks(K::Integer) = map(Pop, 1:K)
ys(K::Integer) = map(Y, product(ks(K), ks(K)))

end
