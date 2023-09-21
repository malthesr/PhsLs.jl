module Types

import StaticArrays: FieldVector, SMatrix
import Base.Iterators: product

export Vec, Mat, Arr3, Arr4, Arr5, Allele, ref, alt, G, gs, Gl, GlVec, GlMat,
    Cluster, Z, zs, zzs, Ancestry, Y, ys, yys, Jump, J, js,
    inds, sites, clusters, populations, site, ind, eachsite, eachind

const Vec = StridedVector{Float64}
const Mat = StridedMatrix{Float64}
const Arr3 = StridedArray{Float64, 3}
const Arr4 = StridedArray{Float64, 4}
const Arr5 = StridedArray{Float64, 5}

function inds end
function sites end
function clusters end
function populations end

@inline function eachind(iters...)
    zip(map(eachind, iters)...)
end
@inline function eachsite(iters...)
    zip(map(eachsite, iters)...)
end

@enum Allele begin
    ref = 0
    alt = 1
end

@inline Base.to_index(A, a::Allele) = Base.to_index(A, Int(a) + 1)

struct G <: FieldVector{2, Allele}
    fst::Allele
    snd::Allele
end

G(a::Integer, b::Integer)::G = G(Allele(a), Allele(b))

const gs = SMatrix{2, 2}(G(0, 0), G(0, 1), G(1, 0), G(1, 1))

struct Gl <: FieldVector{3, Float64}
    ref::Float64
    het::Float64
    hom::Float64
end

const GlVec = StridedVector{Gl}
const GlMat = StridedMatrix{Gl}

@inline Base.getindex(gl::Gl, g::G) = getindex(gl, sum(Int.(g)) + 1)

ind(gl::GlMat, i::Int) = view(gl, i, :)
site(gl::GlMat, s::Int) = view(gl, :, s)

@inline inds(gl::GlMat) = size(gl, 1)
@inline sites(gl::GlMat) = size(gl, 2)

@inline eachind(gl::GlMat) = eachrow(gl)
@inline eachsite(gl::GlVec) = gl

const Cluster = UInt8

struct Z <: FieldVector{2, Cluster}
    fst::Cluster
    snd::Cluster
end

@inline Z(a::Integer, b::Integer) = Z(Cluster(a), Cluster(b))

@inline zs(C::Integer) = map(Cluster, 1:UInt8(C))
@inline zzs(C::Integer) = product(zs(C), zs(C))

const Ancestry = UInt8

struct Y <: FieldVector{2, Ancestry}
    fst::Ancestry
    snd::Ancestry
end

@inline Y(a::Integer, b::Integer) = Y(Ancestry(a), Ancestry(b))

@inline ys(K::Integer) = map(Ancestry, 1:UInt8(K))
@inline yys(K::Integer) = product(ys(K), ys(K))

@enum Jump begin
    stay = 0
    jump = 1
end

struct J <: FieldVector{2, Jump}
    fst::Jump
    snd::Jump
end

@inline J(a::Integer, b::Integer)::J = J(Jump(a), Jump(b))

const js = SMatrix{2, 2}(J(0, 0), J(0, 1), J(1, 0), J(1, 1))

end
