module ForwardBackward

export FwdBwd, FwdBwdSite, forwardbackward, fwd, bwd, scaling, loglikelihood

using ..Utils
using ..Types
using ..Parameters
using ..Emission

struct FwdBwd{A<:Arr3, V<:Vec}
    fwd::A
    bwd::A
    scaling::V
end

struct FwdBwdSite{M<:Mat}
    fwd::M
    bwd::M
    scaling::Float64
end

Base.size(ab::FwdBwd) = size(fwd(ab))
function Base.getindex(ab::FwdBwd, s::Integer)
    FwdBwdSite(
        view(ab.fwd, s, :, :), 
        view(ab.bwd, s, :, :), 
        ab.scaling[s]
    )
end

@inline fwd(ab) = ab.fwd
@inline bwd(ab) = ab.bwd
@inline scaling(ab) = ab.scaling

Types.sites(ab::FwdBwd) = length(scaling(ab))
Types.clusters(ab::FwdBwd) = size(fwd(ab), 2)

Types.eachsite(ab::FwdBwd) = map(s -> ab[s], 1:sites(ab))
Types.clusters(ab::FwdBwdSite) = size(fwd(ab), 1)

loglikelihood(ab::FwdBwd) = reduce(+, log.(scaling(ab)))

function forwardbackward(gl::GlVec, par::Par)
    c, a = forward(gl, par)
    b = backward(gl, c, par)
    FwdBwd(a, b, c)
end

mutable struct FwdSums{V<:Vec}
    z::V
end

function Base.zeros(::Type{FwdSums}, C::Integer)
    FwdSums(zeros(C))
end

@inline function forwardsums!(sums::FwdSums, a::Mat)
    (C, C) = size(a)
    sums.z .= 0.0
    @inbounds for (z1, z2) in zzs(C)
        sums.z[z1] += a[z1, z2]
    end
end

function forwardinit!(a::Mat, gl::Gl, par::ParSite)
    @inbounds for (z1, z2) in zzs(clusters(par))
        emit = emission(gl, Z(z1, z2), par.P)
        f1, f2 = (par.F[z1], par.F[z2])
        a[z1, z2] = emit * f1 * f2
    end
    cnorm!(a)
end

function forward!(a::Mat, prev::Mat, prevsums::FwdSums, gl::Gl, par::ParSite)
    C = clusters(par)
    (; P, F, er) = par
    @inbounds for (z1, z2) in zzs(C)
        emit = emission(gl, Z(z1, z2), P)
        a[z1, z2] = emit * (
            er^2 * prev[z1, z2] + 
            er * (1 - er) * (F[z1] * prevsums.z[z2] + F[z2] * prevsums.z[z1]) +
            (1 - er)^2 * F[z1] * F[z2]
        )
    end
    cnorm!(a)
end

function forward(gl::GlVec, par::Par)
    (S, C) = size(par)
    a = zeros(Float64, S, C, C)
    c = zeros(Float64, S)
    c[1] = forwardinit!(view(a, 1, :, :), gl[1], par[1])
    prevsums = zeros(FwdSums, C)
    for s in 2:S
        curr = view(a, s, :, :)
        prev = view(a, s - 1, :, :)
        forwardsums!(prevsums, prev)
        c[s] = forward!(curr, prev, prevsums, gl[s], par[s])
    end
    (c, a)
end

mutable struct BwdSums{M<:Mat, V<:Vec}
    bemit::M
    z::V
    zz::Float64
end

function Base.zeros(::Type{BwdSums}, C::Integer)
    BwdSums(
        zeros(C, C),
        zeros(C),
        0.0,
    )
end

@inline function backwardsums!(sums::BwdSums, b::Mat, gl::Gl, par::ParSite)
    C = clusters(par)
    sums.bemit .= 0.0
    sums.z .= 0.0
    sums.zz = 0.0
    @inbounds for (z1, z2) in zzs(C)
        emit = emission(gl, Z(z1, z2), par.P)
        f1, f2 = (par.F[z1], par.F[z2])
        bemit = emit * b[z1, z2]
        sums.bemit[z1, z2] += bemit
        sums.z[z1] += bemit * f2
        sums.zz += bemit * f1 * f2
    end
end

function backward!(b::Mat, nextsums::BwdSums, c::Float64, par::ParSite)
    C = clusters(par)
    (; P, F, er) = par
    @inbounds for (z1, z2) in zzs(C)
        b[z1, z2] = (
            er^2 * nextsums.bemit[z1, z2] +
            er * (1 - er) * (nextsums.z[z1] + nextsums.z[z2]) +
            (1 - er)^2 * nextsums.zz
        ) / c
    end
end

function backward(gl::GlVec, c::Vec, par::Par)
    (S, C) = size(par)
    b = zeros(Float64, S, C, C)
    b[S, :, :] .= 1.0
    nextsums = zeros(BwdSums, C)
    for s in reverse(2:S)
        curr = view(b, s - 1, :, :)
        next = view(b, s, :, :)
        backwardsums!(nextsums, next, gl[s], par[s])
        backward!(curr, nextsums, c[s], par[s])
    end
    b
end

end
