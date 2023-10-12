module ForwardBackward

export FwdBwd, FwdBwdSite, forwardbackward, forwardbackward!, loglikelihood

using ..Utils
using ..Types
using ..Parameters
using ..Emission

struct FwdBwd{A<:Arr3, V<:Vec}
    fwd::A
    bwd::A
    scaling::V
end

function Base.zeros(::Type{FwdBwd}, S::Integer, C::Integer)
    FwdBwd(zeros(S, C, C), zeros(S, C, C), zeros(S))
end

struct FwdBwdSite{M<:Mat}
    fwd::M
    bwd::M
    scaling::Float64
end

Base.size(ab::FwdBwd) = size(ab.fwd)
function Base.getindex(ab::FwdBwd, s::Integer)
    FwdBwdSite(
        view(ab.fwd, s, :, :), 
        view(ab.bwd, s, :, :), 
        ab.scaling[s]
    )
end

Types.sites(ab::FwdBwd) = length(ab.scaling)
Types.clusters(ab::FwdBwd) = size(ab.fwd, 2)

Types.eachsite(ab::FwdBwd) = map(s -> ab[s], 1:sites(ab))
Types.clusters(ab::FwdBwdSite) = size(ab.fwd, 1)

loglikelihood(ab::FwdBwd) = reduce(+, log.(ab.scaling))

function forwardbackward!(ab::FwdBwd, gl::GlVec, h::Mat, par::ParInd)
    forward!(ab.fwd, ab.scaling, gl, h, par)
    backward!(ab.bwd, gl, ab.scaling, h, par)
end

function forwardbackward(gl::GlVec, h::Mat, par::ParInd)
    (S, C, K) = size(par)
    ab = zeros(FwdBwd, S, C)
    forwardbackward!(ab, gl, h, par)
    ab
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

function forwardinit!(a::Mat, gl::Gl, h::Vec, par::ParSite)
    @inbounds for (z1, z2) in zzs(clusters(par))
        emit = emission(gl, Z(z1, z2), par.P)
        a[z1, z2] = emit * h[z1] * h[z2]
    end
    cnorm!(a)
end

function forward!(a::Mat, prev::Mat, prevsums::FwdSums, gl::Gl, h::Vec, par::ParSite)
    C = clusters(par)
    er = par.er
    @inbounds for (z1, z2) in zzs(C)
        emit = emission(gl, Z(z1, z2), par.P)
        a[z1, z2] = emit * (
            er^2 * prev[z1, z2] + 
            er * (1 - er) * (h[z1] * prevsums.z[z2] + h[z2] * prevsums.z[z1]) +
            (1 - er)^2 * h[z1] * h[z2]
        )
    end
    cnorm!(a)
end

function forward!(a::Arr3, c::Vec, gl::GlVec, h::Mat, par::ParInd)
    (S, C, K) = size(par)
    c[1] = forwardinit!(view(a, 1, :, :), gl[1], h[1, :], par[1])
    prevsums = zeros(FwdSums, C)
    for s in 2:S
        curr = view(a, s, :, :)
        prev = view(a, s - 1, :, :)
        forwardsums!(prevsums, prev)
        c[s] = forward!(curr, prev, prevsums, gl[s], h[s, :], par[s])
    end
end

function forward(gl::GlVec, h::Mat, par::ParInd)
    (S, C, K) = size(par)
    a = zeros(Float64, S, C, C)
    c = zeros(Float64, S)
    forward!(a, c, gl, h, par)
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

@inline function backwardsums!(sums::BwdSums, b::Mat, gl::Gl, h::Vec, par::ParSite)
    C = clusters(par)
    sums.bemit .= 0.0
    sums.z .= 0.0
    sums.zz = 0.0
    @inbounds for (z1, z2) in zzs(C)
        emit = emission(gl, Z(z1, z2), par.P)
        bemit = emit * b[z1, z2]
        sums.bemit[z1, z2] += bemit
        sums.z[z1] += bemit * h[z2]
        sums.zz += bemit * h[z1] * h[z2]
    end
end

function backward!(b::Mat, nextsums::BwdSums, c::Float64, par::ParSite)
    C = clusters(par)
    er = par.er
    @inbounds for (z1, z2) in zzs(C)
        b[z1, z2] = (
            er^2 * nextsums.bemit[z1, z2] +
            er * (1 - er) * (nextsums.z[z1] + nextsums.z[z2]) +
            (1 - er)^2 * nextsums.zz
        ) / c
    end
end

function backward!(b::Arr3, gl::GlVec, c::Vec, h::Mat, par::ParInd)
    (S, C, K) = size(par)
    b[S, :, :] .= 1.0
    nextsums = zeros(BwdSums, C)
    for s in reverse(2:S)
        curr = view(b, s - 1, :, :)
        next = view(b, s, :, :)
        backwardsums!(nextsums, next, gl[s], h[s, :], par[s])
        backward!(curr, nextsums, c[s], par[s])
    end
end

function backward(gl::GlVec, c::Vec, h::Mat, par::ParInd)
    (S, C, K) = size(par)
    b = zeros(Float64, S, C, C)
    backward!(b, gl, c, h, par)
    b
end

end
