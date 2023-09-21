module ForwardBackward

export FwdBwd, FwdBwdSite, forwardbackward, loglikelihood, fwd, bwd, scaling

using ..Utils
using ..Types
using ..Parameters
using ..Emission

struct FwdBwd{A<:Arr5, V<:Vec}
    fwd::A
    bwd::A
    scaling::V
end

struct FwdBwdSite{A<:Arr4}
    fwd::A
    bwd::A
    scaling::Float64
end

@inline Base.getindex(ab::FwdBwd, s::Integer) =
    FwdBwdSite(fwd(ab, s), bwd(ab, s), scaling(ab, s))

@inline fwd(ab::FwdBwd) = view(ab.fwd, :, :, :, :, :)
@inline fwd(ab::FwdBwd, s::Integer) = view(ab.fwd, s, :, :, :, :)
@inline bwd(ab::FwdBwd) = view(ab.bwd, :, :, :, :, :)
@inline bwd(ab::FwdBwd, s::Integer) = view(ab.bwd, s, :, :, :, :)
@inline scaling(ab::FwdBwd) = view(ab.scaling, :)
@inline scaling(ab::FwdBwd, s::Integer) = ab.scaling[s]

@inline fwd(ab::FwdBwdSite) = ab.fwd
@inline bwd(ab::FwdBwdSite) = ab.bwd
@inline scaling(ab::FwdBwdSite) = ab.scaling

@inline Types.sites(ab::FwdBwd) = length(scaling(ab))
@inline Types.clusters(ab::FwdBwd) = size(fwd(ab), 2)
@inline Types.populations(ab::FwdBwd) = size(fwd(ab), 4)
@inline Types.eachsite(ab::FwdBwd) = map(s -> ab[s], 1:sites(ab))

@inline Types.clusters(ab::FwdBwdSite) = size(fwd(ab), 1)
@inline Types.populations(ab::FwdBwdSite) = size(fwd(ab), 3)

loglikelihood(ab::FwdBwd) = reduce(+, log.(scaling(ab)))

function forwardbackward(gl::GlVec, par::ParInd)
    c, a = forward(gl, par)
    b = backward(gl, c, par)
    FwdBwd(a, b, c)
end

mutable struct FwdSums{A<:Arr3, M<:Mat, V<:Vec}
    z::A
    zz::M
    zy::M
    zzy::V
end

function Base.zeros(::Type{FwdSums}, C::Integer, K::Integer)
    FwdSums(
        zeros(C, K, K),
        zeros(K, K),
        zeros(C, K),
        zeros(K),
    )
end

@inline function forwardsums!(sums::FwdSums, a::Arr4)
    (C, C, K, K) = size(a)
    sums.z .= 0.
    sums.zz .= 0.
    sums.zy .= 0.
    sums.zzy .= 0.
    @inbounds for (z1, z2) in zzs(C)
        for (y1, y2) in yys(K)
            v = a[z1, z2, y1, y2]
            sums.z[z2, y1, y2] += v
            sums.zz[y1, y2] += v
            sums.zy[z1, y1] += v
            sums.zzy[y1] += v
        end
    end
end

function forward!(a::Arr4, prev::Arr4, prevsums::FwdSums, gl::Gl, par::ParSite)
    (C, K) = size(par)
    (; P, F, Q, er, et) = par
    @inbounds for (z1, z2) in zzs(C)
        emit = emission(gl, Z(z1, z2), P)
        for (y1, y2) in yys(K)
            q1, q2, f1, f2 = (Q[y1], Q[y2], F[z1, y1], F[z2, y2])
            staystay = et^2 * (
                er^2 * prev[z1, z2, y1, y2] +
                er * (1 - er) * (
                    f2 * prevsums.z[z1, y2, y1] +
                    f1 * prevsums.z[z2, y1, y2]
                ) +
                (1 - er)^2 * f1 * f2 * prevsums.zz[y1, y2]
            )
            stayjump = et * (1 - et) * (
                q2 * f2 * (
                    er * prevsums.zy[z1, y1] +
                    (1 - er) * f1 * prevsums.zzy[y1]
                ) +
                q1 * f1 * (
                    er * prevsums.zy[z2, y2] +
                    (1 - er) * f2 * prevsums.zzy[y2]
                )
            )
            jumpjump = (1 - et)^2 * q1 * q2 * f1 * f2
            a[z1, z2, y1, y2] = emit * (staystay + stayjump + jumpjump)
        end
    end
    cnorm!(a)
end

function forwardinit!(a::Arr4, gl::Gl, par::ParSite)
    (C, K) = size(par)
    (; P, F, Q, er, et) = par
    @inbounds for (z1, z2) in zzs(C)
        emit = emission(gl, Z(z1, z2), par.P)
        for (y1, y2) in yys(K)
            q1, q2, f1, f2 = (Q[y1], Q[y2], F[z1, y1], F[z2, y2])
            a[z1, z2, y1, y2] = emit * f1 * f2 * q1 * q2
        end
    end
    cnorm!(a)
end

function forward(gl::GlVec, par::ParInd)
    (S, C, K) = size(par)
    a = zeros(S, C, C, K, K)
    c = zeros(S)
    c[1] = forwardinit!(view(a, 1, :, :, :, :), gl[1], par[1])
    prevsums = zeros(FwdSums, C, K)
    @inbounds for s in 2:S
        curr = view(a, s, :, :, :, :)
        prev = view(a, s - 1, :, :, :, :)
        forwardsums!(prevsums, prev)
        c[s] = forward!(curr, prev, prevsums, gl[s], par[s])
    end
    (c, a)
end

mutable struct BwdSums{B<:Arr4, A<:Arr3, M<:Mat, V<:Vec}
    bemit::B
    z::A
    zz::M
    zy::M
    zzy::V
    zzyy::Float64
end

function Base.zeros(::Type{BwdSums}, C::Integer, K::Integer)
    BwdSums(
        zeros(C, C, K, K),
        zeros(C, K, K),
        zeros(K, K),
        zeros(C, K),
        zeros(K),
        0.0,
    )
end

@inline function backwardsums!(sums::BwdSums, b::Arr4, gl::Gl, par::ParSite)
    (; P, F, Q, er, et) = par
    (C, K) = size(par)
    sums.bemit .= 0.
    sums.z .= 0.
    sums.zz .= 0.
    sums.zy .= 0.
    sums.zzy .= 0.
    sums.zzyy = 0.
    @inbounds for (z1, z2) in zzs(C)
        emit = emission(gl, Z(z1, z2), P)
        for (y1, y2) in yys(K)
            q1, q2, f1, f2 = (Q[y1], Q[y2], F[z1, y1], F[z2, y2])
            bemit = emit * b[z1, z2, y1, y2]
            sums.bemit[z1, z2, y1, y2] += bemit
            sums.z[z1, y1, y2] += bemit * f2
            sums.zz[y1, y2] += bemit * f1 * f2
            sums.zy[z1, y1] += bemit * q2 * f2
            sums.zzy[y1] += bemit * q2 * f1 * f2
            sums.zzyy += bemit * q1 * q2 * f1 * f2
        end
    end
end

function backward!(b::Arr4, nextsums::BwdSums, c::Float64, par::ParSite)
    (C, K) = size(par)
    (; P, F, Q, er, et) = par
    @inbounds for (z1, z2) in zzs(C)
        for (y1, y2) in yys(K)
            staystay = et^2 * (
                er^2 * nextsums.bemit[z1, z2, y1, y2] + 
                er * (1 - er) * (
                    nextsums.z[z1, y1, y2] + 
                    nextsums.z[z2, y2, y1]
                ) +
                (1 - er)^2 * nextsums.zz[y1, y2]
            )
            stayjump = et * (1 - et) * (
                er * (nextsums.zy[z1, y1] + nextsums.zy[z2, y2]) +
                (1 - er) * (nextsums.zzy[y1] + nextsums.zzy[y2])
            )
            jumpjump = (1 - et)^2 * nextsums.zzyy
            b[z1, z2, y1, y2] = (staystay +  stayjump + jumpjump) / c
        end
    end
end

function backward(gl::GlVec, c::Vec, par::ParInd)
    (S, C, K) = size(par)
    b = zeros(S, C, C, K, K)
    b[S, :, :, :, :] .= 1
    nextsums = zeros(BwdSums, C, K)
    for s in reverse(2:S)
        curr = view(b, s - 1, :, :, :, :)
        next = view(b, s, :, :, :, :)
        backwardsums!(nextsums, next, gl[s], par[s])
        backward!(curr, nextsums, c[s], par[s])
    end
    b
end

end
