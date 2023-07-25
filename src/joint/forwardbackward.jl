module ForwardBackward

export FwdBwd, FwdBwdSite, forwardbackward, fwd, bwd, scaling, loglikelihood

using ..Utils
using ..Types
using ..Parameters
using ..Emission
using ..Transition

struct FwdBwd{A<:Arr, V<:Vec}
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

fwd(ab) = ab.fwd
bwd(ab) = ab.bwd
scaling(ab) = ab.scaling

Types.sites(ab::FwdBwd) = length(scaling(ab))
Types.clusters(ab::FwdBwd) = size(fwd(ab), 2)
Types.eachsite(ab::FwdBwd) = map(s -> ab[s], 1:sites(ab))
Types.clusters(ab::FwdBwdSite) = size(fwd(ab), 1)

loglikelihood(ab::FwdBwd) = reduce(+, log.(scaling(ab)))

function forwardbackward(gl::GlVec, par::ParInd)
    c, a = forward(gl, par)
    b = backward(gl, c, par)
    FwdBwd(a, b, c)
end

function forward(gl::GlVec, par::ParInd)
    S, C, K = (sites(par), clusters(par), populations(par))
    c, a = (zeros(Float64, S), zeros(Float64, S, C, C))
    allz, ally = (zs(C), ys(K))
    c[1], a[1, :, :] = cnorm(emission(gl[1], P(par[1])) .* 
        outer(rowsum(F(par[1]))))
    aprevsums = zeros(Float64, C)
    @inbounds for s in 2:S
        stay = stayfreq(par[s])
        jump = jumpfreq(par[s])
        colsum!(view(aprevsums, :), view(a, s - 1, :, :))
        for z in view(allz, :, :)
            for y in view(ally, :, :)
                q1, q2 = (Q(par[s])[y[1]], Q(par[s])[y[2]])
                f1, f2 = (F(par[s])[z[1], y[1]], F(par[s])[z[2], y[2]])
                a[s, z...] += q1 * q2 * (
                    stay^2 * a[s - 1, z...] +
                    stay * jump * (
                        f1 * aprevsums[z[2]] + 
                        f2 * aprevsums[z[1]]
                    ) +
                    jump^2 * f1 * f2
                )
            end
            a[s, z...] *= emission(gl[s], z, P(par[s]))
        end
        c[s] = cnorm!(view(a, s, :, :))
    end
    (c, a)
end

function backward(gl::GlVec, c::Vec, par::ParInd)
    (S, C, K) = size(F(par))
    b = zeros(Float64, S, C, C)
    b[S, :, :] .= 1.0
    allz, ally = (zs(C), ys(K))
    (sums1, sums2) = (zeros(Float64, C), zeros(Float64, C))
    bemit = zeros(Float64, C, C)
    @inbounds for s in reverse(2:S)
        stay = stayfreq(par[s])
        jump = jumpfreq(par[s])
        emission!(bemit, gl[s], P(par[s]))
        bemit .*= b[s, :, :]
        for y in view(ally, :, :)
            q1, q2 = (Q(par[s])[y[1]], Q(par[s])[y[2]])
            sums1 .= 0.0; sums2 .= 0.0; allsum = 0.0
            for z in view(allz, :, :)
                f1, f2 = (F(par[s])[z[1], y[1]], F(par[s])[z[2], y[2]])
                sums1[z[1]] += bemit[z[1], z[2]] * f2
                sums2[z[2]] += bemit[z[2], z[1]] * f1
                allsum += bemit[z...] * f1 * f2
            end
            for z in view(allz, :, :)
                b[s - 1, z...] += q1 * q2 * (
                    stay^2 * bemit[z...] +
                    stay * jump * (sums1[z[1]] + sums2[z[2]]) +
                    jump^2 * allsum
                )
            end
        end
        b[s - 1, :, :] ./= c[s]
    end
    b
end

end
