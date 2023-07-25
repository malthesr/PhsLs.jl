module Expectation

export clusteralleleexpect, clusterexpect, popexpect, popexpect, jumpclusterpopexpect

using Base.Iterators: product

using ..Utils
using ..Types
using ..Parameters
using ..Emission
using ..ForwardBackward
using ..Posterior

function clusteralleleexpect(gl::GlVec, ab::FwdBwd, par::ParInd)
    (S, C) = (sites(par), clusters(par))
    expect = zeros(Float64, S, C, 2)
    for s in 1:S
        af = P(par[s])
        k = clusterpost(ab[s]) ./ emission(gl[s], af)
        for z in cs(C)
            expect[s, z, 1] = sum(
                k[z, :] .* (1 - af[z]) .* (gl[s][1] .* (1 .- af) .+ gl[s][2] .* af)
            )
            expect[s, z, 2] = sum(
                k[z, :] .* af[z] .* (gl[s][2] .* (1 .- af) + gl[s][3] .* af)
            )
        end
    end
    expect
end

function clusterexpect(ab::FwdBwdSite)
    colsum(clusterpost(ab))
end

function popexpect(gl::GlVec, ab::FwdBwd, par::ParInd)
    (S, C, K) = (sites(par), clusters(par), populations(par))
    expect = zeros(Float64, S, K)
    expect[1, :, :] = colsum(outer(clusterexpect(ab[1]), Q(par[1])))
    aprevsums = zeros(Float64, C)
    (allzs, allks) = (zs(C), ks(K))
    @inbounds for s in 2:S
        stay = stayfreq(par[s])
        jump = jumpfreq(par[s])
        aprev = view(fwd(ab[s - 1]), :, :)
        colsum!(view(aprevsums, :), aprev)
        b = view(bwd(ab[s]), :, :)
        c = scaling(ab[s])
        for y1 in view(allks, :)
            for z in view(allzs, :)
                tmp = 0
                for y2 in view(allks, :)
                    q2 = Q(par[s])[y2]
                    f1, f2 = (F(par[s])[z[1], y1], F(par[s])[z[2], y2])
                    tmp += q2 * (
                        stay^2 * aprev[z...] +
                        stay * jump * (
                            f1 * aprevsums[z[2]] + 
                            f2 * aprevsums[z[1]]
                        ) +
                        jump^2 * f1 * f2
                    )
                end
                emit = emission(gl[s], z, P(par[s])) 
                expect[s, y1] += emit * b[z...] * tmp
            end
            q1 = Q(par[s])[y1]
            expect[s, y1] *= q1 / c
        end
    end
    expect
end

function jumpclusterpopexpect(gl::GlVec, ab::FwdBwd, par::ParInd)
    (S, C, K) = (sites(par), clusters(par), populations(par))
    expect = zeros(Float64, S, C, K)
    expect[1, :, :] = outer(clusterexpect(ab[1]), Q(par[1]))
    aprevsums = zeros(Float64, C)
    (allcs, allks) = (cs(C), ks(K))
    @inbounds for s in 2:S
        stay = stayfreq(par[s])
        jump = jumpfreq(par[s])
        colsum!(aprevsums, view(fwd(ab[s - 1]), :, :))
        b = view(bwd(ab[s]), :, :)
        c = scaling(ab[s])
        for z1 in view(allcs, :)
            for y1 in view(allks, :)
                q1, f1 = (Q(par[s])[y1], F(par[s])[z1, y1])
                for z2 in view(allcs, :)
                    tmp = 0
                    for y2 in view(allks, :)
                        q2, f2 = (Q(par[s])[y2], F(par[s])[z2, y2])
                        tmp += q2 * (stay * jump * aprevsums[z2] + jump^2 * f2)
                    end
                    emit = emission(gl[s], Z(z1, z2), P(par[s])) 
                    expect[s, z1, y1] += emit * b[z1, z2] * tmp
                end
                expect[s, z1, y1] *= f1 * q1 / c
            end
        end
    end
    expect
end

end
