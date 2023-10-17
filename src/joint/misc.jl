module Misc

export clusterfreq, clusterfreq!

using ..Utils
using ..Types
using ..Parameters

function clusterfreq!(p::Arr3, h::Mat, par::ParInd)
    S, C, K = size(par)
    @inbounds for (z1, z2) in zzs(C)
        p[1, z1, z2] = h[1, z1] * h[1, z2]
    end
    pprevsum = zeros(C)
    @inbounds for s in 2:S
        for z1 in zs(C)
            pprevsum[z1] = 0.0
            for z2 in zs(C)
                pprevsum[z1] += p[s - 1, z1, z2]
            end
        end
        er = par[s].er
        for (z1, z2) in zzs(C)
            h1, h2 = (h[s, z1], h[s, z2])
            p[s, z1, z2] = er^2 * p[s - 1, z1, z2] + 
                er * (1 - er) * (h[s, z1] * pprevsum[z2] + h[s, z2] * pprevsum[z1]) +
                (1 - er)^2 * h[s, z1] * h[s, z2] 
        end
    end
end

function clusterfreq(h::Mat, par::ParInd)
    S, C, K = size(par)
    p = zeros(S, C, C)
    clusterfreq!(p, h, par)
    p
end

end
