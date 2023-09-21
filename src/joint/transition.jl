module Transition

export transition

using ..Types
using ..Parameters

@inline function transition(z::Cluster, y::Ancestry, zprev::Cluster, yprev::Ancestry, par::ParSite)
    (; P, F, Q, er, et) = par
    p = (1 - et)Q[y]F[z, y] 
    if y == yprev
        p += et * ((z == zprev)er + (1 - er)F[z, y])
    end
    p
end

end
