module ForwardBackward

using StaticArrays: SMatrix

export FwdBwd, FwdBwdSite, Fwd, Bwd, fwd, bwd, scaling, loglikelihood

using ..Utils
using ..Types
using ..Parameters
using ..Emission

const Fwd{C} = SMatrix{C, C, Float64}
const Bwd{C} = SMatrix{C, C, Float64}

struct FwdBwd{C}
    fwd::Vector{Fwd{C}}
    bwd::Vector{Bwd{C}}
    scaling::Vector{Float64}
end

function FwdBwd(gl::Vec{Gl}, par::Par{C}) where {C}
    c, a = forward(gl, par)
    b = backward(gl, c, par)
    FwdBwd{C}(a, b, c)
end

fwd(ab::FwdBwd{C}) where {C} = ab.fwd
bwd(ab::FwdBwd{C}) where {C} = ab.bwd
scaling(ab::FwdBwd{C}) where {C} = ab.scaling

Base.getindex(ab::FwdBwd{C}, s::Int) where {C} = 
    FwdBwdSite{C}(fwd(ab)[s], bwd(ab)[s], scaling(ab)[s])

Types.sites(ab::FwdBwd{C}) where {C} = length(scaling(ab))
Types.eachsite(ab::FwdBwd{C}) where {C} = map(s -> ab[s], 1:sites(ab))

loglikelihood(ab::FwdBwd) = reduce(+, log.(scaling(ab)))

struct FwdBwdSite{C}
    fwd::Fwd{C}
    bwd::Bwd{C}
    scaling::Float64
end

fwd(ab::FwdBwdSite{C}) where {C} = ab.fwd
bwd(ab::FwdBwdSite{C}) where {C} = ab.bwd
scaling(ab::FwdBwdSite{C}) where {C} = ab.scaling

function forward(gl::Vec{Gl}, par::Par{C}) where {C}
    S = sites(par)
    a = zeros(Fwd{C}, S)
    c = zeros(Float64, S)
    (c[1], a[1]) = cnorm(emission(gl[1], par[1]) .* 
        outer(jumpclusterfreqs(par[1])))
    for s in 2:S
        e = stayfreq(par[s])
        sums = symouter(jumpclusterfreqs(par[s]), colsum(a[s - 1]))
        (c[s], a[s]) = cnorm(emission(gl[s], par[s]) .* (
            e^2 .* a[s - 1] .+ 
            e .* (1 - e) .* sums .+
            (1 - e)^2 .* outer(jumpclusterfreqs(par[s]))
        ))
    end
    (c, a)
end

function backward(gl::Vec{Gl}, c::Vec{Float64}, par::Par{C}) where {C}
    S = sites(par)
    b = zeros(Bwd{C}, S)
    b[S] = ones(Bwd{C})
    for s in reverse(2:S)
        e = stayfreq(par[s])
        be = emission(gl[s], par[s]) .* b[s]
        colsums = colsum(jumpclusterfreqs(par[s]) .* be)
        sums = outer(+, colsums, colsums) 
        allsum = sum(outer(jumpclusterfreqs(par[s])) .* be)
        b[s - 1] = (
            e^2 .* be .+
            e .* (1 - e) .* sums .+
            (1 - e)^2 .* allsum
        ) ./ c[s]
    end
    b
end

end
