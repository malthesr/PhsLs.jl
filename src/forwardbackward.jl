module ForwardBackward

export FwdBwd, FwdBwdSite, Fwd, Bwd, fwd, bwd, scaling, loglikelihood

using ..Utils
using ..Types
using ..Parameters
using ..Emission

struct FwdBwd{A<:Arr{Float64}}
    fwd::A
    bwd::A
    scaling::Vector{Float64}
end

function FwdBwd(gl::Vec{Gl}, par::Par)
    c, a = forward(gl, par)
    b = backward(gl, c, par)
    FwdBwd(a, b, c)
end

fwd(ab::FwdBwd) = ab.fwd
bwd(ab::FwdBwd) = ab.bwd
scaling(ab::FwdBwd) = ab.scaling

function Base.getindex(ab::FwdBwd, s::Int)
    FwdBwdSite(
        view(ab.fwd, s, :, :), 
        view(ab.bwd, s, :, :), 
        ab.scaling[s]
    )
end

Types.sites(ab::FwdBwd) = length(scaling(ab))
Types.eachsite(ab::FwdBwd) = map(s -> ab[s], 1:sites(ab))

loglikelihood(ab::FwdBwd) = reduce(+, log.(scaling(ab)))

struct FwdBwdSite{M<:Mat{Float64}}
    fwd::M
    bwd::M
    scaling::Float64
end

fwd(ab::FwdBwdSite) = ab.fwd
bwd(ab::FwdBwdSite) = ab.bwd
scaling(ab::FwdBwdSite) = ab.scaling

function forward(gl::Vec{Gl}, par::Par)
    (S, C) = size(par)
    a = zeros(Float64, S, C, C)
    c = zeros(Float64, S)
    (c[1], a[1, :, :]) = cnorm(emission(gl[1], par[1]) .* 
        outer(jumpclusterfreqs(par[1])))
    for s in 2:S
        e = stayfreq(par[s])
        sums = symouter(jumpclusterfreqs(par[s]), colsum(a[s - 1, :, :]))
        (c[s], a[s, :, :]) = cnorm(emission(gl[s], par[s]) .* (
            e^2 .* a[s - 1, :, :] .+ 
            e .* (1 - e) .* sums .+
            (1 - e)^2 .* outer(jumpclusterfreqs(par[s]))
        ))
    end
    (c, a)
end

function backward(gl::Vec{Gl}, c::Vec{Float64}, par::Par)
    (S, C) = size(par)
    b = zeros(Float64, S, C, C)
    b[S, :, :] .= 1.0
    for s in reverse(2:S)
        e = stayfreq(par[s])
        be = emission(gl[s], par[s]) .* b[s, :, :]
        colsums = colsum(jumpclusterfreqs(par[s]) .* be)
        sums = outer(+, colsums, colsums) 
        allsum = sum(outer(jumpclusterfreqs(par[s])) .* be)
        b[s - 1, :, :] = (
            e^2 .* be .+
            e .* (1 - e) .* sums .+
            (1 - e)^2 .* allsum
        ) ./ c[s]
    end
    b
end

end
