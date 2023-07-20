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

Base.size(ab::FwdBwd) = size(fwd(ab))

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
    emissionbuf = emission(gl[1], allelefreqs(par[1]))
    (c[1], a[1, :, :]) = cnorm(emissionbuf .* 
        outer(jumpclusterfreqs(par[1])))
    for s in 2:S
        e = stayfreq(par[s])
        sums = symouter(jumpclusterfreqs(par[s]), colsum(a[s - 1, :, :]))
        emission!(emissionbuf, gl[s], allelefreqs(par[s]))
        (c[s], a[s, :, :]) = cnorm(emissionbuf .* (
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
    buf = zeros(Float64, C, C)
    for s in reverse(2:S)
        e = stayfreq(par[s])
        emission!(buf, gl[s], allelefreqs(par[s]))
        buf .*= b[s, :, :]
        colsums = colsum(jumpclusterfreqs(par[s]) .* buf)
        sums = outer(+, colsums, colsums) 
        allsum = sum(outer(jumpclusterfreqs(par[s])) .* buf)
        b[s - 1, :, :] = (
            e^2 .* buf .+
            e .* (1 - e) .* sums .+
            (1 - e)^2 .* allsum
        ) ./ c[s]
    end
    b
end

function Base.read(prefix::AbstractString, ::Type{FwdBwd}, ind::Integer; kwargs...)
    fwdpath = prefix * ".alpha.bin"
    bwdpath = prefix * ".beta.bin"
    read(fwdpath, bwdpath, FwdBwd, ind; kwargs...)
end

function Base.read(fwdpath::AbstractString, bwdpath::AbstractString, ::Type{FwdBwd}, ind::Integer; kwargs...)
    fwdio = open(fwdpath, "r")
    bwdio = open(bwdpath, "r")
    read(fwdio, bwdio, FwdBwd, ind; kwargs...)
end

function Base.read(fwdio::IO, bwdio::IO, ::Type{FwdBwd}, ind::Integer; kwargs...)
    a = readind(fwdio, ind; kwargs...)
    b = readind(bwdio, ind; kwargs...)
    c = fill(NaN, size(a, 1)) # FIXME: Get Zilong to print the scaling factors
    FwdBwd(a, b, c)
end

function readhdr(io::IO)
    seekstart(io)
    C = read(io, UInt32)
    I = read(io, UInt32)
    S = read(io, UInt32)
    (C, I, S)
end

function readind(io::IO, ind::Integer; maxsites=nothing)
    (C, I, S) = readhdr(io)
    @assert(ind <= I)
    offset = 3 * sizeof(UInt32) + sizeof(Float64) * (ind - 1) * C^2 * S
    seek(io, offset)
    sites = isnothing(maxsites) ? S : min(maxsites, S)
    data = Array{Float64}(undef, C, C, sites)
    read!(io, data)
    permutedims(data, (3, 2, 1))
end

end
