module Input

export Beagle

using Base.Iterators: partition
using GZip: gzopen, GZipStream

using ..Types: Gl

struct Beagle
    samples::Vector{String}
    chr::String
    pos::Vector{UInt64}
    gl::Matrix{Gl}
end

samples(beagle::Beagle) = beagle.samples
chr(beagle::Beagle) = beagle.chr
positions(beagle::Beagle) = beagle.pos
gl(beagle::Beagle) = beagle.gl

struct BeagleRow 
    chr::String
    pos::UInt64
    gl::Vector{Gl}
end

function Base.parse(::Type{BeagleRow}, s; indices=nothing)
    line = split(strip(s), isspace)
    chr, rawpos = split(line[1], "_")
    pos = parse(UInt64, rawpos)
    values = parse.(Float64, line[4:end])
    gl = map(Gl, partition(values, 3))
    if !isnothing(indices)
        gl = gl[indices]
    end
    BeagleRow(chr, pos, gl)
end

function Base.read(io::GZipStream, ::Type{Beagle}; samples=nothing)
    header = readline(io)
    allsamples = split(strip(header), isspace)[4:3:end]
    if !isnothing(samples)
        @assert(all(s in allsamples for s in samples))
        indices = indexin(samples, allsamples)
    else
        samples = allsamples
        indices = nothing
    end
    lines = map(s -> parse(BeagleRow, s, indices=indices), eachline(io))
    chr = unique(getfield.(lines, :chr))
    @assert(length(chr) == 1, "more than one chr")
    pos = getfield.(lines, :pos)
    gl = reduce(hcat, getfield.(lines, :gl))
    Beagle(samples, chr[1], pos, gl)
end

Base.read(s::AbstractString, ::Type{Beagle}; kwargs...) = 
    read(gzopen(s), Beagle; kwargs...)

end
