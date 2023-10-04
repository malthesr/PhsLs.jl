module Input

export Beagle

using Base.Iterators: partition
using GZip: gzopen, GZipStream

using ..Types: Gl
using ..Utils

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

struct BeagleChr
    chr::String
    pos::Vector{UInt64}
    gl::Matrix{Gl}
end

struct Beagle
    samples::Vector{String}
    chrs::Vector{BeagleChr}
end

mutable struct BeagleReader
    io::GZipStream
    samples::Vector{String}
    sampleindices::Option{Vector{Int}}
    chrs::Option{Vector{String}}
    peeked::Option{BeagleRow}
end

function BeagleReader(io::GZipStream;
                      samples::Option{<:AbstractVector{String}}=nothing,
                      chrs::Option{<:AbstractVector{String}}=nothing)
    header = readline(io)
    allsamples = split(strip(header), isspace)[4:3:end]
    if !isnothing(samples)
        @assert(all(s in allsamples for s in samples))
        sampleindices = indexin(samples, allsamples)
    else
        samples = allsamples
        sampleindices = nothing
    end
    BeagleReader(io, samples, sampleindices, chrs, nothing)
end

Base.eof(reader::BeagleReader) = isnothing(peek(reader))

function Base.peek(reader::BeagleReader)
    if isnothing(reader.peeked)
        reader.peeked = read(reader, BeagleRow)
    end
    reader.peeked
end

function Base.skip(reader::BeagleReader, ::Type{BeagleRow})
    if !isnothing(reader.peeked)
        reader.peeked = nothing
    else
        s = readline(reader.io)
    end
end

function Base.read(reader::BeagleReader, ::Type{BeagleRow})
    if !isnothing(reader.peeked)
        row = reader.peeked
        reader.peeked = nothing
        return row
    end

    s = readline(reader.io)
    if s == ""
        nothing
    else
        parse(BeagleRow, s, indices=reader.sampleindices)
    end
end

function Base.read(reader::BeagleReader, ::Type{BeagleChr})
    chr = peek(reader).chr
    @assert(!isnothing(chr))
    lines = BeagleRow[]
    while true
        if eof(reader) || reader.peeked.chr != chr
            break
        end
        line = read(reader, BeagleRow)
        push!(lines, line)
    end
    pos = getfield.(lines, :pos)
    gl = reduce(hcat, getfield.(lines, :gl))
    BeagleChr(chr, pos, gl)
end

function Base.skip(reader::BeagleReader, ::Type{BeagleChr})
    chr = peek(reader).chr
    @assert(!isnothing(chr))
    while true
        if eof(reader) || reader.peeked.chr != chr
            break
        end
        skip(reader, BeagleRow)
    end
end

function Base.read(reader::BeagleReader, ::Type{Beagle})
    chrs = BeagleChr[]
    while !eof(reader)
        if !isnothing(reader.chrs) && !(peek(reader).chr in reader.chrs)
            skip(reader, BeagleChr)
            continue
        end
        chr = read(reader, BeagleChr)
        push!(chrs, chr)
    end
    Beagle(reader.samples, chrs)
end

function Base.read(io::GZipStream, ::Type{Beagle}; samples=nothing, chrs=nothing)
    reader = BeagleReader(io; samples=samples, chrs=chrs)
    read(reader, Beagle)
end

Base.read(s::AbstractString, ::Type{Beagle}; kwargs...) = 
    read(gzopen(s), Beagle; kwargs...)

end
