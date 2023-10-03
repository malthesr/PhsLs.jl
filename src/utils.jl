module Utils

export Option, norm!, cnorm!, sumdrop, colsum, rowsum

const Option{T} = Union{T, Nothing}

function norm!(x; dims=nothing)
    if isnothing(dims)
        x[:] /= sum(x);
    else
        foreach(norm!, eachslice(x, dims=dims));
    end
end

function cnorm!(x)
    s = sum(x)
    x[:] /= sum(x)
    s
end

@inline sumdrop(x; dims) = dropdims(sum(x, dims=dims), dims=dims)
@inline colsum(x) = sumdrop(x, dims=1)
@inline rowsum(x) = sumdrop(x, dims=2)

end
