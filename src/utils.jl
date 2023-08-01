module Utils

export norm!, cnorm!, sumdrop

function norm!(x; dims=nothing)
    if isnothing(dims)
        x[:] /= sum(x)
    else
        foreach(norm!, eachslice(x, dims=dims))
    end
end

function cnorm!(x)
    s = sum(x)
    x[:] /= sum(x)
    s
end

sumdrop(x; dims) = dropdims(sum(x, dims=dims), dims=dims)

end
