module Utils

using ThreadsX

export norm!, cnorm!, sumdrop, parmapreduce, outer, symouter, cnorm,
    colsum, rowsum, colsum!, rowsum!

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

@inline parmapreduce(f, op, it) = ThreadsX.mapreduce(f, op, it)

end
