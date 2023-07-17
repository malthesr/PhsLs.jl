module Utils

using Transducers: foldxt, Map

export norm, norm!, cnorm, allsame, outer, symouter, 
    colsum, rowsum, sumdrop, parmapreduce

allsame(x) = all(y -> y == first(x), x)

norm(x; kwargs...) = x ./ sum(x; kwargs...)
function norm!(x; kwargs...)
    x[:] /= sum(x; kwargs...)
end
function cnorm(x)
    c = sum(x)
    (c, x ./ c)
end

outer(f, x, y) = broadcast(f, x, transpose(y))
outer(x) = outer(*, x, x)
outer(x, y) = outer(*, x, y)
symouter(f, x, y) = outer(f, x, y) .+ transpose(outer(f, x, y))
symouter(x, y) = symouter(*, x, y)

sumdrop(x; dims) = dropdims(sum(x, dims=dims), dims=dims)
colsum(x::AbstractMatrix{Float64})::Vector{Float64} = sumdrop(x, dims=1)
rowsum(x::AbstractMatrix{Float64})::Vector{Float64} = sumdrop(x, dims=2)

parmapreduce(f, op, it) = foldxt(op, it |> Map(f))

end
