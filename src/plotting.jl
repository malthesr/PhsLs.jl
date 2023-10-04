module Plotting

using StatsPlots

export plotadmixture

function plotadmixture(Q;
                       pops=nothing,
                       colorscheme=nothing,
                       labels=nothing,
                       xticks=nothing,
                       yticks=([0, 0.5, 1], ["0", "0.5", "1"]),
                       legend=:outerright,
                       kwargs...)
    I, K = size(Q)
    if isnothing(colorscheme)
        colorscheme = Symbol("Set1_$(K)")
    end
    colors = reshape(palette(colorscheme, K)[1:K], 1, K)
    if isnothing(labels)
        labels=reshape(["K = $(k)" for k in 1:K], 1, K)
    end
    groupedbar(
        Q,
        barpositions=:stack,
        barwidths=1,
        color=colors,
        linecolor=colors,
        labels=labels,
        xticks=xticks,
        ylabel="Admixture proportion",
        yticks=yticks,
        xlims=(0.5, I + 0.5),
        legend=legend,
        kwargs...
    )
    if !isnothing(pops)
        popend = cumsum(last.(pops))
        popstart = [1; popend[1:end-1] .+ 1]
        linepos = popend[1:end - 1] .- 0.5
        labelpos = (popstart .+ popend) ./ 2 .- 0.5
        labels = first.(pops)
        plot!(linepos, seriestype=:vline, lw=1, color=:black, label=nothing)
        plot!(xlabel = "Population", xticks=(labelpos, labels))
    end
end

end
