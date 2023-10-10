module Plotting

using StatsPlots

export plotadmixture, plotclusterfreqs

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
        labels = reshape(["K = $(k)" for k in 1:K], 1, K)
    end
    p = groupedbar(
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
        plot!(p, linepos, seriestype=:vline, lw=1, color=:black, label=nothing)
        plot!(p, xlabel = "Population", xticks=(labelpos, labels))
    end
    p
end

function plotclusterfreqs(cf;
                          colorscheme=nothing,
                          labels=nothing,
                          xticks=nothing,
                          yticks=([0, 0.5, 1], ["0", "0.5", "1"]),
                          legend=:outerright,
                          kwargs...)
    S, C = size(cf)
    if isnothing(colorscheme)
        colorscheme = Symbol("Set2_$(C)")
    end
    colors = reshape(palette(colorscheme, C)[1:C], 1, C)
    if isnothing(labels)
        labels = reshape(["C = $(c)" for c in 1:C], 1, C)
    end
    groupedbar(
        cf,
        barpositions=:stack,
        barwidths=1,
        color=colors,
        linecolor=colors,
        labels=labels,
        ylabel="Cluster frequency",
        yticks=yticks,
        xlabel="Site",
        xlims=(0.5, S + 0.5),
        legend=legend,
        kwargs...
    )
end

end

