module PhsLs

export Phase, Admixture, Joint

using Reexport

include("utils.jl")
include("types.jl")
include("input.jl")
include("emcore.jl")
include("plotting.jl")

@reexport using .Types
@reexport using .Input
@reexport using .Plotting

module Phase

using Reexport

using ..Utils
using ..Types
using ..Input
using ..EmCore

include("phase/parameters.jl")
include("phase/emission.jl")
include("phase/forwardbackward.jl")
include("phase/posterior.jl")
include("phase/em.jl")
include("phase/misc.jl")

@reexport using .Parameters
@reexport using .Emission
@reexport using .ForwardBackward
@reexport using .Posterior
@reexport using .Em
@reexport using .Misc

end

module Admixture

using Reexport

using ..Utils
using ..Types
using ..Input
using ..EmCore

include("admixture/parameters.jl")
include("admixture/posterior.jl")
include("admixture/em.jl")

@reexport using .Parameters
@reexport using .Posterior
@reexport using .Em

end

module Joint

using Reexport

using ..Utils
using ..Types
using ..Input
using ..EmCore

include("joint/parameters.jl")
include("joint/emission.jl")
include("joint/forwardbackward.jl")
include("joint/posterior.jl")
include("joint/em.jl")

@reexport using .Parameters
@reexport using .Emission
@reexport using .ForwardBackward
@reexport using .Posterior
@reexport using .Em

end

module Joint2

using Reexport

using ..Utils
using ..Types
using ..Input
using ..EmCore

include("joint2/parameters.jl")
include("joint2/emission.jl")
include("joint2/forwardbackward.jl")
# include("joint/posterior.jl")
# include("joint/em.jl")

@reexport using .Parameters
@reexport using .Emission
@reexport using .ForwardBackward
# @reexport using .Posterior
# @reexport using .Em

end

end
