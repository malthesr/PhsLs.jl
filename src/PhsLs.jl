module PhsLs

export Phase, Admixture

using Reexport

include("utils.jl")
include("types.jl")
include("input.jl")
include("emcore.jl")

@reexport using .Types
@reexport using .Input
@reexport using .EmCore

module Phase

using Reexport

using ..Utils
using ..Types
using ..EmCore

include("phase/parameters.jl")
include("phase/emission.jl")
include("phase/transition.jl")
include("phase/forwardbackward.jl")
include("phase/posterior.jl")
include("phase/expectation.jl")
include("phase/em.jl")
include("phase/misc.jl")

@reexport using .Parameters
@reexport using .Emission
@reexport using .Transition
@reexport using .ForwardBackward
@reexport using .Posterior
@reexport using .Expectation
@reexport using .Em
@reexport using .Misc

end

module Admixture

using Reexport

using ..Utils
using ..Types
using ..EmCore

include("admixture/parameters.jl")
include("admixture/posterior.jl")
include("admixture/misc.jl")
include("admixture/expectation.jl")
include("admixture/em.jl")

@reexport using .Parameters
@reexport using .Posterior
@reexport using .Expectation
@reexport using .Em
@reexport using .Misc

end

end
