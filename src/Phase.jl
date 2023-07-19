module Phase

using Reexport

include("utils.jl")
include("types.jl")
include("input.jl")
include("parameters.jl")
include("emission.jl")
include("forwardbackward.jl")
include("posterior.jl")
include("expectation.jl")
include("em.jl")
include("misc.jl")

@reexport using .Types
@reexport using .Input
@reexport using .Parameters
@reexport using .Emission
@reexport using .ForwardBackward
@reexport using .Posterior
@reexport using .Expectation
@reexport using .Em
@reexport using .Misc

end
