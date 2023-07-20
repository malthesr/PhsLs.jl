module PhsLs

export Phase

using Reexport

include("utils.jl")
include("types.jl")
include("input.jl")

@reexport using .Types
@reexport using .Input

module Phase

using Reexport

using ..Utils
using ..Types

include("phase/parameters.jl")
include("phase/emission.jl")
include("phase/forwardbackward.jl")
include("phase/posterior.jl")
include("phase/expectation.jl")
include("phase/em.jl")
include("phase/misc.jl")

@reexport using .Parameters
@reexport using .Emission
@reexport using .ForwardBackward
@reexport using .Posterior
@reexport using .Expectation
@reexport using .Em
@reexport using .Misc

end

end
