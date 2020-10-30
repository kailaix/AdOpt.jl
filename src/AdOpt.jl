module AdOpt

    using SparseArrays
    using LinearAlgebra
    using PyCall
    using PyPlot
    using Reexport
    using Statistics
    using LineSearches
    using MAT
    using Parameters
    @reexport using ADCME

    function __init__()
    end

    include("utils.jl")
    include("optimizers/optim.jl")
    include("optimizers/bfgs.jl")
    include("optimizers/lbfgs.jl")
    include("optimizers/ncg.jl")
    include("optimizers/ncgbfgs.jl")
    include("optimizers/adam.jl")
    include("optimizers/adaptive_bfgs.jl")
    include("api.jl")

end 
