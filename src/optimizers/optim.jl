import Optim 
export OptimOptimizer

@with_kw mutable struct OptimOptimizer <: AbstractOptimizer 
    Optimizer
    options::Optim.Options
    function OptimOptimizer(opt)
        new(opt, Optim.Options())
    end
    function OptimOptimizer(opt, options)
        new(opt, options)
    end
end

function setOptions!(opt::OptimOptimizer; 
        max_iter::Int64, callback::Union{Missing, Function}, kwargs...)
    @warn "max_iter is not used. You need to specify `iteration=max_iter` when you construct OptimOptimizer"
    if !ismissing(callback)
        @warn "callback is not used"
    end
    keyword_not_used(kwargs)
end

function (opt::OptimOptimizer)(f::Function, g!::Function, x0::Array{Float64,1})
    Optim.optimize(f, g!, x0, opt.Optimizer, opt.options)
end
