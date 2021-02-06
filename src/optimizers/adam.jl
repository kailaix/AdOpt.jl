export ADAMOptimizer

@with_kw mutable struct ADAMOptimizer <: AbstractOptimizer
    max_iter::Int64 = 15000
    callback::Union{Missing, Function} = missing
    eta::Float64 = 0.001
    beta::Tuple{Float64, Float64} =  (0.9, 0.999)
    angles::Array{Float64} = []
end

function setOptions!(opt::ADAMOptimizer; 
    max_iter::Int64, callback::Union{Missing, Function}, kwargs...)
    opt.max_iter = max_iter
    opt.callback = callback
    keyword_not_used(kwargs)
end

function (opt::ADAMOptimizer)(f::Function, g!::Function, x0::Array{Float64,1})
    η, β = opt.eta, opt.beta
    x = x0 
    mt = zeros(length(x))
    vt = zeros(length(x))
    G = zeros(length(x))
    ϵ = 1e-8
    βp = β
    losses = [f(x)]
    for i = 1:opt.max_iter
        if ADCME.options.training.verbose
            println("=============== STEP $i ===============")
        end
        g!(G, x)
        mt = β[1] * mt + (1 - β[1]) * G
        vt = β[2] * vt + (1 - β[2]) * G.^2
        Δ =  @. mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) 
        βp = βp .* β

        x -=  η * Δ
        push!(opt.angles, G'*Δ/norm(G)/norm(Δ))
        push!(losses, f(x))
    end
    
    return losses
end
