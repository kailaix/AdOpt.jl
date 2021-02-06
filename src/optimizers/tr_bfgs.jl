export HybridAdamBFGSOptimizer

@with_kw mutable struct HybridAdamBFGSOptimizer <: AbstractOptimizer
    max_iter::Int64 = 15000
    callback::Union{Missing, Function} = missing
    eta::Float64 = 0.001
    beta::Tuple{Float64, Float64} =  (0.9, 0.999)
    g_tol::Float64 = 1e-15
    warm_start::Int64 = 0
    x::Union{Missing, Array{Float64}} = missing # optimal solution 
    is_bfgs::Array{Bool,1} = []
    angles::Array{Float64,1} = []
    decay_rate::Float64 = 0.05
end

function setOptions!(opt::HybridAdamBFGSOptimizer; 
    max_iter::Int64, callback::Union{Missing, Function}, kwargs...)
    opt.max_iter = max_iter
    opt.callback = callback
    ks = Set(keys(kwargs))
    if :g_tol in ks 
        opt.g_tol = kwargs[:g_tol]
    end
    keyword_not_used(kwargs)
end

function (opt::HybridAdamBFGSOptimizer)(f::Function, g!::Function, x0::Array{Float64,1})

    x = x0 

    losses = Float64[]
    max_iter = opt.max_iter
    η, β = opt.eta, opt.beta
    ϵ = 1e-8
    βp = β
    
    # preallocate memories 
    n = length(x)
    G_ = zeros(n)
    G = zeros(n)
    x_ = zeros(n)
    f__ = f(x)
    mt = zeros(length(x))
    vt = zeros(length(x))
    B = I
    push!(losses, f__)
    push!(opt.angles, G'*G/norm(G)/norm(G))
    opt.x = x
    fmin = f__

    is_bfgs = false
    ####### the first step is always ADAM
    g!(G, x)
    mt = β[1] * mt + (1 - β[1]) * G
    vt = β[2] * vt + (1 - β[2]) * G.^2
    Δ =  @. mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) 
    βp = βp .* β
    x -=  η * Δ
    
    G_ = copy(G)
    x_ = x0
    push!(losses, f(x))
    push!(opt.angles, G'*Δ/norm(Δ)/norm(G))
    if losses[end] < fmin
        opt.x = x 
    end
    
    

    α = 0.0
    f_bfgs = NaN
    for i = 1:max_iter-1
        η = opt.eta * exp(-opt.decay_rate*i)
        push!(opt.is_bfgs, is_bfgs)

        if ADCME.options.training.verbose
            IS_BFGS = is_bfgs ? "BFGS" : "ADAM"
            println("=========== STEP $i ($IS_BFGS) ===========")
        end

        g!(G, x)

        ####### Check convergence 
        if norm(G)<opt.g_tol || isnan(f__)
            break
        end

        ####### first try BFGS 
        s = x - x_ 
        y = G - G_ 
        B = (I - s*y'/(y'*s)) * B * (I - y*s'/(y'*s)) + s*s'/(y'*s)
        d = -B*G 
        # line search 
        dφ0 = d'*G
        φ = α->f(x + α*d)
        φ0 = f__
        

        α0 = 10.0
        if α>0
            α0 = min(10.0, 10α)
        end
        
        try 
            if dφ0>0 || i < opt.warm_start
                throw(Exception())
            end
            
            res = BackTracking(order=3)(φ, α0, φ0,dφ0)
            α, f_bfgs = res
        catch
            B = I
            α, f_bfgs = 0.0, f__
        end

        x_bfgs = x + α * d
        
        ####### then try ADAM 
        mt = β[1] * mt + (1 - β[1]) * G
        vt = β[2] * vt + (1 - β[2]) * G.^2
        Δ =  @. mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) 
        βp = βp .* β


        x_adam = x - η * Δ
        f_adam = f(x_adam)

        x_ = x 
        G_ = copy(G)
        if α>0 && f_bfgs<f_adam
            x = x_bfgs
            f__ = f_bfgs
            push!(opt.angles, -G'*d/norm(d)/norm(G))
            is_bfgs = true 
        else 
            x = x_adam
            f__ = f_adam
            push!(opt.angles, -G'*Δ/norm(Δ)/norm(G))
            is_bfgs = false
        end 
        
        push!(losses, f__)
        if losses[end] < fmin
            opt.x = x 
        end

    end

    return losses
end
