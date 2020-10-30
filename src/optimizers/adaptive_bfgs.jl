export AdaptiveBFGSOptimizer

@with_kw mutable struct AdaptiveBFGSOptimizer <: AbstractOptimizer
    max_iter::Int64 = 15000
    callback::Union{Missing, Function} = missing
    g_tol::Float64 = 1e-15
    threshold::Float64 = 1e-3
    patience::Int64 = 5
    breakpoint::Int64 = -1
    min_adam::Int64 = 100
    eta::Float64 = 0.001
    beta::Tuple{Float64, Float64} =  (0.9, 0.999)
    angles::Array{Float64} = []
end

function setOptions!(opt::AdaptiveBFGSOptimizer; 
    max_iter::Int64, callback::Union{Missing, Function}, kwargs...)
    opt.max_iter = max_iter
    opt.callback = callback
    ks = Set(keys(kwargs))
    if :g_tol in ks 
        opt.g_tol = kwargs[:g_tol]
    end
    if :patience in ks 
        opt.patience = kwargs[:g_tol]
    end
    if :threshold in ks 
        opt.threshold = kwargs[:g_tol]
    end
    keyword_not_used(kwargs)
end

function (opt::AdaptiveBFGSOptimizer)(f::Function, g!::Function, x0::Array{Float64,1})
    η, β = opt.eta, opt.beta
    x = x0 
    # preallocate memories 
    n = length(x)
    G_ = zeros(n)
    G = zeros(n)
    x_ = zeros(n)
    mt = zeros(length(x))
    vt = zeros(length(x))
    ϵ = 1e-8
    βp = β
    losses = Float64[]

    __iter = 0
    stall = 0
    ###### Phase I: Adam Optimizer ######
    for i = 1:opt.max_iter
        G_ = copy(G)
        x_ = copy(x)

        __iter += 1
        if ADCME.options.training.verbose
            if stall>0
                println("=============== STEP $i (ADAM, Stall = $stall/$(opt.patience)) ===============")
            else
                println("=============== STEP $i (ADAM) ===============")
            end
        end
        g!(G, x)
        mt = β[1] * mt + (1 - β[1]) * G
        vt = β[2] * vt + (1 - β[2]) * G.^2
        Δ =  @. mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) 
        βp = βp .* β

        x -=  η * Δ
        push!(opt.angles, G'*Δ/norm(G)/norm(Δ))
        push!(losses, f(x))
        

        if __iter<=opt.min_adam
            continue 
        end

        if (losses[end-1]-losses[end])/losses[end-1]<opt.threshold
            stall+=1
        else
            stall=0
        end
        if stall==opt.patience
            opt.breakpoint = __iter
            break
        end
    end

    ###### Phase II: BFGS Optimizer ######
    B = I
    
    ###### the first step is a backtracking linesearch 
    x_ = copy(x)
    d = -G 
    φ = α->f(x + α*d)
    dφ = α->begin 
        g = zeros(n)
        g!(g, x + α*d)
        g'*d
    end
    φdφ(x) = φ(x), dφ(x)
    φ0 = losses[end]
    dφ0 = dφ(0.0)
    res = BackTracking()(φ, dφ, φdφ, 1.0, φ0,dφ0)
    α = res[1]

    push!(opt.angles, -G'*d/norm(G)/norm(d))
    x = x + α * d 
    f__ = f(x)
    G_ = copy(G)
    g!(G, x)
    push!(losses, f__)

    iterations = opt.max_iter-__iter-1
    for i = 1:iterations
        __iter+=1
        
        if ADCME.options.training.verbose
            println("=============== STEP $i/TOTAL $__iter (BFGS) ===============\nStepsize = $α")
        end
        if !ismissing(opt.callback)
            opt.callback(x, i, f__)
        end

        s = x - x_ 
        y = G - G_ 
        B = (I - s*y'/(y'*s)) * B * (I - y*s'/(y'*s)) + s*s'/(y'*s)
        d = -B*G 
        
        # line search 
        φ = α->f(x + α*d)
        dφ = α->begin 
            g = zeros(n)
            g!(g, x + α*d)
            g'*d
        end
        φdφ(x) = φ(x), dφ(x)
        φ0 = f__
        dφ0 = dφ(0.0)
        
        α0 = min(10.0, 10α)
        res = BackTracking()(φ, dφ, φdφ, α0, φ0,dφ0)
        α = res[1]
        

        if abs(α)<1e-15
            @warn "Step size too small"
            return losses 
        end

        push!(opt.angles, -G'*d/norm(G)/norm(d))
        xnew = x + α * d
        @. G_ = G
        g!(G, xnew)
        fnew = f(xnew)
        x, x_ = xnew, x 
        f__, f_ = fnew, f__
        
        # check for convergence 
        if norm(G)<opt.g_tol || isnan(f__)
            break
        end
        push!(losses, f__)
    end

    return losses
end
