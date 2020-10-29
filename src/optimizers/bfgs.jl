export BFGSOptimizer

@with_kw mutable struct BFGSOptimizer <: AbstractOptimizer
    max_iter::Int64 = 15000
    callback::Union{Missing, Function} = missing
    g_tol::Float64 = 1e-15
end

function setOptions!(opt::BFGSOptimizer; 
    max_iter::Int64, callback::Union{Missing, Function}, kwargs...)
    opt.max_iter = max_iter
    opt.callback = callback
    ks = Set(keys(kwargs))
    if :g_tol in ks 
        opt.g_tol = kwargs[:g_tol]
    end
    keyword_not_used(kwargs)
end

function (opt::BFGSOptimizer)(f::Function, g!::Function, x0::Array{Float64,1})
    x = x0 
    α0 = 1.0

    losses = Float64[]

    max_iter = opt.max_iter
    
    # preallocate memories 
    n = length(x)
    G_ = zeros(n)
    G = zeros(n)
    x_ = zeros(n)
    f_ = 0.0
    f__ = 0.0

    B = I

    # first step: gradient descent 
    g!(G, x)
    f__ = f(x)
    
    # the first step is a backtracking linesearch 
    d = -G 
    φ = α->f(x + α*d)
    dφ = α->begin 
        g = zeros(n)
        g!(g, x + α*d)
        g'*d
    end
    φdφ(x) = φ(x), dφ(x)
    φ0 = φ(0.0)
    dφ0 = dφ(0.0)
    res = BackTracking()(φ, dφ, φdφ, α0, φ0,dφ0)
    α = res[1]


    xnew = x + α * d 
    @. G_ = G
    g!(G, xnew)
    fnew = f(xnew)
    x, x_ = xnew, x 
    f__, f_ = fnew, f__ 
    d_ = d
    push!(losses, f__)

    
    # from second step: BFGS
    for i = 1:max_iter-1
        
        if ADCME.options.training.verbose
            println("=============== STEP $i ===============\nStepsize = $α")
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
        φ0 = φ(0.0)
        dφ0 = dφ(0.0)
        
        α0 = min(10.0, 10α)
        res = BackTracking()(φ, dφ, φdφ, α0, φ0,dφ0)
        α = res[1]
        

        if abs(α)<1e-15
            @warn "Step size too small"
            return losses 
        end


        xnew = x + α * d
        @. G_ = G
        g!(G, xnew)
        fnew = f(xnew)
        x, x_ = xnew, x 
        f__, f_ = fnew, f__
        d_ = d
        
        # check for convergence 
        if norm(G)<opt.g_tol || isnan(f__)
            break
        end
        push!(losses, f__)
    end

    return losses
end
