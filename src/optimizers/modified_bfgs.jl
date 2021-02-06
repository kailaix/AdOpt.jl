export ModifiedBFGSOptimizer

@with_kw mutable struct ModifiedBFGSOptimizer <: AbstractOptimizer
    max_iter::Int64 = 15000
    callback::Union{Missing, Function} = missing
    eta::Float64 = 0.001
    beta::Tuple{Float64, Float64} =  (0.001, 0.001)
    g_tol::Float64 = 1e-15
    t::Float64 = 1
end

function setOptions!(opt::ModifiedBFGSOptimizer; 
    max_iter::Int64, callback::Union{Missing, Function}, kwargs...)
    opt.max_iter = max_iter
    opt.callback = callback
    ks = Set(keys(kwargs))
    if :g_tol in ks 
        opt.g_tol = kwargs[:g_tol]
    end
    keyword_not_used(kwargs)
end

function (opt::ModifiedBFGSOptimizer)(f::Function, g!::Function, x0::Array{Float64,1})
    η, β = opt.eta, opt.beta
    βp = β
    mt = zeros(length(x0))
    vt = zeros(length(x0))
    vt_ = zeros(length(x0))
    ϵ = 1e-8

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
    φ0 = f__
    dφ0 = dφ(0.0)
    res = BackTracking()(φ, dφ, φdφ, α0, φ0,dφ0)
    α = res[1]

    xnew = x + α * d 
    @. G_ = G
    g!(G, xnew)
    fnew = f(xnew)
    x, x_ = xnew, x 
    f__, f_ = fnew, f__ 
    push!(losses, f__)
    
    Δ_ = G_
    
    # from second step: BFGS
    for i = 1:max_iter-1
        
        if ADCME.options.training.verbose
            println("=============== STEP $i ===============\nStepsize = $α")
        end
        if !ismissing(opt.callback)
            opt.callback(x, i, f__)
        end

        mt = β[1] * mt + (1 - β[1]) * G
        vt = β[2] * vt + (1 - β[2]) * G.^2
        # Δ =  @. mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) 
        Δ =  mt 
        βp = βp .* β
        
        # s = x - x_
        # y = Δ - Δ_ 

        s = x - x_
        V = @. √(vt / (1 - βp[2]))
        # y = G - G_ + opt.t * diagm(0=>V)*s
        y = G - G_ 


        B = (I - s*y'/(y'*s)) * B * (I - y*s'/(y'*s)) + s*s'/(y'*s)
        # d = - B*G
        d = - B * G 
        
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
        
        if abs(res[1])<1e-15
            @warn "Step size too small"
            return losses 
        end


        xnew = x + α * d
        @. G_ = G
        g!(G, xnew)
        fnew = f(xnew)
        x, x_ = xnew, x 
        f__, f_ = fnew, f__
        Δ_ = Δ
        
        # check for convergence 
        if norm(G)<opt.g_tol || isnan(f__)
            break
        end
        push!(losses, f__)
    end

    return losses
end
