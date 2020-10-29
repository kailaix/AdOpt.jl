export LBFGSOptimizer

@with_kw mutable struct LBFGSOptimizer <: AbstractOptimizer
    max_iter::Int64 = 15000
    callback::Union{Missing, Function} = missing
    g_tol::Float64 = 1e-15
    m::Int64 = 50
end

function setOptions!(opt::LBFGSOptimizer; 
    max_iter::Int64, callback::Union{Missing, Function}, kwargs...)
    opt.max_iter = max_iter
    opt.callback = callback
    ks = Set(keys(kwargs))
    if :g_tol in ks 
        opt.g_tol = kwargs[:g_tol]
    end
    if :m in ks 
        opt.m = kwargs[:m]
    end
    keyword_not_used(kwargs)
end
function (opt::LBFGSOptimizer)(f::Function, g!::Function, x0::Array{Float64, 1})
    x = x0
    α0 = 1.0

    losses = Float64[]

    max_iter = opt.max_iter
    m = opt.m 
    
    # preallocate memories 
    Ss = Vector{Float64}[]
    Ys = Vector{Float64}[]
    αs = zeros(m)

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
    f__ = f(xnew)
    x, x_ = xnew, x 
    push!(losses, f__)
    pushfirst!(Ss, x - x_ )
    pushfirst!(Ys, G - G_ )

    # from second step: BFGS
    for i = 1:max_iter-1
        
        if ADCME.options.training.verbose
            println("=============== STEP $i ===============\nStepsize = $α")
        end
        if !ismissing(opt.callback)
            opt.callback(x, i, f__)
        end

        d = -G 
        for j = 1:length(Ss)
            αs[j] = Ss[j]'*d/(Ys[j]'*Ss[j])
            d -= αs[j]*Ys[j]
        end
        d = Ys[1]'*Ss[1]/(Ys[1]'*Ys[1]) * d
        for j = length(Ss):-1:1
            β = Ys[j]'*d / (Ys[j]'*Ss[j])
            d += (αs[j] - β) * Ss[j]
        end
        
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
        α0 = min(1.0, 10α)
        res = BackTracking()(φ, dφ, φdφ, α0, φ0,dφ0)
        α = res[1]

        if abs(α)<1e-15
            @warn "Step size too small"
            return losses 
        end

        xnew = x + α * d
        @. G_ = G
        g!(G, xnew)
        f__ = f(xnew)
        x, x_ = xnew, x 
        
        # check for convergence 
        if norm(G)<opt.g_tol || isnan(f__)
            break
        end
        push!(losses, f__)
        pushfirst!(Ss, x - x_ )
        pushfirst!(Ys, G - G_ )
        if length(Ss)>m
            pop!(Ss)
            pop!(Ys)
        end
    end

    return losses
end