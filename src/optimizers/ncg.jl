export NCGOptimizer

@with_kw mutable struct NCGOptimizer <: AbstractOptimizer
    max_iter::Int64 = 15000
    callback::Union{Missing, Function} = missing
    g_tol::Float64 = 1e-15
    method::String = "N"
end

function setOptions!(opt::NCGOptimizer; 
    max_iter::Int64, callback::Union{Missing, Function},  kwargs...)
    opt.max_iter = max_iter
    opt.callback = callback
    ks = Set(keys(kwargs))
    if :g_tol in ks 
        opt.g_tol = kwargs[:g_tol]
    end
    keyword_not_used(kwargs)
end

function (opt::NCGOptimizer)(f::Function, g!::Function, x0::Array{Float64,1})
    max_iter = opt.max_iter
    x = x0 
    G = zeros(length(x))
    G_ = zeros(length(x))
    g!(G, x)
    p = -G 
    losses = Float64[]
    α = 1.0
    for i = 1:max_iter
        ################# Store historic results 
        G_ = copy(G)
        f__ = f(x)
        push!(losses, f(x))

        ################# Print information
        if ADCME.options.training.verbose
            println("=============== STEP $i ===============\nStepsize = $α")
        end
        if !ismissing(opt.callback)
            opt.callback(x, i, f__)
        end

        ################# line search 
        φ = α->f(x + α*p)
        dφ = α->begin 
            g = zeros(length(x0))
            g!(g, x + α*p)
            g'*p
        end
        φdφ(x) = φ(x), dφ(x)
        φ0 = f__
        dφ0 = dφ(0.0)
        α0 = min(10.0, 10α)
        res = BackTracking()(φ, dφ, φdφ, α0, φ0,dφ0)
        α = res[1]
        x = x + α*p 
        

        if abs(α)<1e-15
            @warn "Step size too small: $α"
            return losses 
        end

        ################# computes beta 
        g!(G, x)
        y = G - G_ 
        if opt.method == "HS"            
            β = G'*y/(p'*y)
        elseif opt.method=="FR"
            β = norm(G)^2/norm(G_)^2
        elseif opt.method=="PRP"
            β = G'*y/norm(G_)^2
        elseif opt.method=="CD"
            β = norm(G)^2/(-p'*G_)
        elseif opt.method=="LS"
            β = G'*y/(-p'*G_)
        elseif opt.method=="DY"
            β = norm(G)^2/(p'*y)
        elseif opt.method=="N"
            β = (y - 2*p*norm(y)^2/(p'*y))'*G/(p'*y)
        else 
            error("Method $(opt.method) not valid")
        end

        if norm(G)<opt.g_tol || isnan(β)
            break
        end

        ################# update direction
        p = -G+β*p 
    end
    return losses
end
