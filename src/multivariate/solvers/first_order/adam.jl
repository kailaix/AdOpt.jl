struct Adam{IL, L, T, Tprep<:Union{Function, Nothing}} <: FirstOrderOptimizer
    alphaguess!::IL
    linesearch!::L
    P::T
    precondprep!::Tprep
    manifold::Manifold
    eta::Float64
    beta::Tuple{Float64,Float64}
end

Base.summary(::Adam) = "Adam Optimizer"

function Adam(η = 0.001; alphaguess = LineSearches.InitialStatic(), # TODO: Investigate good defaults.
                           linesearch = LineSearches.Static(),      # TODO: Investigate good defaults
                           P = nothing,
                           precondprep = (P, x) -> nothing,
                           manifold::Manifold=Flat())
    Adam(_alphaguess(alphaguess), linesearch, P, precondprep, manifold,
            η, (0.9, 0.8))
end

mutable struct AdamState{Tx, T} <: AbstractOptimizerState
    x
    x_previous
    f_x_previous
    s
    mt 
    vt 
    βp
    @add_linesearch_fields()
end

function initial_state(method::Adam, options, d, initial_x::AbstractArray{T}) where T
    initial_x = copy(initial_x)

    AdamState(initial_x, # Maintain current state in state.x
            copy(initial_x),
            0.0,
            similar(initial_x),
            zero(initial_x),
            zero(initial_x),
            method.beta,
            @initial_linesearch()...)
end

function update_state!(d, state::AdamState, method::Adam)
    value_gradient!(d, state.x)
    Δ = gradient(d)
    η, β = method.eta, method.beta 
    mt, vt, βp = state.mt, state.vt, state.βp 
    @. mt = β[1] * mt + (1-β[1])*Δ
    @. vt = β[2] * vt + (1-β[2]) * Δ^2 
    @. Δ = mt/(1-βp[1])/(sqrt(vt/(1-βp[2]))+1e-8)*η
    state.s = Δ
    state.x -= η*state.alpha * state.s
    state.mt, state.vt, state.βp = mt, vt, βp .* β
    false
end

function trace!(tr, d, state, iteration, method::Adam, options, curr_time=time())
  common_trace!(tr, d, state, iteration, method, options, curr_time)
end
