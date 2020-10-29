import ADCME:Optimize!, AbstractOptimizer
export Optimize!, AbstractOptimizer

"""
    Optimize!(
        sess::PyObject, loss::PyObject, optimizer::AbstractOptimizer, max_iter::Int64 = 15000;
        vars::Union{Array{PyObject},PyObject, Missing} = missing, 
        grads::Union{Array{T},Nothing,PyObject, Missing} = missing, 
        step_callback::Union{Function, Missing} = missing,
        loss_callback::Union{Function, Missing} = missing; 
        kwargs...) where 
        {T<:Union{Nothing, PyObject}, S<:AbstractOptimizer}

Optimizing the loss function using a user-defined optimizer `optimizer`. 

- `sess`: A TensorFlow session 
- `loss`: a scalar loss tensor 
- `max_iter`: maximum number of iterations
- `vars`, `grads`: optimizable variables and gradients. If `vars` or `grads` are not provided, it is inferred from the current computational graph. 
- `optimizer`: An instance of AbstractOptimizer. The optimizer must be callable. 
- `callback`: callback after each linesearch completion (NOT one step in the linesearch). It accepts three arguments, `loss_callback(x, iter, loss)`.
- `loss_callback`: callback after each loss function evaluation
- `kwargs`: keyword arguments that are passed to `optimizer`

```julia
optimizer(f, g!, x0; callback = callback, kwargs...)
```

"""
function Optimize!(
    sess::PyObject, loss::PyObject, optimizer::Union{S, Missing}, max_iter::Int64 = 15000;
    vars::Union{Array{PyObject},PyObject, Missing} = missing, 
    grads::Union{Array{T},Nothing,PyObject, Missing} = missing, 
    callback::Union{Function, Missing} = missing,
    loss_callback::Union{Function, Missing} = missing, kwargs...) where 
    {T<:Union{Nothing, PyObject}, S<:AbstractOptimizer}
    vars = coalesce(vars, get_collection())
    grads = coalesce(grads, gradients(loss, vars))
    if isa(vars, PyObject); vars = [vars]; end
    if isa(grads, PyObject); grads = [grads]; end
    if length(grads)!=length(vars); error("AdOpt: length of grads and vars do not match"); end

    if !all(is_variable.(vars))
        error("AdOpt: the input `vars` should be trainable variables")
    end

    idx = ones(Bool, length(grads))
    pynothing = pytypeof(PyObject(nothing))
    for i = 1:length(grads)
        if isnothing(grads[i]) || pytypeof(grads[i])==pynothing
            idx[i] = false
        end
    end
    grads = grads[idx]
    vars = vars[idx]
    sizes = []
    for v in vars
        push!(sizes, size(v))
    end
    grds = vcat([tf.reshape(g, (-1,)) for g in grads]...)
    vs = vcat([tf.reshape(v, (-1,)) for v in vars]...); x0 = run(sess, vs)
    pl = placeholder(x0)
    n = 0
    assign_ops = PyObject[]
    for (k,v) in enumerate(vars)
        push!(assign_ops, assign(v, tf.reshape(pl[n+1:n+prod(sizes[k])], sizes[k])))
        n += prod(sizes[k])
    end
    
    __loss = 0.0
    __losses = Float64[]
    __iter = 0
    __value = nothing
    __ls_iter = 0
    function f(x)
        run(sess, assign_ops, pl=>x)
        __ls_iter += 1
        __loss = run(sess, loss)
        if !ismissing(loss_callback)
            loss_callback(x, __ls_iter, __loss)
        end
        ADCME.options.training.verbose && (println("iter $__ls_iter, current loss = $__loss"))
        return __loss
    end

    function g!(G, x)
        run(sess, assign_ops, pl=>x)
        __value = x
        G[:] = run(sess, grds)
    end

    @info "Optimization starts..."
    setOptions!(optimizer; callback = callback, max_iter = max_iter, kwargs...)
    __losses = optimizer(f, g!, x0)
    return __losses
end