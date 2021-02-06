function mpi_optimize(_f::Function, _g!::Function, x0::Array{Float64,1}, method, options::Options = Options(); step_callback = missing)
    flag = zeros(Int64, 1)

    function f(x)
        flag[1] = 1
        mpi_sync!(flag)
        return _f(x)
    end

    function g!(G, x)
        flag[1] = 2
        mpi_sync!(flag)
        _g!(G, x)
    end

    r = mpi_rank()
    if r==0
        result = optimize(f, g!, x0, method, options; step_callback = step_callback)
        flag[1] = 0
        mpi_sync!(flag)
        return result
    else 
        while true 
            mpi_sync!(flag)
            if flag[1]==1
                _f(x0)
            elseif flag[1]==2
                _g!(zero(x0), x0)
            else 
                break 
            end
        end
        return nothing
    end
end


function mpi_optimize(sess::PyObject, f::PyObject, g::PyObject, x0::Array{Float64,1}, method, options::Options = Options())
    _f = x->run(sess, f)
    _g = (G, x)->begin 
        G[:] = run(sess, g)
    end
    mpi_optimize(_f, _g, x0, method, options)
end