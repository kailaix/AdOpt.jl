function mpi_optimize(_f::Function, _g!::Function, x0::Array{Float64,1}, method, options::Options = Options())
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
        result = optimize(f, g!, x0, method, options)
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