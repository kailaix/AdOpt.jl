ulimit -u 10000

# julia forward.jl 
for seed in 2 23 233 2333 23333
do 
    # julia bfgs.jl $seed &
    # julia bfgs_adam.jl $seed &
    # julia lbfgs.jl $seed &
    # julia adam.jl $seed &
    julia lbfgs_adam.jl $seed &
    wait 
    julia plot.jl $seed
done 

