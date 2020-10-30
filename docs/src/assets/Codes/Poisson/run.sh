ulimit -u 10000

# julia forward.jl 
for seed in 2 23 233 2333 23333
do 
    julia bfgs.jl $seed &
    julia bfgs_adam.jl $seed &
    julia lbfgs.jl $seed &
    julia adam.jl $seed &
    julia lbfgs_adam.jl $seed &
    julia ncg.jl $seed &
    julia ncg_adam.jl $seed &
    julia adaptive_bfgs.jl $seed &
    julia bfgs_adam_early.jl $seed &
    julia bfgs_adam_late.jl $seed &
    wait 
done 
