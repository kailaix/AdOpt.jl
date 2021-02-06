ulimit -u 10000

for seed in 2 23 233 2333 23333
do 
    julia bfgs_adam.jl $seed &
    julia adam.jl $seed &
    julia lbfgs_adam.jl $seed &
    julia tr_bfgs.jl $seed &
    wait
done 
