ulimit -u 10000

for SEED in 2 23 233 2333 23333
do 
    julia adam.jl $SEED &
    julia bfgs_adam.jl $SEED &
    julia lbfgs_adam.jl $SEED &
    julia tr_bfgs.jl $SEED &
    wait
done 