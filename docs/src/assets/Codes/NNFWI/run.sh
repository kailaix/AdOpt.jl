ulimit -u 10000
for SEED in 2 23
do 
julia inverse_bfgs.jl $SEED &
julia inverse_hybrid.jl $SEED &
julia inverse_adam.jl $SEED &
done 
wait 