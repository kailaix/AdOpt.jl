for SEED in 1 2 3 4 5
do 
julia adam.jl $SEED &
julia bfgs.jl $SEED &
julia lbfgs.jl $SEED &
wait 
julia plot.jl $SEED 
done 