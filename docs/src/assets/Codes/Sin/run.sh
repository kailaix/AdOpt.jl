for SEED in  2 23 233 2333 23333
do 
julia adam.jl $SEED &
julia bfgs.jl $SEED &
julia lbfgs.jl $SEED &
julia ncg.jl $SEED &
wait 
julia plot.jl $SEED 
done 