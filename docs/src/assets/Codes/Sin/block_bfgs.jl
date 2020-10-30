
SEED = 233
if length(ARGS)==1
    global SEED = parse(Int64, ARGS[1])
end
if isfile("data/block_bfgs$SEED.jld2")
    exit()
end

@info "SEED = $SEED"

using Revise 
using ADCME 
using LinearAlgebra
using LineSearches
using JLD2 
using AdOpt



using Random; Random.seed!(SEED)

x = LinRange(0, 1, 500)|>Array
y = sin.(10π*x)
θ = Variable(ae_init([1,20,20,20,1]))
z = squeeze(fc(x, [20, 20, 20, 1], θ))
loss = sum((z-y)^2)

sess = Session(); init(sess)
losses = Optimize!(sess, loss, BlockBFGSOptimizer(), 2000)

make_directory("data")
@save "data/block_bfgs$SEED.jld2" losses 

