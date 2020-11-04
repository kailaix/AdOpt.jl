
SEED = 233
if length(ARGS)==1
    global SEED = parse(Int64, ARGS[1])
end
if isfile("data/modified_bfgs$SEED.jld2")
    exit()
end

using Revise 
using ADCME 
using LinearAlgebra
using LineSearches
using JLD2 
using AdOpt
using PyPlot



using Random; Random.seed!(SEED)

x = LinRange(0, 1, 500)|>Array
y = sin.(10π*x)
θ = Variable(ae_init([1,20,20,20,1]))
z = squeeze(fc(x, [20, 20, 20, 1], θ))
loss = sum((z-y)^2)

sess = Session(); init(sess)
opt = ModifiedBFGSOptimizer()
losses = Optimize!(sess, loss, opt, 2000)

@save "data/modified_bfgs$SEED.jld2" losses 

