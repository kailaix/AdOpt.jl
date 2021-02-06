
SEED = 233
if length(ARGS)==1
    global SEED = parse(Int64, ARGS[1])
end
if isfile("data/bfgs$SEED.jld2")
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
opt = BFGSOptimizer()
losses = Optimize!(sess, loss, opt, 2000)
angles = opt.angles
make_directory("data")
figure(figsize = (4,10))
subplot(211)
semilogy(angles)
subplot(212)
semilogy(losses)
savefig("data/bfgs$SEED.png")
@save "data/bfgs$SEED.jld2" losses 

