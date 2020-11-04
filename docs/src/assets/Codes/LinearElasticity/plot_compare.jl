using PyPlot 
using ADCME
using JLD2

for SEED in [2,23,233,2333,23333]
close("all")
@load "data/resultbfgs_adam/data$SEED.jld2" losses THETA
semilogy(losses, label = "BFGS+ADAM")
@load "data/resultlbfgs_adam/data$SEED.jld2" losses THETA
semilogy(losses, label = "LBFGS+ADAM")
@load "data/resulttr_bfgs/data$SEED.jld2" losses THETA
semilogy(losses, label = "Hybrid")
@load "data/resultadam/data$SEED.jld2" losses THETA
semilogy(losses, label = "ADAM")
legend()
savefig("$SEED.png")
end
