using PyPlot 
using JLD2
using ADCME

make_directory("data")

close("all")
@load "data/adam.jld2" losses
semilogy(losses, label = "Adam")


@load "data/bfgs.jld2" losses
semilogy(losses, label = "BFGS")

@load "data/lbfgs.jld2" losses
semilogy(losses, label = "LBFGS")

legend()
xlabel("Iterations")
ylabel("Loss")
savefig("data/sinloss.png")
