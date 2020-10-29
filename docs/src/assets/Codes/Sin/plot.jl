
SEED = 233
if length(ARGS)==1
    global SEED = parse(Int64, ARGS[1])
end
if isfile("data/sinloss$SEED.png")
    exit()
end

using PyPlot 
using JLD2
using ADCME


using Random; Random.seed!(SEED)

make_directory("data")

close("all")
@load "data/adam$SEED.jld2" losses
semilogy(losses, label = "Adam")


@load "data/bfgs$SEED.jld2" losses
semilogy(losses, label = "BFGS")

@load "data/lbfgs$SEED.jld2" losses
semilogy(losses, label = "LBFGS")

@load "data/ncg$SEED.jld2" losses
semilogy(losses, label = "NCG")

legend()
xlabel("Iterations")
ylabel("Loss")
savefig("data/sinloss$SEED.png")
