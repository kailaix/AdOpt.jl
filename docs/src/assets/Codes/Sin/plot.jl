
SEED = 233
if length(ARGS)==1
    global SEED = parse(Int64, ARGS[1])
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

@load "data/ncgbfgs$SEED.jld2" losses
semilogy(losses, label = "NCGBFGS")

@load "data/adaptive_bfgs$SEED.jld2" losses
semilogy(losses, label = "Adaptive BFGS")


legend()
xlabel("Iterations")
ylabel("Loss")
savefig("data/sinloss$SEED.png")
