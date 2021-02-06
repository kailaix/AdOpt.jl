
using ADCME
using PyPlot 
using Random
using AdOpt
using MAT
using DelimitedFiles
using JLD2 
mpl = pyimport("tikzplotlib")
@save "results/timing.jld2" NS FWD BWD 

for SEED in [2,23,233,2333,23333]
    close("all")
    @load "results/cH_ADAM$SEED.jld2" losses xH dif
    semilogy(losses, "+-", label = "ADAM")
    @load "results/cH_Hybrid$SEED.jld2" losses xH dif
    semilogy(losses, "x-", label = "Hybrid")
    @load "results/cH_BFGS$SEED.jld2" losses xH dif
    semilogy(losses, ".-", label = "BFGS")
    xlabel("Iterations"); ylabel("Loss")
    legend()
    savefig("Loss$SEED.png")
end

for SEED in [2,23,233,2333,23333]
    close("all")
    @load "results/cH_ADAM$SEED.jld2" losses xH dif
    semilogy(xH, dif, label = "ADAM")
    @load "results/cH_Hybrid$SEED.jld2" losses xH dif
    semilogy(xH, dif, label = "Hybrid")
    @load "results/cH_BFGS$SEED.jld2" losses xH dif
    semilogy(xH, dif, label = "BFGS")
    xlabel("x"); ylabel("Difference")
    legend()
    savefig("diff$SEED.png")
end

for SEED in [2,23,233,2333,23333]
    close("all")
    @load "results/cH_Hybrid$SEED.jld2" is_bfgs
    plot(is_bfgs, ".")
    xlabel("Iterations")
    savefig("isbfgs$SEED.png")
end
