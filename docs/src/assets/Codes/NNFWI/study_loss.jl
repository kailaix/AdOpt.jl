
using ADCME
using PyPlot 
using Random
using AdOpt
using MAT
using DelimitedFiles
using JLD2 
mpl = pyimport("tikzplotlib")

for SEED in [2,2333]
    close("all")
    @load "results/cH_ADAM$SEED.jld2" losses xH dif
    semilogy(losses, label = "ADAM")
    @load "results/cH_Hybrid$SEED.jld2" losses xH dif is_bfgs
    N = length(losses)
    semilogy(losses, label = "Hybrid")
    is_adam = @. !is_bfgs
    is_adam = [true;true;is_adam]
    plot((0:N-1)[is_adam], losses[is_adam], "x", label = "ADAM")
    @load "results/cH_BFGS$SEED.jld2" losses xH dif

    semilogy(losses, label = "BFGS")
    xlabel("Iterations"); ylabel("Loss")
    legend()
    savefig("figures/nnfwi_Loss$SEED.png")
    mpl.save("figures/nnfwi_Loss$SEED.tex")
end

for SEED in [2,2333]
    close("all")
    @load "results/cH_ADAM$SEED.jld2" losses xH dif
    semilogy(xH, dif, label = "ADAM")
    @load "results/cH_Hybrid$SEED.jld2" losses xH dif
    semilogy(xH, dif, label = "Hybrid")
    @load "results/cH_BFGS$SEED.jld2" losses xH dif
    semilogy(xH, dif, label = "BFGS")
    xlabel("x"); ylabel("Difference")
    legend()
    savefig("figures/nnfwi_diff$SEED.png")
    mpl.save("figures/nnfwi_diff$SEED.tex")
end
