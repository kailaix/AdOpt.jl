using ADCME
using PyPlot 
using JLD2
using AdFem
using ADCME
using PyPlot 
using JLD2
using Statistics 



function kappa(x, y)
    return 2 + exp(10x) - (10y)^2
end

function f(x, y)
    return sin(2π*10y+π/8)
end

mmesh = Mesh(joinpath(PDATA, "twoholes_large.stl"))

Kappa = eval_f_on_gauss_pts(kappa, mmesh)
xy = gauss_nodes(mmesh)

sess = Session()

MSE = Float64[]

# adam 
for SEED in [2, 23, 233, 2333, 23333]
    @load "data/adam$SEED.jld2" losses w
    loss1 = losses
    KappaNN = squeeze(fc(xy, [20, 20, 20, 1], w)) + 1.0
    push!(MSE, mean((run(sess,KappaNN)-Kappa).^2))

    @load "data/bfgs_adam$SEED.jld2" losses w
    loss3 = losses
    KappaNN = squeeze(fc(xy, [20, 20, 20, 1], w)) + 1.0
    push!(MSE, mean((run(sess,KappaNN)-Kappa).^2))


    @load "data/lbfgs_adam$SEED.jld2" losses w
    loss5 = losses
    KappaNN = squeeze(fc(xy, [20, 20, 20, 1], w)) + 1.0
    push!(MSE, mean((run(sess,KappaNN)-Kappa).^2))

    @load "data/tr_bfgs$SEED.jld2" losses w
    loss6 = losses
    KappaNN = squeeze(fc(xy, [20, 20, 20, 1], w)) + 1.0
    push!(MSE, mean((run(sess,KappaNN)-Kappa).^2))


    close("all")
    semilogy(loss1, label="Adam")
    semilogy(loss3, label="BFGS+Adam")
    semilogy(loss5, label="LBFGS+Adam")
    semilogy(loss6, label="Hybrid")
    legend()
    xlabel("Iterations")
    ylabel("Loss")
    savefig("data/loss$SEED.png")

end