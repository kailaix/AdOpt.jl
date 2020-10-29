using ADCME
using PyPlot 
using JLD2
using AdFem
using ADCME
using PyPlot 
using JLD2
using Statistics 

SEED = 233
if length(ARGS)>=1
    SEED = parse(Int64, ARGS[1])
end
@info "seed = $SEED"

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
@load "data/adam$SEED.jld2" losses w
loss1 = losses
KappaNN = squeeze(fc(xy, [20, 20, 20, 1], w)) + 1.0
push!(MSE, mean((run(sess,KappaNN)-Kappa).^2))

@load "data/bfgs$SEED.jld2" losses w
loss2 = losses
KappaNN = squeeze(fc(xy, [20, 20, 20, 1], w)) + 1.0
push!(MSE, mean((run(sess,KappaNN)-Kappa).^2))


@load "data/bfgs_adam$SEED.jld2" losses w
loss3 = losses
KappaNN = squeeze(fc(xy, [20, 20, 20, 1], w)) + 1.0
push!(MSE, mean((run(sess,KappaNN)-Kappa).^2))

@load "data/lbfgs$SEED.jld2" losses w
loss4 = losses
KappaNN = squeeze(fc(xy, [20, 20, 20, 1], w)) + 1.0
push!(MSE, mean((run(sess,KappaNN)-Kappa).^2))

@load "data/lbfgs_adam$SEED.jld2" losses w
loss5 = losses
KappaNN = squeeze(fc(xy, [20, 20, 20, 1], w)) + 1.0
push!(MSE, mean((run(sess,KappaNN)-Kappa).^2))


MSE = round.(MSE, sigdigits=2)

close("all")
semilogy(loss1, label="Adam")
semilogy(loss2, label="BFGS")
semilogy(loss3, label="BFGS+Adam")
semilogy(loss4, label="LBFGS")
semilogy(loss5, label="LBFGS+Adam")
legend()
xlabel("Iterations")
ylabel("Loss")
savefig("data/loss$SEED.png")
