SEED = 233
if length(ARGS)>=1
    SEED = parse(Int64, ARGS[1])
end
@info "seed = $SEED"
MODE = "tr_bfgs"

if isfile("data/result$MODE/data$SEED.jld2")
    exit()
end

include("inverse.jl")

make_directory("data/result$MODE")

opt = HybridAdamBFGSOptimizer()
losses = Optimize!(sess, loss, opt, 500)

THETA = run(sess, θ1)

x, isbfgs = opt.x, opt.is_bfgs
@save "data/result$MODE/data$SEED.jld2" THETA losses x isbfgs

close("all")
semilogy(losses)
savefig("data/result$MODE/loss$SEED.png")

E_, nu_ = run(sess, [E, nu])
close("all")
figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_gauss_points(E_, mmesh)
subplot(122)
visualize_scalar_on_gauss_points(nu_, mmesh)
savefig("data/result$MODE/final$SEED.png")
