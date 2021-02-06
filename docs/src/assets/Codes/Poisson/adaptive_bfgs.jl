SEED = 233
if length(ARGS)>=1
    SEED = parse(Int64, ARGS[1])
end
@info "seed = $SEED"

if isfile("data/adaptive_bfgs$SEED.jld2")
    exit()
end

include("inverse.jl")


sess = Session(); init(sess)
opt = AdaptiveBFGSOptimizer()
losses = Optimize!(sess, loss, opt, 1000)

w = run(sess, θ)
make_directory("data")


# figure(figsize = (4,10))
# subplot(211)
# semilogy(opt.angles)
# subplot(212)
# semilogy(losses)
# savefig("data/adaptive_bfgs_angle$SEED.png")


@save "data/adaptive_bfgs$SEED.jld2" losses w 

figure(figsize = (10, 4))
subplot(121)
semilogy(losses)
xlabel("Iterations"); ylabel("Loss")
subplot(122)
visualize_scalar_on_gauss_points(run(sess, Kappa), mmesh)
savefig("data/adaptive_bfgs$SEED.png")