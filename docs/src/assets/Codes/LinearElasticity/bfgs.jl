SEED = 233
if length(ARGS)>=1
    SEED = parse(Int64, ARGS[1])
end
@info "seed = $SEED"
MODE = "bfgs"

if isfile("data/result$MODE/data$SEED.jld2")
    exit()
end

include("inverse.jl")

make_directory("data/result$MODE")

losses = Optimize!(sess, loss, BFGSOptimizer(), 500)

THETA = run(sess, θ1)

@save "data/result$MODE/data$MODE.jld2" THETA losses

close("all")
semilogy(losses)
savefig("data/result$MODE/loss$MODE.png")

E_, nu_ = run(sess, [E, nu])
close("all")
figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_gauss_points(E_, mmesh)
subplot(122)
visualize_scalar_on_gauss_points(nu_, mmesh)
savefig("data/result$MODE/final$MODE.png")
