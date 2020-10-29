SEED = 233
if length(ARGS)>=1
    SEED = parse(Int64, ARGS[1])
end
@info "seed = $SEED"

if isfile("data/bfgs$SEED.jld2")
    exit()
end

include("inverse.jl")

opt = AdamOptimizer().minimize(loss)
g = tf.convert_to_tensor(gradients(loss, θ))
sess = Session(); init(sess)

losses0 = Float64[]

# error()

losses = Optimize!(sess, loss, BFGSOptimizer(), 1000)

losses = [losses;losses0]
w = run(sess, θ)
make_directory("data")
@save "data/bfgs$SEED.jld2" losses w 

figure(figsize = (10, 4))
subplot(121)
semilogy(losses)
xlabel("Iterations"); ylabel("Loss")
subplot(122)
visualize_scalar_on_gauss_points(run(sess, Kappa), mmesh)
savefig("data/bfgs$SEED.png")