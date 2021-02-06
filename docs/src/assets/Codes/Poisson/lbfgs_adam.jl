SEED = 233
if length(ARGS)>=1
    SEED = parse(Int64, ARGS[1])
end
@info "seed = $SEED"

# if isfile("data/lbfgs_adam$SEED.jld2")
#     exit()
# end

include("inverse.jl")

N = 50

opt = AdamOptimizer().minimize(loss)
g = tf.convert_to_tensor(gradients(loss, θ))
sess = Session(); init(sess)

losses0 = Float64[]

for i = 1:N
    _, l = run(sess, [opt, loss])
    @info i, l 
    push!(losses0, l)
end

# error()

losses = Optimize!(sess, loss, LBFGSOptimizer(), 1000-N)

losses = [losses0;losses]
w = run(sess, θ)
make_directory("data")
@save "data/lbfgs_adam$SEED.jld2" losses w 

figure(figsize = (10, 4))
subplot(121)
semilogy(losses)
xlabel("Iterations"); ylabel("Loss")
subplot(122)
visualize_scalar_on_gauss_points(run(sess, Kappa), mmesh)
savefig("data/lbfgs_adam$SEED.png")