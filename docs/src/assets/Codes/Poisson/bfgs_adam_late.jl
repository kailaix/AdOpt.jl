SEED = 233
if length(ARGS)>=1
    SEED = parse(Int64, ARGS[1])
end
@info "seed = $SEED"

if isfile("data/bfgs_adam_late$SEED.jld2")
    exit()
end

include("inverse.jl")

N = 500

opt = AdamOptimizer().minimize(loss)
g = tf.convert_to_tensor(gradients(loss, θ))
sess = Session(); init(sess)

losses0 = Float64[]
B = diagm(0=>ones(length(θ)))
G, THETA = run(sess, [g,θ])

angles = Float64[]
for i = 1:N
    _, G, l = run(sess, [opt, g, loss])
    @info i, l 
    push!(losses0, l)
end

# error()
opt = BFGSOptimizer()
losses = Optimize!(sess, loss, opt, 1000-N)

losses = [losses0;losses;]
w = run(sess, θ)

make_directory("data")


figure(figsize = (4,10))
subplot(211)
semilogy(opt.angles)
subplot(212)
semilogy(losses)
savefig("data/bfgs_angle$SEED.png")


@save "data/bfgs_adam_late$SEED.jld2" losses w 

figure(figsize = (10, 4))
subplot(121)
semilogy(losses)
xlabel("Iterations"); ylabel("Loss")
subplot(122)
visualize_scalar_on_gauss_points(run(sess, Kappa), mmesh)
savefig("data/bfgs_adam_late$SEED.png")