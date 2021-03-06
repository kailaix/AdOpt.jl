SEED = 233
if length(ARGS)>=1
    SEED = parse(Int64, ARGS[1])
end
@info "seed = $SEED"

# if isfile("data/adam$SEED.jld2")
#     exit()
# end

include("inverse.jl")

opt = AdamOptimizer().minimize(loss)
sess = Session(); init(sess)

losses = Float64[run(sess, loss)]
for i = 1:1000
    _, l = run(sess, [opt, loss])
    push!(losses, l)
    @info i, l 
end

w = run(sess, θ)
make_directory("data")
@save "data/adam$SEED.jld2" losses w 

figure(figsize = (10, 4))
subplot(121)
semilogy(losses)
xlabel("Iterations"); ylabel("Loss")
subplot(122)
visualize_scalar_on_gauss_points(run(sess, Kappa), mmesh)
savefig("data/adam$SEED.png")