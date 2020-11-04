SEED = 23
if length(ARGS)>=1
    SEED = parse(Int64, ARGS[1])
end
@info "seed = $SEED"
MODE = "modified_bfgs_adam"

if isfile("data/result$MODE/data$SEED.jld2")
    exit()
end

include("inverse.jl")



make_directory("data/result$MODE")
opt = AdamOptimizer().minimize(loss)

sess = Session(); init(sess)
# run(sess, loss)
losses0 = Float64[]

for i = 1:50
    _, l = run(sess, [opt, loss])
    @info i, l 
    push!(losses0, l)
end


losses = Optimize!(sess, loss, BFGSOptimizer(beta=(0.0, 0.001)), 450)

losses = [losses0; losses]

THETA = run(sess, θ1)

@save "data/result$MODE/data$SEED.jld2" THETA losses

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
