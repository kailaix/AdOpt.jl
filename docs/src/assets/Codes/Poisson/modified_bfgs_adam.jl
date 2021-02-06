SEED = 2333
if length(ARGS)>=1
    SEED = parse(Int64, ARGS[1])
end
@info "seed = $SEED"

if isfile("data/modified_bfgs_adam$SEED.jld2")
    exit()
end

include("inverse.jl")

N = 100

# opt = AdamOptimizer().minimize(loss)
sess = Session(); init(sess)

losses0 = Float64[]

# angles = Float64[]
# for i = 1:N
#     _, l = run(sess, [opt, loss])
#     @info i, l 
#     push!(losses0, l)
# end

# error()

opt = BFGSOptimizer()
losses = Optimize!(sess, loss, opt, 1000-N)


# BFGS!(sess, loss)

figure(figsize = (15, 4))
subplot(131)
visualize_scalar_on_gauss_points(eval_f_on_gauss_pts(kappa, mmesh), mmesh)
subplot(132)
visualize_scalar_on_gauss_points(run(sess, Kappa), mmesh)
subplot(133)
visualize_scalar_on_gauss_points(abs.(run(sess, Kappa)-eval_f_on_gauss_pts(kappa, mmesh)), mmesh)
savefig("data/linear_adam$SEED.png")

error()
opt = BFGSOptimizer()
losses = Optimize!(sess, loss, opt, 1000-N)

losses = [losses0;losses;]
w = run(sess, θ)

make_directory("data")

@save "data/modified_bfgs_adam$SEED.jld2" losses w 

figure(figsize = (10, 4))
subplot(121)
semilogy(losses)
xlabel("Iterations"); ylabel("Loss")
subplot(122)
visualize_scalar_on_gauss_points(run(sess, Kappa), mmesh)
savefig("data/modified_bfgs_adam$SEED.png")