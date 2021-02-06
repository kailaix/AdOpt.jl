SEED = 233
if length(ARGS)>=1
    SEED = parse(Int64, ARGS[1])
end
@info "seed = $SEED"

if isfile("data/adamopt$SEED.jld2")
    exit()
end

include("inverse.jl")

N = 300
sess = Session(); init(sess)

losses0 = Float64[]

opt1 = ADAMOptimizer()
losses = Optimize!(sess, loss, opt1, 1000)
angles = opt1.angles
w = run(sess, θ)

make_directory("data")

figure(figsize = (4,10))
subplot(311)
semilogy(angles)
subplot(312)
semilogy(losses)
subplot(313)
semilogy(abs.(diff(losses) ./ losses[1:end-1]))
tight_layout()
savefig("data/adamopt_angles$SEED.png")
