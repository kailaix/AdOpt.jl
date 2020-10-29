using ADCME
using JLD2
using Optim
using AdOpt

SEED = 233
if length(ARGS)==1
    global SEED = parse(Int64, ARGS[1])
end
using Random; Random.seed!(SEED)

x = LinRange(0, 1, 500)|>Array
y = sin.(10π*x)
θ = Variable(ae_init([1,20,20,20,1]))
z = squeeze(fc(x, [20, 20, 20, 1], θ))
loss = sum((z-y)^2)

opt = AdamOptimizer().minimize(loss)
sess = Session(); init(sess)

losses = Float64[]
for i = 1:2000
    _, l = run(sess, [opt, loss])
    push!(losses, l )
end

make_directory("data")
@save "data/adam.jld2" losses 
