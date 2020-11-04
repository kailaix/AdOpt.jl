using Revise
using AdOpt
using PyPlot
using LinearAlgebra

using Random; Random.seed!(233)
x = Array(collect(LinRange(-10.0, 10.0, 100)))
y = @. (25sin(x)-x^2)/100

θ = Variable(fc_init([1,20,20,20,1]))
p = squeeze(fc(x, [20,20,20,1], θ))

loss = sum((p-y)^2)
sess = Session(); init(sess)
opt = BFGSOptimizer()
loss2 = Optimize!(sess, loss, opt, 1000)

p0 = run(sess, p)
close("all")
plot(x, p0-y)
savefig("p0.png")