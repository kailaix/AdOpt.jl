using Revise
using AdOpt
using PyPlot
using LinearAlgebra


x = Array(collect(LinRange(-10.0, 10.0, 100)))
y = @. (25sin(x)-x^2)/100

xp = Array(collect(LinRange(-10.0, 10.0, 10)))
yp = @. (25sin(xp)-xp^2)/100
s = interp1(xp, yp, x)



using Random; Random.seed!(233)

θ = Variable(fc_init([1,20,20,20,1]))
p = squeeze(fc(x, [20,20,20,1], θ)) 

l = sum((s-p)^2)
loss = sum((p-y)^2)

sess = Session(); init(sess)
BFGS!(sess, l, 1000)
opt = BFGSOptimizer()
loss1 = Optimize!(sess, loss, opt, 1000)

p0 = run(sess, p)
close("all")
plot(x, p0-y)
savefig("p0.png")