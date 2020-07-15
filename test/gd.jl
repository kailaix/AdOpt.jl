using ADOPT 
using Test

function f(x)
    global i 
    i += 1
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function g!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    G[2] = 200.0 * (x[2] - x[1]^2)
end

initial_x = zeros(2)

options = Options()
options.iterations = 5000

result = optimize(f, g!, initial_x, GradientDescent(), options)
@test result.minimizer ≈ ones(2)