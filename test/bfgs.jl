using ADOPT 
using Test
function f(x)
    return (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2
end

function g!(G, x)
    G[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    G[2] = 200.0 * (x[2] - x[1]^2)
end

initial_x = zeros(2)
options = Options()
options.show_trace = true
result = optimize(f, g!, initial_x, LBFGS(), options)
@test result.minimizer ≈ ones(2)