SEED = 233

using ADCME
using PyPlot 
using Random
using AdOpt
using MAT
using DelimitedFiles
matplotlib.use("agg")

FWD = zeros(8)
BWD = zeros(8)
NS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
for (kn,n) in enumerate(NS)
    global Nk = n
    include("timing_forward.jl")
    reset_default_graph()
    tf.set_random_seed(123)
    tf.random.set_random_seed(123)

    R_ = constant(R)

    idx_var = pml+Int(round(0.1/Δx))

    inv_sigmoid(y) = @. - log(1.0/y - 1.0)

    # case = "fwi"
    case = "nnfwi"

    ## fwi
    if case == "fwi"
    if model == "step"
        var= Variable(zeros(N-1-idx_var)) + inv_sigmoid(1.3/2.0)
        var = -2.0*sigmoid(var)
        # var= Variable(zeros(N-1-idx_var)) - 1.3
    elseif model == "slop" || model == "cont"
        var = Variable(zeros(N-1-idx_var))
    end
    end

    ## FC
    if case == "nnfwi"
    x0 = collect( range(0, 1, length=N-1-idx_var) )
    using Random; Random.seed!(SEED)
    init_guess = fc_init([1, 30, 30, 30, 1])
    θ = Variable(init_guess)
    if model == "step"
        var= fc(x0, [30, 30, 30, 1], θ) * 0.1 + inv_sigmoid(1.3/2.0)
        var = -2.0*sigmoid(var)
    elseif model == "slop" || model == "cont"
        global var= fc(x0, [30, 30, 30, 1], θ) * 0.1 
    end
    end


    cH = scatter_add(constant(cH_init), idx_var+1:N-pml, var[1:end-pml+1])

    cE = (cH[1:end-1]+cH[2:end])/2

    reg_TV = sum(abs(cH[2:end]-cH[1:end-1]))
    reg_smooth = sum((cH[1:end-2] + cH[3:end] - 2*cH[2:end-1])^2)

    function condition(i, E_arr, H_arr)
        i<=NT+1
    end

    function body(i, E_arr, H_arr)
        E = read(E_arr, i-1)
        H = read(H_arr, i-1)
        
        ΔH = cH * (E[2:end]-E[1:end-1])/Δx - σH*H
        H += ΔH * Δt
        ΔH = 1/(24Δx) * cH[2:N-2] * (-E[4:end] + 3E[3:end-1] - 3E[2:end-2] + E[1:end-3]) 
        H = scatter_add(H, 2:N-2, ΔH * Δt)
        
        ΔE = cE * (H[2:end]-H[1:end-1])/Δx - σE[2:end-1]*E[2:end-1] + R_[i] * Z
        E = scatter_add(E, 2:N-1, ΔE * Δt)
        ΔE = 1/(24Δx) * cE[2:end-1] * (-H[4:end] + 3H[3:end-1] - 3H[2:end-2] + H[1:end-3]) 
        E = scatter_add(E, 3:N-2, ΔE*Δt)

        i+1, write(E_arr, i, E), write(H_arr, i, H)
    end

    E_arr = TensorArray(NT+1)
    H_arr = TensorArray(NT+1)

    E_arr = write(E_arr, 1, zeros(N))
    H_arr = write(H_arr, 1, zeros(N-1))

    i = constant(2, dtype = Int32)

    _, E, H = while_loop(condition, body, [i, E_arr, H_arr])

    E = stack(E); E = set_shape(E, (NT+1, N))
    H = stack(H)

    loss = sum((E[:, idx_rcv] - E_true[:, idx_rcv])^2 * [1.0, 1e-3])
    loss1 = sum((E[:, idx_rcv[1]] - E_true[:, idx_rcv[1]])^2 * 1.0)
    loss2 = sum((E[:, idx_rcv[2]] - E_true[:, idx_rcv[2]])^2 * 1e-3)

    if case == "fwi"
        # loss += 1e3*reg_smooth + 0.0*reg_TV
        # case = string(case, "_reg_smooth")
        loss += 1e2*reg_smooth + 1e5*reg_TV
        case = string(case, "_reg1e2_smooth_TV_3")
    end


    g = gradients(loss, θ)
    sess = Session(); init(sess)

    fwd = Float64[]
    bwd = Float64[]

    for i = 1:11
        d = @timed run(sess, g)
        push!(bwd, d[2])
    end

    for i = 1:11
        d = @timed run(sess, loss)
        push!(fwd, d[2])
    end

    using Statistics 
    bwd = mean(bwd[2:end])
    fwd = mean(fwd[2:end])
    FWD[kn] = fwd
    BWD[kn] = bwd 
    @info n, fwd, bwd
end

using JLD2 
using PyCall 
mpl = pyimport("tikzplotlib")
@save "results/timing.jld2" NS FWD BWD 


close("all")
p1 = bar(NS.-75, FWD, 150)
p2 = bar(NS.+75, BWD, 150)
legend((p1[1], p2[1]), ("Forward Computation", "Gradient Back-propagation"))
grid("on")
xlabel("\$n_T\$")
ylabel("Time (sec)")
savefig("figures/nnfwi_timing.png")
mpl.save("figures/nnfwi_timing.tex")