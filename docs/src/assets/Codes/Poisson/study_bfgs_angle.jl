SEED = 2
include("inverse.jl")
make_directory("figures")
using PyCall
mpl = pyimport("tikzplotlib")

@load "data/tr_bfgs2.jld2" isbfgs losses angles
loss_tr = losses 
tr_angles = angles 

s = 0
for k = length(loss_tr)-1:-1:1
    global s 
    if loss_tr[k]!=loss_tr[k+1]
        s = k + 1
        break 
    end
end
loss_tr = loss_tr[1:s]
isbfgs = isbfgs[1:s]
tr_angles = tr_angles[1:s]
@load "data/bfgs_adam2.jld2" losses angles 

close("all")
semilogy(angles)
xlabel("Iterations")
ylabel("Cosine Similarity")
grid("on")
savefig("figures/bfgs_angles.png")
mpl.save("figures/bfgs_angles.tex")


close("all")
N = 1:length(loss_tr)
semilogy(N[51:end], loss_tr[51:end], label = "Hybrid")
semilogy(losses, label = "BFGS")
semilogy(N[1:50], loss_tr[1:50], "g", label = "Warm Start")
semilogy(N[.!(isbfgs)][51:end], loss_tr[.!(isbfgs)][51:end], "xg", label="ADAM")
legend()
grid("on")
savefig("figures/poisson_isbfgs.png")
mpl.save("figures/poisson_isbfgs.tex")

close("all")
tr_angles[tr_angles.<0.0] .= NaN
semilogy(tr_angles)
xlabel("Iterations")
ylabel("Cosine Similarity")
grid("on")
savefig("figures/tr_angles.png")
mpl.save("figures/tr_angles.tex")


