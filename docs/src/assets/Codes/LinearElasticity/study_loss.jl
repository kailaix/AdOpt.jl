using PyPlot 
using ADCME
using JLD2
using PyCall
using MAT
make_directory("figures")

mpl = pyimport("tikzplotlib")
for SEED in [2,23333]
close("all")
@load "data/resultadam/data$SEED.jld2" losses THETA
semilogy(losses, label = "ADAM")
@load "data/resultbfgs_adam/data$SEED.jld2" losses THETA
semilogy(losses, label = "BFGS+ADAM")
@load "data/resultlbfgs_adam/data$SEED.jld2" losses THETA
semilogy(losses, label = "LBFGS+ADAM")
@load "data/resulttr_bfgs/data$SEED.jld2" losses THETA
semilogy(losses, label = "Hybrid")

legend()
grid("on")
savefig("figures/le_loss$SEED.png")
mpl.save("figures/le_loss$SEED.tex")
end



Eexact = matread("data/fwd.mat")["E"]
include("inverse.jl")
@load "data/resulttr_bfgs/data2.jld2" x
E_ = run(sess, E, θ1=>x)
close("all")
figure(figsize=(10,4))
subplot(121)
visualize_scalar_on_gauss_points(E_, mmesh)
title("\$E\$")
subplot(122)
visualize_scalar_on_gauss_points(abs.(E_-Eexact), mmesh)
title("Error")
savefig("figures/le_compare.png")
savefig("figures/le_compare.pdf")


function f(x, y)
    sin(10*π*x) + (10y-20x)^2 + 1.0
end

mmesh = Mesh(joinpath(PDATA, "twoholes.stl"), degree=2)

left = bcnode((x,y)->x<1e-5, mmesh)
right = bcedge((x1,y1,x2,y2)->(x1>0.049-1e-5) && (x2>0.049-1e-5), mmesh)

t1 = eval_f_on_boundary_edge((x,y)->1.0e-4, right, mmesh)
t2 = eval_f_on_boundary_edge((x,y)->0.0, right, mmesh)
rhs = compute_fem_traction_term(t1, t2, right, mmesh)

ν = 0.3 * ones(get_ngauss(mmesh))
E = eval_f_on_gauss_pts(f, mmesh)
D = compute_plane_stress_matrix(E, ν)
K = compute_fem_stiffness_matrix(D, mmesh)

bdval = [eval_f_on_boundary_node((x,y)->0.0, left, mmesh);
        eval_f_on_boundary_node((x,y)->0.0, left, mmesh)]
DOF = [left;left .+ mmesh.ndof]
K, rhs = impose_Dirichlet_boundary_conditions(K, rhs, DOF, bdval)
u = K\rhs 
sess = Session(); init(sess)
S = run(sess, u)



function visualize_mesh2(mesh::Mesh)
    nodes, elems = mesh.nodes, mesh.elems
    patches = PyObject[]
    for i = 1:size(elems,1)
        e = elems[i,:]
        p = plt.Polygon(nodes[e,:],edgecolor="k",lw=0.3,alpha = 0.5, fc=nothing,fill=false)
        push!(patches, p)
    end
    p = matplotlib.collections.PatchCollection(patches, match_original=true)
    gca().add_collection(p)
    axis("scaled")
    xlabel("x")
    ylabel("y")
end

close("all")
figure(figsize=(20, 5))
subplot(131)
visualize_scalar_on_gauss_points(E, mmesh)
title("\$E\$")
subplot(132)
visualize_mesh2(mmesh)
visualize_vector_on_fem_points(S[1:mmesh.nnode], S[1+mmesh.ndof:mmesh.nnode + mmesh.ndof], mmesh)
title("Displacement")
subplot(133)
Dval = run(sess, D)
visualize_von_mises_stress(Dval, S, mmesh)
title("von Mises Stress")
savefig("figures/le_solution.png")
savefig("figures/le_solution.pdf")