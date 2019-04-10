# tests for 1d example
using Test

push!(LOAD_PATH, "/home/felix/Work/Research/Code")
cd("/home/felix/Work/Research/Code/BSE_k_ISDF/experiments")

using Revise # remove after debugging
using BSE_k_ISDF, LinearAlgebra, FFTW


l = 1.5
V_sp = r -> 20 * cos(4π / l * r) + 0.2 * sin(2π / l * r)

V_func = (r_1, r_2) -> 1 /  sqrt((r_1 - r_2)^2 + 0.01)
W_func = (r_1, r_2, l, L) -> 0.0625 * (3.0 + sin(2π / l * r_1)) * (3.0 + cos(4π / l * r_2)) *
    exp(-abs(BSE_k_ISDF.supercell_difference(r_1, r_2, L))^2 / (2 * (4 * l)^2)) * V_func(BSE_k_ISDF.supercell_difference(r_1, r_2, L), 0.)

# fixed parameters
N_r = 128

N_core = 0
N_v = 4
N_c = 10
N_k = 256

N_μ_cc = 22
N_μ_vv = 16
N_μ_vc = 21

sp_prob = BSE_k_ISDF.SPProblem1D(V_sp, l, N_r, N_k);
prob = BSE_k_ISDF.BSEProblem1D(sp_prob, N_core, N_v, N_c, V_func, W_func);
isdf = BSE_k_ISDF.ISDF(prob, N_μ_vv, N_μ_cc, N_μ_vc);

ir = 4100
jr = 34
r = prob.prob.r_super[ir]
rp = prob.prob.r_unit[jr]
W_func(r, rp, l, N_k * l)

1 / (N_k * l) * sum(exp(im * (q + G) * r) * prob.w_hat[iG, iGp, iq] * exp(-im * (q + Gp) * rp)
    for (iq, q) in enumerate(2 * pi / l .* (prob.prob.k_bz .- 0.5)),
    (iG, G) in enumerate(2 * pi / l .* (0:127)), (iGp, Gp) in enumerate(2 * pi / l .* (0:127)))

w_realspace = W_func.(prob.prob.r_super, prob.prob.r_unit', l, N_k * l)

D = BSE_k_ISDF.setup_D(prob);
V = BSE_k_ISDF.setup_V(prob, isdf);
W = BSE_k_ISDF.setup_W(prob, isdf);
H = BSE_k_ISDF.setup_H(prob, isdf);

@time V_dense = Matrix(V);
@time V_exact = BSE_k_ISDF.assemble_exact_V(prob)
# @time V_realspace = BSE_k_ISDF.assemble_exact_V_realspace(prob, V_func)
norm(V_dense - V_exact) / norm(V_exact)

@time W_dense = Matrix(W);
@time W_exact = BSE_k_ISDF.assemble_exact_W(prob)
# @time W_realspace = BSE_k_ISDF.assemble_exact_W_realspace(prob, W_func)
norm(W_dense - W_exact) / norm(W_exact)



direction = 1
N_iter = 200
σ = 0.1
g = ω -> 1 / π * σ / (ω^2 + σ^2)
E_min, E_max = 0.0, 30.0
Erange = E_min:0.001:E_max

optical_absorption_lanc = BSE_k_ISDF.lanczos_optical_absorption(prob, isdf, direction, N_iter, g, Erange)

plot!(Erange, optical_absorption_lanc)
