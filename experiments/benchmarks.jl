# benchmarks for ISDF in BSE

# %%
# loading packages
using BenchmarkTools, JLD2, FileIO, LinearAlgebra, FFTW
using Plots, LaTeXStrings

BLAS.set_num_threads(1)
FFTW.set_num_threads(1)

push!(LOAD_PATH, "/home/oim3l/Work/Projects/Excitons/Code")
cd("/home/oim3l/Work/Projects/Excitons/Code/BSE_k_ISDF/experiments")

using Revise #remove after debugging
using BSE_k_ISDF

# general problem setup

# example_string = "ex1"
# l = 3.0
# V_sp = r -> -100 * exp(-(r - 0.3 * l)^2 / (2 * (0.05 * l)^2)) - 60 * exp(-(r - 0.6 * l)^2 / (2 * (0.05 * l)^2))

example_string = "ex2"
l = 1.5
V_sp = r -> 20 * cos(4π / l * r) + 0.2 * sin(2π / l * r)

V = (r_1, r_2) -> 1 /  sqrt((r_1 - r_2)^2 + 0.01)
W = (r_1, r_2, l, L) -> 0.0625 * (3.0 + sin(2π / l * r_1)) * (3.0 + cos(4π / l * r_2)) *
    exp(-abs(BSE_k_ISDF.supercell_difference(r_1, r_2, L))^2 / (2 * (4 * l)^2)) * V(BSE_k_ISDF.supercell_difference(r_1, r_2, L), 0.)

isdir("results_" * example_string) || mkdir("results_" * example_string)

# %% benchmark different values of N_k

# fixed parameters
N_unit = 128

N_core = 0
N_v = 4
N_c = 5

N_μ_cc = 22
N_μ_vv = 16
N_μ_vc = 21

# variable parameters
N_k_vec = 2 .^(4:12)

# benchmark loop

for N_k in N_k_vec
    sp_prob = BSE_k_ISDF.SPProblem(V_sp, l, N_unit, N_k)
    prob = BSE_k_ISDF.BSEProblem(sp_prob, N_core, N_v, N_c, V, W)

    isdf = BSE_k_ISDF.ISDF(prob, N_μ_vv, N_μ_cc, N_μ_vc)
    t_isdf = @benchmark BSE_k_ISDF.ISDF($prob, $N_μ_vv, $N_μ_cc, $N_μ_vc)

    H = BSE_k_ISDF.setup_H(prob, isdf)
    t_H_setup = @benchmark BSE_k_ISDF.setup_H($prob, $isdf)

    x = rand(N_v * N_c * N_k) + im * rand(N_v * N_c * N_k)
    H * x
    t_H_x = @benchmark $H * $x

    save("results_" * example_string * "/benchmark_$(N_unit)_$(N_k).jld2", "t_isdf", t_isdf, "t_H_setup", t_H_setup, "t_H_x", t_H_x)
end

# %% benchmark different values of N_μ

# fixed parameters
N_unit = 128

N_core = 0
N_v = 4
N_c = 5
N_k = 256

sp_prob = BSE_k_ISDF.SPProblem(V_sp, l, N_unit, N_k)
prob = BSE_k_ISDF.BSEProblem(sp_prob, N_core, N_v, N_c, V, W)

N_μ_cc_vec, N_μ_vv_vec, N_μ_vc_vec, errors_M_cc, errors_M_vv, errors_M_vc =  load("results_" * example_string * "/errors_M_$(N_unit)_$(N_k).jld2", "N_μ_cc_vec", "N_μ_vv_vec", "N_μ_vc_vec", "errors_M_cc", "errors_M_vv", "errors_M_vc")

# variable parameters
M_tol_vec = [0.8, 0.5, 0.25, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

# benchmark loop
for M_tol in M_tol_vec
    N_μ_vv = findfirst(errors_M_vv .<= M_tol)
    N_μ_cc = findfirst(errors_M_cc .<= M_tol)
    N_μ_vc = findfirst(errors_M_vc .<= M_tol)

    t_H_entry = @benchmark BSE_k_ISDF.H_entry_fast($prob.v_hat, $prob.w_hat, 1, 3, 15, 2, 3, 60, $prob.E_v, $prob.E_c, $prob.u_v, $prob.u_c, $prob.prob.r_super, $prob.prob.r_unit, $prob.prob.k_bz)

    isdf = BSE_k_ISDF.ISDF(prob, N_μ_vv, N_μ_cc, N_μ_vc)
    t_isdf = @benchmark BSE_k_ISDF.ISDF($prob, $N_μ_vv, $N_μ_cc, $N_μ_vc)

    H = BSE_k_ISDF.setup_H(prob, isdf)
    t_H_setup = @benchmark BSE_k_ISDF.setup_H($prob, $isdf)

    x = rand(N_v * N_c * N_k) + im * rand(N_v * N_c * N_k)
    H * x
    t_H_x = @benchmark $H * $x

    save("results_" * example_string * "/benchmark_$(N_unit)_$(N_k)_$(M_tol).jld2", "t_isdf", t_isdf, "t_H_setup", t_H_setup, "t_H_x", t_H_x, "t_H_entry", t_H_entry)
end
