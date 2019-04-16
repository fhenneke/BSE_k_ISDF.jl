# error analysis for ISDF in BSE

# %%
# loading packages
using BenchmarkTools, JLD2, FileIO, LinearAlgebra, FFTW, Statistics, Arpack
# using Plots, LaTeXStrings
# pyplot()
# theme(:dark)

BLAS.set_num_threads(1)
FFTW.set_num_threads(1)

push!(LOAD_PATH, "/home/felix/Work/Research/Code")
cd("/home/felix/Work/Research/Code/BSE_k_ISDF/experiments")

using Revise #remove after debugging
using BSE_k_ISDF

# general problem setup
l = 1.5
V_sp = r -> 20 * cos(4π / l * r) + 0.2 * sin(2π / l * r)

V_func = (r_1, r_2) -> 1 /  sqrt((r_1 - r_2)^2 + 0.01)
W_func = (r_1, r_2, l, L) -> 0.0625 * (3.0 + sin(2π / l * r_1)) * (3.0 + cos(4π / l * r_2)) *
    exp(-abs(BSE_k_ISDF.supercell_difference(r_1, r_2, L))^2 / (2 * (4 * l)^2)) * V_func(BSE_k_ISDF.supercell_difference(r_1, r_2, L), 0.)

# fixed parameters
N_r = 128

N_core = 0
N_v = 4
N_c = 5

# set up problem
sp_prob = BSE_k_ISDF.SPProblem1D(V_sp, l, N_r, N_k);
prob = BSE_k_ISDF.BSEProblem1D(sp_prob, N_core, N_v, N_c, V_func, W_func);

u_v, u_c = prob.u_v, prob.u_c

# variable parameters
N_μ_vec = 1:40


errors_Z_vv = []
errors_Z_cc = []
errors_Z_vc = []

@time for N_μ_vv in N_μ_vec
    r_μ_vv_indices = BSE_k_ISDF.find_r_μ(prob, N_μ_vv)
    ζ_vv = BSE_k_ISDF.assemble_ζ(u_v, r_μ_vv_indices)
    # error_Z_vv = BSE_k_ISDF.isdf_error_estimate(u_v, ζ_vv, r_μ_vv_indices, N_k_samples)
    error_Z_vv = BSE_k_ISDF.isdf_error(u_v, ζ_vv, r_μ_vv_indices)

    push!(errors_Z_vv, error_Z_vv)
end
@time for N_μ_cc in N_μ_vec
    r_μ_cc_indices = BSE_k_ISDF.find_r_μ(prob, N_μ_cc)
    ζ_cc = BSE_k_ISDF.assemble_ζ(u_c, r_μ_cc_indices)
    # error_Z_cc = BSE_k_ISDF.isdf_error_estimate(u_c, ζ_cc, r_μ_cc_indices, N_k_samples)
    error_Z_cc = BSE_k_ISDF.isdf_error(u_c, ζ_cc, r_μ_cc_indices)

    push!(errors_Z_cc, error_Z_cc)
end
@time for N_μ_vc in N_μ_vec
    r_μ_vc_indices = BSE_k_ISDF.find_r_μ(prob, N_μ_vc)
    ζ_vc = BSE_k_ISDF.assemble_ζ(u_v, u_c, r_μ_vc_indices)
    # error_Z_vc = BSE_k_ISDF.isdf_error_estimate(u_v, u_c, ζ_vc, r_μ_vc_indices, N_k_samples)
    error_Z_vc = BSE_k_ISDF.isdf_error(u_v, u_c, ζ_vc, r_μ_vc_indices)

    push!(errors_Z_vc, error_Z_vc)
end

save("1d_old/errors_Z.jld2", "N_μ_vec", N_μ_vec, "errors_Z_vv", errors_Z_vv, "errors_Z_cc", errors_Z_cc, "errors_Z_vc", errors_Z_vc)

# %% set up reference Hamiltonian

# fixed parameters
N_unit = 128

N_core = 0
N_v = 4
N_c = 5
N_k = 256

# set up problem
sp_prob = BSE_k_ISDF.SPProblem1D(V_sp, l, N_unit, N_k);
prob = BSE_k_ISDF.BSEProblem1D(sp_prob, N_core, N_v, N_c, V_func, W_func);

# compute reference Hamiltonian
t_V_entry = @benchmark BSE_k_ISDF.V_entry($prob.v_hat, 1, 1, 15, 1, 1, 60, $prob.u_v, $prob.u_c, $l, $(BSE_k_ISDF.size_r(prob)), $(BSE_k_ISDF.size_k(prob)))
w_hat, q_2bz_ind, q_2bz_shift = BSE_k_ISDF.compute_w_hat(prob);
ikkp2iq = BSE_k_ISDF.ikkp2iq_matrix(BSE_k_ISDF.k_lattice(prob), BSE_k_ISDF.q_lattice(prob));
t_W_entry = @benchmark BSE_k_ISDF.W_entry($w_hat, 1, 1, 15, 1, 1, 60, $prob.u_v, $prob.u_c, $l, $(BSE_k_ISDF.size_r(prob)), $ikkp2iq, $q_2bz_ind, $q_2bz_shift)

println("estimated time to assemble H_exact is ", (mean(t_V_entry).time + mean(t_W_entry).time) * (N_v * N_c * N_k)^2 * 1e-9, " seconds")
sleep(1) # to make sure the estimated time is printed

D = BSE_k_ISDF.setup_D(prob);
V_exact = BSE_k_ISDF.assemble_exact_V(prob)
W_exact = BSE_k_ISDF.assemble_exact_W(prob)
H_exact = D + 2 * V_exact - W_exact

@time F = eigen(H_exact)

save("1d_old/H_exact_$(N_unit)_$(N_v)_$(N_c)_$(N_k).jld2", "H_exact", H_exact, "ev", F.values, "ef", F.vectors)

# %% copute error in H for different errors in M (it might not be necessary to compute this)

N_unit = 128

N_core = 0
N_v = 4
N_c = 5
N_k = 256

H_exact = load("1d_old/H_exact_$(N_unit)_$(N_v)_$(N_c)_$(N_k).jld2", "H_exact")

N_μ_vec, errors_Z_vv, errors_Z_cc, errors_Z_vc = load("1d_old/errors_Z.jld2", "N_μ_vec", "errors_Z_vv", "errors_Z_cc", "errors_Z_vc")

N_μ_vv_vec = []
N_μ_cc_vec = []
N_μ_vc_vec = []
errors_H = []

Z_tol_vec = [0.8, 0.4, 0.2, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
for Z_tol in Z_tol_vec
    N_μ_vv = N_μ_vec[findfirst(errors_Z_vv .<= Z_tol)]
    N_μ_cc = N_μ_vec[findfirst(errors_Z_cc .<= Z_tol)]
    N_μ_vc = N_μ_vec[findfirst(errors_Z_vc .<= Z_tol)]

    isdf = BSE_k_ISDF.ISDF(prob, N_μ_vv, N_μ_cc, N_μ_vc)

    H = BSE_k_ISDF.setup_H(prob, isdf)

    @time H_full = Matrix(H)

    error_H = norm(H_full - H_exact) / norm(H_exact)

    push!(N_μ_vv_vec, N_μ_vv)
    push!(N_μ_cc_vec, N_μ_cc)
    push!(N_μ_vc_vec, N_μ_vc)
    push!(errors_H, error_H)
end

save("1d_old/errors_H_$(N_unit)_$(N_v)_$(N_c)_$(N_k).jld2", "Z_tol_vec", Z_tol_vec, "N_μ_vv_vec", N_μ_vv_vec, "N_μ_cc_vec", N_μ_cc_vec, "N_μ_vc_vec", N_μ_vc_vec, "errors_H", errors_H)

# %% compute reference absorption spectrum

# fixed parameters
N_unit = 128

N_core = 0
N_v = 4
N_c = 5
N_k = 256

# broadening
σ = 0.1
g = (ω, σ) -> 1 / π * σ / (ω^2 + σ^2)
# energy range
E_min, E_max = 0.0, 20.0
Erange = E_min:0.01:E_max
# parameter for lanczos
N_iter = 200

# set up problem
sp_prob = BSE_k_ISDF.SPProblem1D(V_sp, l, N_unit, N_k)
prob = BSE_k_ISDF.BSEProblem1D(sp_prob, N_core, N_v, N_c, V, W)

H_exact, ev, ef = load("results_" * example_string * "/H_exact_$(N_unit)_$(N_v)_$(N_c)_$(N_k).jld2", "H_exact", "ev", "ef")

optical_absorption = BSE_k_ISDF.optical_absorption(ev, ef, prob.u_v, prob.u_c, prob.E_v, prob.E_c, prob.prob.r_unit, prob.prob.k_bz, ω -> g(ω, σ), Erange)
d = BSE_k_ISDF.optical_absorption_vector(prob)
optical_absorption_lanc = 8 * π^2 / l * BSE_k_ISDF.lanczos_optical_absorption(H_exact, d, N_iter, ω -> g(ω, σ), Erange)

save("results_" * example_string * "/optical_absorption_ref_$(N_unit)_$(N_k).jld2", "Erange", Erange, "optical_absorption", optical_absorption)
save("results_" * example_string * "/optical_absorption_lanc_ref_$(N_unit)_$(N_k)_$(N_iter).jld2", "Erange", Erange, "optical_absorption_lanc", optical_absorption_lanc)

# %% compute absorption spectra for different k

# fixed parameters
N_unit = 128

N_core = 0
N_v = 4
N_c = 5

N_μ_cc = 22
N_μ_vv = 16
N_μ_vc = 21

# broadening
σ = 0.1
g = (ω, σ) -> 1 / π * σ / (ω^2 + σ^2)
# energy range
E_min, E_max = 0.0, 20.0
Erange = E_min:0.01:E_max
# parameter for lanczos
N_iter = 200

#variable parameters
N_k_vec = 2 .^(4:10)

for N_k in N_k_vec
    sp_prob = BSE_k_ISDF.SPProblem1D(V_sp, l, N_unit, N_k)
    prob = BSE_k_ISDF.BSEProblem1D(sp_prob, N_core, N_v, N_c, V, W)

    isdf = BSE_k_ISDF.ISDF(prob, N_μ_vv, N_μ_cc, N_μ_vc)

    H = BSE_k_ISDF.setup_H(prob, isdf)

    optical_absorption_lanc = BSE_k_ISDF.lanczos_optical_absorption(prob, isdf, N_iter, ω -> g(ω, σ), Erange)
    ev, ef = eigs(H, which=:SR, nev = 1, maxiter=1000)

    # save results
    save("results_" * example_string * "/optical_absorption_lanczos_$(N_unit)_$(N_k)_$(N_iter).jld2", "Erange", Erange, "optical_absorption_lanc", optical_absorption_lanc)

    save("results_" * example_string * "/eigs_$(N_unit)_$(N_k).jld2", "ev", ev, "ef", ef)
end

# %% compute absorption spectra for different N_μ

# fixed parameters
N_unit = 128

N_core = 0
N_v = 4
N_c = 5
N_k = 256

sp_prob = BSE_k_ISDF.SPProblem1D(V_sp, l, N_unit, N_k)
prob = BSE_k_ISDF.BSEProblem1D(sp_prob, N_core, N_v, N_c, V, W)

N_μ_cc_vec, N_μ_vv_vec, N_μ_vc_vec, errors_M_cc, errors_M_vv, errors_M_vc =  load("results_" * example_string * "/errors_M_$(N_unit)_$(N_k).jld2", "N_μ_cc_vec", "N_μ_vv_vec", "N_μ_vc_vec", "errors_M_cc", "errors_M_vv", "errors_M_vc")

# broadening
σ = 0.1
g = (ω, σ) -> 1 / π * σ / (ω^2 + σ^2)
# energy range
E_min, E_max = 0.0, 20.0
Erange = E_min:0.01:E_max
# parameter for lanczos
N_iter = 200

#variable parameters
M_tol_vec = [0.8, 0.5, 0.25, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

for M_tol in M_tol_vec
    N_μ_vv = findfirst(errors_M_vv .<= M_tol)
    N_μ_cc = findfirst(errors_M_cc .<= M_tol)
    N_μ_vc = findfirst(errors_M_vc .<= M_tol)

    isdf = BSE_k_ISDF.ISDF(prob, N_μ_vv, N_μ_cc, N_μ_vc)

    H = BSE_k_ISDF.setup_H(prob, isdf)

    optical_absorption_lanc = BSE_k_ISDF.lanczos_optical_absorption(prob, isdf, N_iter, ω -> g(ω, σ), Erange)

    ev, ef = eigs(H, which=:SR, nev = 1, maxiter=1000)

    # save results
    save("results_" * example_string * "/optical_absorption_lanczos_$(N_unit)_$(N_k)_$(N_iter)_$(M_tol).jld2", "Erange", Erange, "optical_absorption_lanc", optical_absorption_lanc)

    save("results_" * example_string * "/eigs_$(N_unit)_$(N_k)_$(M_tol).jld2", "ev", ev, "ef", ef)
end
