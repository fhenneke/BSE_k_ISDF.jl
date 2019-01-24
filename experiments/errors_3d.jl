# error analysis for ISDF in BSE

# %%
# loading packages
using BenchmarkTools, JLD2, FileIO, LinearAlgebra, FFTW, Statistics
# using Plots, LaTeXStrings
# pyplot()
# theme(:dark)

BLAS.set_num_threads(1)
FFTW.set_num_threads(1)

push!(LOAD_PATH, "/home/felix/Work/Research/Code")
cd("/home/felix/Work/Research/Code/BSE_k_ISDF/experiments")

using Revise #remove after debugging
using BSE_k_ISDF

# %% error in M for different N_μ

path = "/home/felix/Work/Research/Code/BSE_k_ISDF/experiments/diamond/131313_20/"

N_1d = 20 # TODO: read from file
N_rs = (N_1d, N_1d, N_1d)
N_core = 0
N_v = 4
N_c = 10
N_ks = (13, 13, 13) # TODO: read from file
@time prob = BSE_k_ISDF.BSEProblemExciting(N_core, N_v, N_c, N_ks, N_rs, path);

N_k_samples = 10

# variable parameters

N_μs_vec = [((2, 2, 2), 2 * 2^3), ((3, 3, 3), 2 * 3^3), ((4, 4, 4), 2 * 4^3), ((5, 5, 5), 2 * 5^3), ((6, 6, 6), 2 * 6^3)]

u_v, u_c = prob.u_v, prob.u_c

errors_M_vv = []
errors_M_cc = []
errors_M_vc = []

@time for N_μ_vvs in N_μs_vec
    r_μ_vv_indices = BSE_k_ISDF.find_r_μ(prob, N_μ_vvs[1], N_μ_vvs[2])
    ζ_vv = BSE_k_ISDF.assemble_ζ(u_v, r_μ_vv_indices)
    error_M_vv = BSE_k_ISDF.isdf_error_estimate(u_v, ζ_vv, r_μ_vv_indices, N_k_samples)

    push!(errors_M_vv, error_M_vv)
end
@time for N_μ_ccs in N_μs_vec
    r_μ_cc_indices = BSE_k_ISDF.find_r_μ(prob, N_μ_ccs[1], N_μ_ccs[2])
    ζ_cc = BSE_k_ISDF.assemble_ζ(u_c, r_μ_cc_indices)
    error_M_cc = BSE_k_ISDF.isdf_error_estimate(u_c, ζ_cc, r_μ_cc_indices, N_k_samples)

    push!(errors_M_cc, error_M_cc)
end
@time for N_μ_vcs in N_μs_vec
    r_μ_vc_indices = BSE_k_ISDF.find_r_μ(prob, N_μ_vcs[1], N_μ_vcs[2])
    ζ_vc = BSE_k_ISDF.assemble_ζ(u_v, u_c, r_μ_vc_indices)
    error_M_vc = BSE_k_ISDF.isdf_error_estimate(u_v, u_c, ζ_vc, r_μ_vc_indices, N_k_samples)

    push!(errors_M_vc, error_M_vc)
end

save(path * "/errors_M.jld2", "N_μs_vec", N_μs_vec, "errors_M_vv", errors_M_vv, "errors_M_cc", errors_M_cc, "errors_M_vc", errors_M_vc)

# %% set up reference Hamiltonian

# fixed parameters
N_unit = 128

N_core = 0
N_v = 4
N_c = 5
N_k = 256
# N_core = 3
# N_v = 1
# N_c = 1
# N_k = 4096

# set up problem
sp_prob = BSE_k_ISDF.SPProblem1D(V_sp, l, N_unit, N_k)
prob = BSE_k_ISDF.BSEProblem1D(sp_prob, N_core, N_v, N_c, V, W)

# compute reference Hamiltonian
t_H_entry = @benchmark BSE_k_ISDF.H_entry_fast($prob.v_hat, $prob.w_hat, 1, 1, 15, 1, 1, 60, $prob.E_v, $prob.E_c, $prob.u_v, $prob.u_c, $prob.prob.r_super, $prob.prob.r_unit, $prob.prob.k_bz)

println("estimated time to assemble H_exact is ", mean(t_H_entry).time * (N_v * N_c * N_k)^2 / 2 * 1e-9, " seconds")
sleep(1) # to make sure the estimated time is printed

@time H_exact = BSE_k_ISDF.assemble_exact_H(prob)
@time F = eigen(H_exact)
save("results_" * example_string * "/H_exact_$(N_unit)_$(N_v)_$(N_c)_$(N_k).jld2", "H_exact", H_exact, "ev", F.values, "ef", F.vectors)

# %% copute error in H for different errors in M (it might not be necessary to compute this)

N_unit = 128

N_core = 0
N_v = 4
N_c = 5
N_k = 128

H_exact = load("results_" * example_string * "/H_exact_$(N_unit)_$(N_v)_$(N_c)_$(N_k).jld2", "H_exact")

N_μ_vv_vec = []
N_μ_cc_vec = []
N_μ_vc_vec = []
errors_H = []

M_tol_vec = [0.8, 0.4, 0.2, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
for M_tol in M_tol_vec
    N_μ_vv = findfirst(errors_M_vv .<= M_tol)
    N_μ_cc = findfirst(errors_M_cc .<= M_tol)
    N_μ_vc = findfirst(errors_M_vc .<= M_tol)

    isdf = BSE_k_ISDF.ISDF(prob, N_μ_vv, N_μ_cc, N_μ_vc)

    H = BSE_k_ISDF.setup_H(prob, isdf)

    @time H_full = Matrix(H)

    error_H = norm(H_full - H_exact) / norm(H_exact)

    push!(N_μ_vv_vec, N_μ_vv)
    push!(N_μ_cc_vec, N_μ_cc)
    push!(N_μ_vc_vec, N_μ_vc)
    push!(errors_H, error_H)
end

save("results_" * example_string * "/errors_H_$(N_unit)_$(N_k).jld2", "M_tol_vec", M_tol_vec, "N_μ_vv_vec", N_μ_vv_vec, "N_μ_cc_vec", N_μ_cc_vec, "N_μ_vc_vec", N_μ_vc_vec, "errors_H", errors_H)

# %% compute reference absorption spectrum

# fixed parameters
N_unit = 128

N_core = 0
N_v = 4
N_c = 5
N_k = 256
# N_core = 3
# N_v = 1
# N_c = 1
# N_k = 4096

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
