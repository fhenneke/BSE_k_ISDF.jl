# error analysis for ISDF in BSE

# %%
# loading packages
using BenchmarkTools, JLD2, FileIO, DelimitedFiles, LinearAlgebra, FFTW, Statistics
# using Plots, LaTeXStrings
# pyplot()
# theme(:dark)

BLAS.set_num_threads(1)
FFTW.set_num_threads(1)

push!(LOAD_PATH, "/home/felix/Work/Research/Code")
cd("/home/felix/Work/Research/Code/BSE_k_ISDF/experiments")

using Revise #remove after debugging
using BSE_k_ISDF

# %% set up problem

path = "/home/felix/Work/Research/Code/BSE_k_ISDF/experiments/diamond/131313_20/"

N_1d = 20 # TODO: read from file
N_rs = (N_1d, N_1d, N_1d)
N_core = 0
N_v = 4
N_c = 5 # maybe also use 5 here
N_ks = (13, 13, 13) # TODO: read from file
@time prob = BSE_k_ISDF.BSEProblemExciting(N_core, N_v, N_c, N_ks, N_rs, path);

# %% error for different N_μ

N_k_samples = 20 #TODO: maybe set a little higher

# variable parameters

f = N_μ -> begin
    N_1d = 1
    while (N_1d + 1)^3 * 2 < N_μ
        N_1d += 1
    end
    ((N_1d, N_1d, N_1d), N_μ - N_1d^3)
end

N_μs_vec = [f(N_μ) for N_μ in [10, 20, 40, 60, 80, 100, 120, 140, 180, 220, 260, 300, 350, 400, 450, 500, 600, 700, 800, 1000]]

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

# %% compute spectra for different N_μ

# load erros in M
N_μs_vec, errors_M_vv, errors_M_cc, errors_M_vc =  load(path * "/errors_M.jld2", "N_μs_vec", "errors_M_vv", "errors_M_cc", "errors_M_vc")

ev = vec([prob.E_c[ic, ik] - prob.E_v[iv, ik] for iv in 1:N_v, ic in 1:N_c, ik in 1:(prod(N_ks))]);
pmat = BSE_k_ISDF.read_pmat(N_core, N_v, N_c, prod(N_ks), path)

# fixed parameters
direction = 1
N_iter = 200

#variable parameters
M_tol_vec = [0.5, 0.25, 0.1]

# compute spectra
for M_tol in M_tol_vec
    N_μ_vv = N_μs_vec[findfirst(errors_M_vv .<= M_tol)]
    N_μ_cc = N_μs_vec[findfirst(errors_M_cc .<= M_tol)]
    N_μ_vc = N_μs_vec[findfirst(errors_M_vc .<= M_tol)]

    isdf = BSE_k_ISDF.ISDF(prob, N_μ_vv, N_μ_cc, N_μ_vc)

    # if M_tol == 0.1
    #     prob.u_v = 0
    #     prob.u_c = 0
    #     GC.gc()
    # end

    H = BSE_k_ISDF.setup_H(prob, isdf)

    d = pmat[:, direction] ./ ev

    α, β, U = BSE_k_ISDF.lanczos(H, normalize(d), I, N_iter)

    # eigenvalues, eigenvectors = eigs(H, which=:SR, nev = 1, maxiter=1000)

    # save results
    save(path * "/optical_absorption_lanczos_$(M_tol).jld2", "alpha", α, "beta", β, "norm d^2", norm(d)^2)

    # save(path * "/eigs_$(M_tol).jld2", "eigenvalues", eigenvalues, "eigenvectors", eigenvectors)
end

# %% compute errors

# load reference spectrum
Erange_exciting = readdlm(path * "/EPSILON/EPSILON_BSE-singlet-TDA-BAR_SCR-full_OC11.OUT")[19:end, 1]
absorption_exciting = zeros(length(Erange_exciting), 3)
absorption_exciting[:, 1] = readdlm(path * "/EPSILON/EPSILON_BSE-singlet-TDA-BAR_SCR-full_OC11.OUT")[19:end, 3]
absorption_exciting[:, 2] = readdlm(path * "/EPSILON/EPSILON_BSE-singlet-TDA-BAR_SCR-full_OC22.OUT")[19:end, 3]
absorption_exciting[:, 3] = readdlm(path * "/EPSILON/EPSILON_BSE-singlet-TDA-BAR_SCR-full_OC33.OUT")[19:end, 3];

E0 = 0.

# some parameters
σ = 0.0055
g = ω -> 1 / π * σ / (ω^2 + σ^2)
Ha_to_eV = 27.205144510369827
Erange = Erange_exciting ./ Ha_to_eV

errors_optical_absorption = []
errors_ground_state_energy = []
for M_tol in M_tol_vec
    # save results
    α, β, norm_d2 = load(path * "/optical_absorption_lanczos_$(M_tol).jld2", "alpha", "beta", "norm d^2")

    optical_absorption_lanc = norm_d2 * 8 * pi^2 / (prod(prob.N_ks) * prob.Ω0_vol) * BSE_k_ISDF.lanczos_optical_absorption(α, β, N_iter, g, Erange)

    error_optical_absorption = norm(optical_absorption_lanc - absorption_exciting[:, direction], 1) * Ha_to_eV / length(optical_absorption_lanc)
    push!(errors_optical_absorption, error_optical_absorption)

    # eigenvalues, eigenvectors = eigs(H, which=:SR, nev = 1, maxiter=1000)

end

save(path * "/errors_spectrum.jld2", "M_tol_vec", M_tol_vec,  "errors_optical_absorption", errors_optical_absorption)
