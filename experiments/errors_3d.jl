# error analysis for ISDF in BSE

# %%
# loading packages
using BenchmarkTools, JLD2, FileIO, DelimitedFiles, LinearAlgebra, FFTW, Statistics, Arpack, HDF5
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

example_path = "diamond/131313_20/" # "graphene/"

N_1d = 20 # TODO: read from file
N_rs = (N_1d, N_1d, N_1d) # (15, 15, 50) for graphene
N_core = 0
N_v = 4
N_c = 10
N_ks = (13, 13, 13) # (42, 42, 1) for graphene # TODO: read from file
@time prob = BSE_k_ISDF.BSEProblemExciting(N_core, N_v, N_c, N_ks, N_rs, example_path);

# %% error for different N_μ

N_k_samples = 20 #TODO: maybe set a little higher

# variable parameters

# f = N_μ -> begin
#     N_1d = 1
#     while (N_1d + 1)^3 * 2 < N_μ
#         N_1d += 1
#     end
#     ((N_1d, N_1d, N_1d), N_μ - N_1d^3)
# end
# N_μs_vec = [f(N_μ) for N_μ in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 180, 220, 260, 300, 350, 400, 450, 500]]#, 600, 700, 800, 1000]] # only up to 350 for graphene

N_μs_vec = [((i, i, i), (j, j, j)) for j in 2:N_1d, i in 2:6 if i < j && j <= 2 * i]

N_μ_vec = [length(BSE_k_ISDF.find_r_μ(prob, N_μs[1], N_μs[2])) for N_μs in N_μs_vec]

u_v, u_c = prob.u_v, prob.u_c

errors_Z_vv = []
errors_Z_cc = []
errors_Z_vc = []

@time for N_μ_vvs in N_μs_vec
    r_μ_vv_indices = BSE_k_ISDF.find_r_μ(prob, N_μ_vvs[1], N_μ_vvs[2])
    ζ_vv = BSE_k_ISDF.assemble_ζ(u_v, r_μ_vv_indices)
    error_Z_vv = BSE_k_ISDF.isdf_error_estimate(u_v, ζ_vv, r_μ_vv_indices, N_k_samples)
    # error_Z_vv = BSE_k_ISDF.isdf_error(u_v, ζ_vv, r_μ_vv_indices)

    push!(errors_Z_vv, error_Z_vv)
end
@time for N_μ_ccs in N_μs_vec
    r_μ_cc_indices = BSE_k_ISDF.find_r_μ(prob, N_μ_ccs[1], N_μ_ccs[2])
    ζ_cc = BSE_k_ISDF.assemble_ζ(u_c, r_μ_cc_indices)
    error_Z_cc = BSE_k_ISDF.isdf_error_estimate(u_c, ζ_cc, r_μ_cc_indices, N_k_samples)
    # error_Z_cc = BSE_k_ISDF.isdf_error(u_c, ζ_cc, r_μ_cc_indices)

    push!(errors_Z_cc, error_Z_cc)
end
@time for N_μ_vcs in N_μs_vec
    r_μ_vc_indices = BSE_k_ISDF.find_r_μ(prob, N_μ_vcs[1], N_μ_vcs[2])
    ζ_vc = BSE_k_ISDF.assemble_ζ(u_v, u_c, r_μ_vc_indices)
    error_Z_vc = BSE_k_ISDF.isdf_error_estimate(u_v, u_c, ζ_vc, r_μ_vc_indices, N_k_samples)
    # error_Z_vc = BSE_k_ISDF.isdf_error(u_v, u_c, ζ_vc, r_μ_vc_indices)

    push!(errors_Z_vc, error_Z_vc)
end

save(example_path * "/errors_Z.jld2", "N_μs_vec", N_μs_vec, "N_μ_vec", N_μ_vec, "errors_Z_vv", errors_Z_vv, "errors_Z_cc", errors_Z_cc, "errors_Z_vc", errors_Z_vc)

# %% compute spectra for different N_μ

# load erros in M
N_μs_vec, errors_Z_vv, errors_Z_cc, errors_Z_vc =  load(example_path * "/errors_Z.jld2", "N_μs_vec", "errors_Z_vv", "errors_Z_cc", "errors_Z_vc")

# fixed parameters
N_iter = 200

#variable parameters
Z_tol_vec = [0.2]#[0.5, 0.25, 0.2]

# compute spectra
for Z_tol in Z_tol_vec
    N_μ_vv = N_μs_vec[findfirst(errors_Z_vv .<= Z_tol)]
    N_μ_cc = N_μs_vec[findfirst(errors_Z_cc .<= Z_tol)]
    N_μ_vc = N_μs_vec[findfirst(errors_Z_vc .<= Z_tol)]

    isdf = BSE_k_ISDF.ISDF(prob, N_μ_vv, N_μ_cc, N_μ_vc)

    H = BSE_k_ISDF.setup_H(prob, isdf)

    for direction in 1:3
        d = BSE_k_ISDF.optical_absorption_vector(prob, direction)

        α, β, U = BSE_k_ISDF.lanczos(H, normalize(d), N_iter) # H + 0.01 * I for graphene

        # save results
        save(example_path * "/optical_absorption_lanczos_$(Z_tol)_$(direction).jld2", "alpha", α, "beta", β, "norm_d", norm(d))
    end

    # eigenvalues, eigenvectors = eigs(H, which=:SR, nev = 1, maxiter=1000)

    # save(example_path * "/eigs_$(Z_tol).jld2", "eigenvalues", eigenvalues, "eigenvectors", eigenvectors)
end

# %% save spectra for different M_tol and N_iter

# load reference spectrum
Erange_reference = readdlm(example_path * "/EPSILON/EPSILON_BSE-singlet-TDA-BAR_SCR-full_OC11.OUT")[19:end, 1]
Ha_to_eV = 27.211396641308
Erange = Erange_reference ./ Ha_to_eV

absorption_reference = zeros(length(Erange_reference), 3)
absorption_reference[:, 1] = readdlm(example_path * "/EPSILON/EPSILON_BSE-singlet-TDA-BAR_SCR-full_OC11.OUT")[19:end, 3]
absorption_reference[:, 2] = readdlm(example_path * "/EPSILON/EPSILON_BSE-singlet-TDA-BAR_SCR-full_OC22.OUT")[19:end, 3]
absorption_reference[:, 3] = readdlm(example_path * "/EPSILON/EPSILON_BSE-singlet-TDA-BAR_SCR-full_OC33.OUT")[19:end, 3]

save(example_path * "/optical_absorption_reference.jld2", "Erange", Erange, "absorption", absorption_reference)

# compute spectra from α and β
σ = 0.0055 # 0.0036 for graphene
g = ω -> 1 / π * σ / (ω^2 + σ^2)

for N_iter in [50, 100, 200]
    for M_tol in M_tol_vec
        absorption = zeros(length(Erange))
        for direction in 1:3 # 1:2 for graphene
            α, β, norm_d = load(example_path * "/optical_absorption_lanczos_$(M_tol)_$(direction).jld2", "alpha", "beta", "norm_d")

            absorption .+= BSE_k_ISDF.lanczos_optical_absorption(α, β, N_iter, g, Erange, norm_d^2 * 8 * pi^2 / (size(prob)[3] * BSE_k_ISDF.Ω0_volume(prob)))
        end
        absorption .*= 1 / 3# 1 / 2 for graphene

        save(example_path * "/optical_absorption_$(M_tol)_$(N_iter).jld2", "Erange", Erange, "absorption", absorption)
    end
end

# %% compute errors

f = h5open(example_path * "/exciton_coefficients.h5")
eigenvalue_reference = read(f["eigvec-singlet-TDA-BAR-full"]["0001"]["evals"])[1]
close(f)
Erange, absorption_reference = load(example_path * "/optical_absorption_reference.jld2", "Erange", "absorption")

N_iter = 200

errors_optical_absorption = []
errors_ground_state_energy = []
for M_tol in M_tol_vec
    # save results
    absorption = load(example_path * "/optical_absorption_$(M_tol)_$(N_iter).jld2", "absorption")

    error_optical_absorption = norm(absorption - absorption_reference[:, 1], 1) / norm(absorption_reference[:, 1], 1)
    push!(errors_optical_absorption, error_optical_absorption)

    eigenvalues = load(example_path * "/eigs_$(M_tol).jld2", "eigenvalues")

    push!(errors_ground_state_energy, abs(eigenvalues[1] - eigenvalue_reference))
end

save(example_path * "/errors_spectrum.jld2", "M_tol_vec", M_tol_vec,  "errors_optical_absorption", errors_optical_absorption, "errors_ground_state_energy", errors_ground_state_energy)
