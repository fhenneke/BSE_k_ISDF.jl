# error analysis for ISDF in BSE

# %%
# loading packages
using BenchmarkTools, JLD2, FileIO, DelimitedFiles, LinearAlgebra, FFTW, Statistics, Arpack, HDF5
using BSE_k_ISDF

BLAS.set_num_threads(1)
FFTW.set_num_threads(1)

# %% set up problem
example_path = "diamond/131313_20/" # "graphene/"

N_1d = 20
N_rs = (N_1d, N_1d, N_1d) # (15, 15, 50) for graphene
N_core = 0
N_v = 4
N_c = 10 # 5 for graphene
N_ks = (13, 13, 13) # (42, 42, 1) for graphene
@time prob = BSE_k_ISDF.BSEProblemExciting(N_core, N_v, N_c, N_ks, N_rs, example_path);

# %% error for different N_μ
N_k_samples = 20

# variable parameters
N_μ_vec = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 180, 220, 260, 300, 350, 400, 450, 500, 600]

u_v, u_c = prob.u_v, prob.u_c

N_sub = 30
F_vv = BSE_k_ISDF.qrcp(u_v, N_sub);
F_cc = BSE_k_ISDF.qrcp(u_c, N_sub);
F_vc = BSE_k_ISDF.qrcp(u_v, u_c, N_sub^2);

# for plotting
rc = BSE_k_ISDF.lattice_matrix(prob) * (mod.(BSE_k_ISDF.r_lattice(prob) .+ 0.375, 1.0) .- 0.375)
r_μ_vv = rc[:, F_vv.p];
r_μ_cc = rc[:, F_cc.p];
r_μ_vc = rc[:, F_vc.p];
save(example_path * "/real_space_grid_$(N_sub).jld2", "rc", rc, "r_μ_vv", r_μ_vv, "r_μ_cc", r_μ_cc, "r_μ_vc", r_μ_vc)

# compute errors in isdf

errors_Z_vv = []
errors_Z_cc = []
errors_Z_vc = []

@time for N_μ_vv in N_μ_vec
    r_μ_vv_indices = F_vv.p[1:N_μ_vv]
    ζ_vv = BSE_k_ISDF.assemble_ζ(u_v, r_μ_vv_indices)
    error_Z_vv = BSE_k_ISDF.isdf_error_estimate(u_v, ζ_vv, r_μ_vv_indices, N_k_samples)
    # error_Z_vv = BSE_k_ISDF.isdf_error(u_v, ζ_vv, r_μ_vv_indices)

    push!(errors_Z_vv, error_Z_vv)
end
@time for N_μ_cc in N_μ_vec
    r_μ_cc_indices = F_cc.p[1:N_μ_cc]
    ζ_cc = BSE_k_ISDF.assemble_ζ(u_c, r_μ_cc_indices)
    error_Z_cc = BSE_k_ISDF.isdf_error_estimate(u_c, ζ_cc, r_μ_cc_indices, N_k_samples)
    # error_Z_cc = BSE_k_ISDF.isdf_error(u_c, ζ_cc, r_μ_cc_indices)

    push!(errors_Z_cc, error_Z_cc)
end
@time for N_μ_vc in N_μ_vec
    r_μ_vc_indices = F_vc.p[1:N_μ_vc]
    ζ_vc = BSE_k_ISDF.assemble_ζ(u_v, u_c, r_μ_vc_indices)
    error_Z_vc = BSE_k_ISDF.isdf_error_estimate(u_v, u_c, ζ_vc, r_μ_vc_indices, N_k_samples)
    # error_Z_vc = BSE_k_ISDF.isdf_error(u_v, u_c, ζ_vc, r_μ_vc_indices)

    push!(errors_Z_vc, error_Z_vc)
end

save(example_path * "/errors_Z.jld2", "N_μ_vec", N_μ_vec, "errors_Z_vv", errors_Z_vv, "errors_Z_cc", errors_Z_cc, "errors_Z_vc", errors_Z_vc)

# %% compute spectra for different N_μ

# load erros in M
N_μ_vec, errors_Z_vv, errors_Z_cc, errors_Z_vc =  load(example_path * "/errors_Z.jld2", "N_μ_vec", "errors_Z_vv", "errors_Z_cc", "errors_Z_vc")

# fixed parameters
N_iter = 200

#variable parameters
Z_tol_vec = [0.5, 0.2, 0.1, 0.05] # [0.5, 0.2, 0.1, 0.05, 0.02, 0.01] for graphene

# compute spectra
for Z_tol in Z_tol_vec
    N_μ_vv = N_μ_vec[findfirst(errors_Z_vv .<= Z_tol)]
    N_μ_cc = N_μ_vec[findfirst(errors_Z_cc .<= Z_tol)]
    N_μ_vc = N_μ_vec[findfirst(errors_Z_vc .<= Z_tol)]
#     @show N_μ_vv, N_μ_cc, N_μ_vc
# end

    isdf = BSE_k_ISDF.ISDF(prob, N_μ_vv, N_μ_cc, N_μ_vc)

    H = BSE_k_ISDF.setup_H(prob, isdf)

    for direction in 1:3
        d = BSE_k_ISDF.optical_absorption_vector(prob, direction)

        α, β, U = BSE_k_ISDF.lanczos(H, normalize(d), N_iter) # H + 0.01 * I for graphene

        # save results
        save(example_path * "/optical_absorption_lanczos_$(Z_tol)_$(direction).jld2", "alpha", α, "beta", β, "norm_d", norm(d))
    end

    eigenvalues, eigenvectors = eigs(H, which=:SR, nev = 1, maxiter=1000)

    save(example_path * "/eigs_$(Z_tol).jld2", "eigenvalues", eigenvalues, "eigenvectors", eigenvectors)
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

for N_iter in [50, 100, 150, 200]
    for Z_tol in Z_tol_vec
        absorption = zeros(length(Erange))
        for direction in 1:3 # 1:2 for graphene
            α, β, norm_d = load(example_path * "/optical_absorption_lanczos_$(Z_tol)_$(direction).jld2", "alpha", "beta", "norm_d")

            absorption .+= BSE_k_ISDF.lanczos_optical_absorption(α, β, N_iter, g, Erange, norm_d^2 * 8 * pi^2 / (size(prob)[3] * BSE_k_ISDF.Ω0_volume(prob)))
        end
        absorption .*= 1 / 3# 1 / 2 for graphene

        save(example_path * "/optical_absorption_$(Z_tol)_$(N_iter).jld2", "Erange", Erange, "absorption", absorption)
    end
end

# %% compute errors

f = h5open(example_path * "/exciton_coefficients.h5")
eigenvalue_reference = read(f["eigvec-singlet-TDA-BAR-full"]["0001"]["evals"])[1]
close(f)
Erange, absorption_reference = load(example_path * "/optical_absorption_reference.jld2", "Erange", "absorption")

N_iter = 150
Z_tol_vec = [0.5, 0.2, 0.1, 0.05] # [0.5, 0.2, 0.1, 0.05, 0.02, 0.01] for graphene

errors_optical_absorption = []
errors_ground_state_energy_abs = []
errors_ground_state_energy_rel = []
for Z_tol in Z_tol_vec
    absorption = load(example_path * "/optical_absorption_$(Z_tol)_$(N_iter).jld2", "absorption")

    error_optical_absorption = norm(absorption - absorption_reference[:, 1], 1) / norm(absorption_reference[:, 1], 1)
    push!(errors_optical_absorption, error_optical_absorption)

    eigenvalues = load(example_path * "/eigs_$(Z_tol).jld2", "eigenvalues")

    push!(errors_ground_state_energy_abs, abs(eigenvalues[1] - eigenvalue_reference))
    push!(errors_ground_state_energy_rel, abs(eigenvalues[1] - eigenvalue_reference) / eigenvalue_reference)
end

save(example_path * "/errors_spectrum.jld2", "Z_tol_vec", Z_tol_vec,  "errors_optical_absorption", errors_optical_absorption, "errors_ground_state_energy_abs", errors_ground_state_energy_abs, "errors_ground_state_energy_rel", errors_ground_state_energy_rel)
