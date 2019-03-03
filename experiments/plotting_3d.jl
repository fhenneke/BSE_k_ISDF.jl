# plotting the results of the experiments

import JLD2
import FileIO: load
import BenchmarkTools
using Plots, LaTeXStrings, LinearAlgebra, Statistics, DelimitedFiles

pyplot()

cd("/home/felix/Work/Research/Code/BSE_k_ISDF/experiments")
push!(LOAD_PATH, "/home/felix/Work/Research/Code")
using BSE_k_ISDF

# %% wave functions of exicon coefficients in 3D

using Makie

cm = RGBA.(RGB.(to_colormap(:viridis, 256)), range(0.0, stop = 1.0, length = 256))

psi =

colorclamp = min(maximum(abs.(psi)), maximum(abs.(psi_reduced)))
psi_abs = vec(abs.(psi))
markersize = psi_abs
c = clamp.(psi_abs, 0, colorclamp)
psi_abs_reduced = vec(abs.(psi_reduced))
markersize_reduced = psi_abs_reduced
c_reduced = clamp.(psi_abs_reduced, 0, colorclamp)

p1 = Makie.scatter(prob.r_cartesian[1, :], prob.r_cartesian[2, :], prob.r_cartesian[3, :],
    color = c, markersize = markersize, colormap = cm)
p2 = Makie.scatter(prob.r_cartesian[1, :], prob.r_cartesian[2, :], prob.r_cartesian[3, :],
    color = c_reduced, markersize = markersize_reduced, colormap = cm)

scene = AbstractPlotting.vbox(p1, p2)



# %% plot benchmark results for diamond

example_path = "diamond/"
N_μ_vvs = ((2, 2, 2), 42)
N_μ_ccs = ((3, 3, 3), 93)
N_μ_vcs = ((3, 3, 3), 53)

#variable parameters
N_ks_vec, setup_times, evaluation_times =  load(example_path * "/benchmark_$(N_μ_vvs)_$(N_μ_ccs)_$(N_μ_vcs).jld2", "N_ks_vec", "setup_times", "evaluation_times")

plot(title = "run time scaling", xlabel = L"N_k", ylabel = "time [s]")
plot!(prod.(N_ks_vec), setup_times, m = :circ, labels = "initial setup", xscale = :log10, yscale = :log10)
plot!(prod.(N_ks_vec), evaluation_times, m = :square, labels = "matrix vector product")
# plot!(N_k_vec, (0.5 * 80 * 1e-6 * 20^2) * N_k_vec.^2, ls = :dash, labels = "entrywise assembly of Hamiltonian")
plot!(prod.(N_ks_vec), 4e-3 * prod.(N_ks_vec), ls = :dash, labels = L"O(N_k)")

savefig(example_path * "/timings_k.pdf")

# %% plotting of error in M
example_path = "diamond/131313_20/"

N_μs_vec, errors_M_vv, errors_M_cc, errors_M_vc =  load(example_path * "/errors_M.jld2", "N_μs_vec", "errors_M_vv", "errors_M_cc", "errors_M_vc")

N_μ_vec = [prod(N_μs[1]) + N_μs[2] for N_μs in N_μs_vec]

plot(title = "Error in ISDF", xlabel = L"N_\mu^{ij}", yscale = :log10, ylims = (1e-4, 1e0))
plot!(N_μ_vec, errors_M_cc, m = :circ, label = L"M_{cc}")
plot!(N_μ_vec, errors_M_vv, m = :square, label = L"M_{vv}")
plot!(N_μ_vec, errors_M_vc, m = :diamond, label = L"M_{vc}")

savefig(example_path * "/errors_M.pdf")

# %% plotting of error in H
M_tol_vec, errors_H = load("results_" * example_string * "/errors_H_128_128.jld2", "M_tol_vec", "errors_H")

plot(title = "Error in " * L"H", xlabel = L"\epsilon_M", xscale = :log10, yscale = :log10, ylims = (1e-11, 1e0))
plot!(M_tol_vec, errors_H, m = :circ, label = L"H")

savefig("results_" * example_string * "/errors_H.pdf")

# %% plot optical absorption spectrum for different N_k

# TODO: do this?

# %% plot  absorption spctrum

example_path = "graphene/" # "diamond/131313_20/"
N_ks = (42, 42, 1) # (13, 13, 13)
Ω0_vol = 76.76103479594097

# load reference
Erange_exciting = readdlm(example_path * "/EPSILON/EPSILON_BSE-singlet-TDA-BAR_SCR-full_OC11.OUT")[19:end, 1]
absorption_exciting = zeros(length(Erange_exciting), 3)
absorption_exciting[:, 1] = readdlm(example_path * "/EPSILON/EPSILON_BSE-singlet-TDA-BAR_SCR-full_OC11.OUT")[19:end, 3]
absorption_exciting[:, 2] = readdlm(example_path * "/EPSILON/EPSILON_BSE-singlet-TDA-BAR_SCR-full_OC22.OUT")[19:end, 3]
absorption_exciting[:, 3] = readdlm(example_path * "/EPSILON/EPSILON_BSE-singlet-TDA-BAR_SCR-full_OC33.OUT")[19:end, 3]

N_iter = 200
direction = 1
σ = 0.0055
g = ω -> 1 / π * σ / (ω^2 + σ^2)
Ha_to_eV = 27.205144510369827
Erange = Erange_exciting ./ Ha_to_eV

M_tol_vec = [0.5, 0.25]
optical_absorption_lanc = []
for M_tol in M_tol_vec
    α, β, norm_d2 = load(example_path * "/optical_absorption_lanczos_$(M_tol).jld2", "alpha", "beta", "norm d^2")

    push!(optical_absorption_lanc, norm_d2 * 8 * pi^2 / (prod(N_ks) * Ω0_vol) * BSE_k_ISDF.lanczos_optical_absorption(α, β, N_iter, g, Erange))
end

p_optical_absorption_M_tol = plot(title = "optical absorption spectrum", xlabel = "E [eV]", xlims = (4, 16))
plot!(p_optical_absorption_M_tol, Erange_exciting, absorption_exciting[:, direction], lw = 4, label = "reference spectrum")
plot!(p_optical_absorption_M_tol, Erange_exciting, [optical_absorption_lanc[1] optical_absorption_lanc[2]], lw = 2, label = "approximate spectrum for " .* L"M_{tol} = " .* string.(transpose(M_tol_vec)))

savefig(example_path * "/optical_absorption_spectrum.pdf")

# %% plot absorption spctrum for different N_μ

example_path = "graphene/"

Ha_to_eV = 27.211396641308

Erange, absorption_reference = load(example_path * "/optical_absorption_reference.jld2", "Erange", "absorption")

M_tol_vec = [0.5, 0.25]
optical_absorption_lanc = []
for M_tol in M_tol_vec
    absorption = load(example_path * "/optical_absorption_$(M_tol).jld2", "absorption")

    push!(optical_absorption_lanc, absorption)
end

p_optical_absorption_M_tol = plot(title = "optical absorption spectrum", xlabel = "E [eV]", xlims = (1, 5), ylims = (0, 20))
plot!(p_optical_absorption_M_tol, Ha_to_eV .* Erange, absorption_reference[:, 1], lw = 4, ls = :dash, label = "reference spectrum")
plot!(p_optical_absorption_M_tol, Ha_to_eV .* Erange, optical_absorption_lanc, lw = 2, label = "approximate spectrum for " .* L"M_{tol} = " .* string.(transpose(M_tol_vec)))

# savefig(example_path * "/optical_absorption_spectrum.pdf")
