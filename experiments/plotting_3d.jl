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

# %% plotting of error in spectrum of H
example_path = "diamond/131313_20/"

M_tol_vec, errors_optical_absorption, errors_ground_state_energy = load(example_path * "/errors_spectrum.jld2", "M_tol_vec",  "errors_optical_absorption", "errors_ground_state_energy")

plot(title = "Error in " * L"H", xlabel = L"\epsilon_M", xscale = :log10, yscale = :log10, ylims = (1e-11, 1e0))
plot!(M_tol_vec, errors_H, m = :circ, label = L"H")

savefig("results_" * example_string * "/errors_H.pdf")

# %% plot optical absorption spectrum

example_path = "diamond/131313_20/" # "graphene/" for graphene

Ha_to_eV = 27.211396641308

Erange, absorption_reference = load(example_path * "/optical_absorption_reference.jld2", "Erange", "absorption")

M_tol = 0.2
N_iter = 200
absorption_isdf = load(example_path * "/optical_absorption_$(M_tol)_$(N_iter).jld2", "absorption")

p_optical_absorption = plot(title = "optical absorption spectrum", xlabel = "E [eV]", xlims = (0, 20), ylims = (0, 35)) #xlims = (1, 5), ylims = (0, 15) for graphene
plot!(p_optical_absorption, Ha_to_eV .* Erange, absorption_reference[:, 1], lw = 4, ls = :dash, label = "reference spectrum")
plot!(p_optical_absorption, Ha_to_eV .* Erange, absorption_isdf, lw = 2, label = "spectrum using ISDF")

savefig(example_path * "/optical_absorption_spectrum.pdf")

# %% plot optical absorption spectrum for different N_iter

example_path = "diamond/131313_20/"

Ha_to_eV = 27.211396641308

Erange, absorption_reference = load(example_path * "/optical_absorption_reference.jld2", "Erange", "absorption")

M_tol = 0.25
N_iter_vec = [200]
optical_absorption_lanc = []
for N_iter in N_iter_vec
    absorption = load(example_path * "/optical_absorption_$(M_tol)_$(N_iter).jld2", "absorption")

    push!(optical_absorption_lanc, absorption)
end

p_optical_absorption_N_k = plot(title = "optical absorption spectrum", xlabel = "E [eV]", xlims = (0, 20), ylims = (0, 35))
plot!(p_optical_absorption_N_k, Ha_to_eV .* Erange, absorption_reference[:, 1], lw = 4, ls = :dash, label = "reference spectrum")
plot!(p_optical_absorption_N_k, Ha_to_eV .* Erange, optical_absorption_lanc, lw = 2, label = "approximate spectrum for " .* L"N_{iter} = " .* string.(transpose(N_iter_vec)))

savefig(example_path * "/optical_absorption_spectrum_lanczos.pdf")
# %% plot optical absorption spctrum for different N_μ

example_path = "diamond/131313_20/"

Ha_to_eV = 27.211396641308

Erange, absorption_reference = load(example_path * "/optical_absorption_reference.jld2", "Erange", "absorption")

M_tol_vec = [0.5, 0.25]
optical_absorption_lanc = []
for M_tol in M_tol_vec
    absorption = load(example_path * "/optical_absorption_$(M_tol).jld2", "absorption")

    push!(optical_absorption_lanc, absorption)
end

p_optical_absorption_M_tol = plot(title = "optical absorption spectrum", xlabel = "E [eV]", xlims = (0, 20), ylims = (0, 35))# xlims = (1, 5), ylims = (0, 15)) for graphene
plot!(p_optical_absorption_M_tol, Ha_to_eV .* Erange, absorption_reference[:, 1], lw = 4, ls = :dash, label = "reference spectrum")
plot!(p_optical_absorption_M_tol, Ha_to_eV .* Erange, optical_absorption_lanc, lw = 2, label = "approximate spectrum for " .* L"M_{tol} = " .* string.(transpose(M_tol_vec)))

savefig(example_path * "/optical_absorption_spectrum_isdf.pdf")
