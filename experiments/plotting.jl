# plotting the results of the experiments

import JLD2
import FileIO: load
import BenchmarkTools: Trial, Parameters
using Plots, LaTeXStrings, LinearAlgebra

pyplot()

cd("/home/oim3l/Work/Projects/Excitons/Code/BSE_k_ISDF/experiments")

example_string = "ex2"

# %% plot potentials and solution
# TODO: save eigenfunctions to file

N_unit = 128
N_v = 4
N_c = 5
N_k = 128

E_v, E_c, k_bz = load("results_" * example_string * "/example_$(N_unit)_$(N_v)_$(N_c)_$(N_k).jld2", "E_v", "E_c", "k_bz")
ef = load("results_" * example_string * "/H_exact_$(N_unit)_$(N_v)_$(N_c)_$(N_k).jld2", "ef")

color_1 = RGBA{Float64}(0.0,0.6056031611752248,0.978680117569607,0.5)
color_2 = RGBA{Float64}(0.8888735002725198,0.43564919034818983,0.2781229361419438,0.5)


ef_vck = reshape(abs2.(ef[:, 1]), N_v, N_c, N_k)
marker_v = 20 .* sqrt.([sum(ef_vck, dims=2)[iv, 1, ik] for ik in 1:N_k, iv in 1:N_v])
marker_c = 20 .* sqrt.([sum(ef_vck, dims=1)[1, ic, ik] for ik in 1:N_k, ic in 1:N_c])

p_ef_full = plot(k_bz, [E_c[ic, ik] for ik in 1:N_k, ic in 1:N_c],
    lc = 1, m = :circle, ms = marker_c, mc = color_1,
    labels = ["conduction bands" "" "" "" ""], ylims = (-10, 100))
plot!(p_ef_full, k_bz, [E_v[iv, ik] for ik in 1:N_k, iv in 1:N_v],
    lc = 2, m = :circle, ms = marker_v, mc = color_2,
    labels = ["valence bands" "" "" ""])

savefig("results_" * example_string * "/band_structure.pdf")

# %% plot benchmark results

N_unit = 128
N_k_vec = 2 .^(4:12)

results = [load("results_" * example_string * "/benchmark_$(N_unit)_$(N_k).jld2") for N_k in N_k_vec]
timings = ["t_isdf", "t_H_setup", "t_H_x"]

setup_times = sum(1e-9 .* time.(minimum.([res[t] for res in results, t in timings[1:2]])); dims = 2)
evaluation_times = 1e-9 .* time.(minimum.([res[t] for res in results, t in timings[3:3]]))

plot(title = "run time scaling", xlabel = L"N_k", ylabel = "time [s]")
plot!(N_k_vec, setup_times, m = :circ, labels = "initial setup", xscale = :log2, yscale = :log10)
plot!(N_k_vec, evaluation_times, m = :square, labels = "matrix vector product")
# plot!(N_k_vec, (0.5 * 80 * 1e-6 * 20^2) * N_k_vec.^2, ls = :dash, labels = "entrywise assembly of Hamiltonian")
plot!(N_k_vec, 1e-4 * N_k_vec, ls = :dash, labels = L"O(N_k)")

savefig("results_" * example_string * "/timings_k.pdf")

# %% plot benchmark results for different N_μ

N_unit = 128
N_k = 256

M_tol_vec = [0.8, 0.5, 0.25, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

results = [load("results_" * example_string * "/benchmark_$(N_unit)_$(N_k)_$(M_tol).jld2") for M_tol in M_tol_vec]
timings = ["t_isdf", "t_H_setup", "t_H_x"]

setup_times = sum(1e-6 .* time.(minimum.([res[t] for res in results, t in timings[1:2]])); dims = 2)
evaluation_times = 1e-6 .* time.(minimum.([res[t] for res in results, t in timings[3:3]]))

plot(title = "run time scaling", xlabel = L"M_{tol}", ylabel = "time [ms]")
plot!(M_tol_vec, setup_times, m = :circ, labels = "initial setup", xscale = :log10, yscale = :log10)
plot!(M_tol_vec, evaluation_times, m = :square, labels = "matrix vector product")
# plot!(M_tol_vec, 1e1 * M_tol_vec .^ (-1 / 5), ls = :dash, labels = L"O(M_tol)")

savefig("results_" * example_string * "/timings_M_tol.pdf")

# %% plot error

# %% plotting of error in M
N_unit = 128
N_k = 256

N_μ_cc_vec, N_μ_vv_vec, N_μ_vc_vec, errors_M_cc, errors_M_vv, errors_M_vc =  load("results_" * example_string * "/errors_M_$(N_unit)_$(N_k).jld2", "N_μ_cc_vec", "N_μ_vv_vec", "N_μ_vc_vec", "errors_M_cc", "errors_M_vv", "errors_M_vc")

plot(title = "Error in ISDF", xlabel = L"N_\mu^{ij}", yscale = :log10, ylims = (1e-10, 1e0))
plot!(N_μ_cc_vec, errors_M_cc, m = :circ, label = L"M_{cc}")
plot!(N_μ_vv_vec, errors_M_vv, m = :square, label = L"M_{vv}")
plot!(N_μ_vc_vec, errors_M_vc, m = :diamond, label = L"M_{vc}")

savefig("results_" * example_string * "/errors_M.pdf")

# %% plotting of error in H
M_tol_vec, errors_H = load("results_" * example_string * "/errors_H_128_128.jld2", "M_tol_vec", "errors_H")

plot(title = "Error in " * L"H", xlabel = L"\epsilon_M", xscale = :log10, yscale = :log10, ylims = (1e-11, 1e0))
plot!(M_tol_vec, errors_H, m = :circ, label = L"H")

savefig("results_" * example_string * "/errors_H.pdf")

# %% plot optical absorption spectrum for different N_k

N_unit = 128
N_k_vec = 2 .^(4:10)
N_iter = 200
N_unit_ref = 128
N_k_ref = 4096
N_iter_ref = 200

optical_absorption_lanc = []
optical_absorption_lanc = [load("results_" * example_string * "/optical_absorption_lanczos_$(N_unit)_$(N_k)_$(N_iter).jld2", "optical_absorption_lanc") for N_k in N_k_vec]

Erange, optical_absorption_ref = load("results_" * example_string * "/optical_absorption_lanc_ref_$(N_unit_ref)_$(N_k_ref)_$(N_iter_ref).jld2", "Erange", "optical_absorption_lanc")

errors_optical_absorption = []
for i in 1:length(N_k_vec)
    push!(errors_optical_absorption, norm(optical_absorption_lanc[i] - optical_absorption_ref, 1) / norm(optical_absorption_ref, 1))
end

p_errors_optical_absorption_k = plot(title = "error in optical absorption spectrum", xlabel = L"N_k", xlims = (N_k_vec[1], N_k_vec[end]), ylims = (5e-3, 1), xscale = :log2, yscale = :log10)
plot!(p_errors_optical_absorption_k, N_k_vec, errors_optical_absorption, m = :circ, label = "")

savefig("results_" * example_string * "/errors_optical_absorption_k.pdf")

plot_indices = [1, 3, 5]
p_optical_absorption = plot(title = "optical absorption spectrum", xlabel = "E")
plot!(p_optical_absorption, Erange, optical_absorption_ref, lw = 3, label = "reference spectrum", xlims = (3, 10))
plot!(p_optical_absorption, Erange, optical_absorption_lanc[plot_indices], lw = 2, label = "approximate spectrum for " .* L"N_k = " .* string.(transpose(N_k_vec[plot_indices])))

savefig("results_" * example_string * "/optical_absorption_spectrum_k.pdf")

# %% error in the first eigenvalue and spectral function for different N_μ

N_unit = 128
N_k = 256
N_iter = 200

N_unit_ref = 128
N_v_ref = 4
N_c_ref = 5
N_k_ref = 256
N_iter_ref = 200

M_tol_vec = [0.8, 0.5, 0.25, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
evs = [load("results_" * example_string * "/eigs_$(N_unit)_$(N_k)_$(M_tol).jld2", "ev")[1] for M_tol in M_tol_vec]
optical_absorption_lanc = [load("results_" * example_string * "/optical_absorption_lanczos_$(N_unit)_$(N_k)_$(N_iter)_$(M_tol).jld2", "optical_absorption_lanc") for M_tol in M_tol_vec]

ev_ref = load("results_" * example_string * "/H_exact_$(N_unit_ref)_$(N_v_ref)_$(N_c_ref)_$(N_k_ref).jld2", "ev")[1]
Erange, optical_absorption_ref = load("results_" * example_string * "/optical_absorption_lanc_ref_$(N_unit_ref)_$(N_k_ref)_$(N_iter_ref).jld2", "Erange", "optical_absorption_lanc")

errors_optical_absorption = []
for i in 1:length(M_tol_vec)
    push!(errors_optical_absorption, norm(optical_absorption_lanc[i] - optical_absorption_ref, 1) / norm(optical_absorption_ref, 1))
end

p_errors_spectrum = plot(title = "Error in spectrum of H", xlabel = L"M_{tol}", xscale = :log10, yscale = :log10)
plot!(p_errors_spectrum, M_tol_vec, abs.(evs .- ev_ref), m = :circ, label = "error in first eigenvalue")
plot!(p_errors_spectrum, M_tol_vec, errors_optical_absorption, m = :square, label = "error in spectral function")

savefig("results_" * example_string * "/errors_spectrum.pdf")

# %% plot  absorption spctrum for different N_μ

N_unit = 128
N_k = 256
N_iter = 200
N_unit_ref = 128
N_k_ref = 256
N_iter_ref = 200

# M_tol_vec = load("results/errors_H_$(N_unit)_$(N_k).jld2", "M_tol_vec")
M_tol_vec = [0.8, 0.5, 0.25, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
optical_absorption_lanc = [load("results_" * example_string * "/optical_absorption_lanczos_$(N_unit)_$(N_k)_$(N_iter)_$(M_tol).jld2", "optical_absorption_lanc") for M_tol in M_tol_vec]

Erange, optical_absorption_ref = load("results_" * example_string * "/optical_absorption_lanc_ref_$(N_unit_ref)_$(N_k_ref)_$(N_iter_ref).jld2", "Erange", "optical_absorption_lanc")

plot_indices = [1, 3]#, 4, 5]
p_optical_absorption_M_tol = plot(title = "optical absorption spectrum", xlabel = "E", xlims = (4, 7))
plot!(p_optical_absorption_M_tol, Erange, optical_absorption_ref, lw = 4, label = "reference spectrum")
plot!(p_optical_absorption_M_tol, Erange, optical_absorption_lanc[plot_indices], lw = 2, label = "approximate spectrum for " .* L"M_{tol} = " .* string.(transpose(M_tol_vec[plot_indices])))

savefig("results_" * example_string * "/optical_absorption_spectrum.pdf")
