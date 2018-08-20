# error analysis for ISDF in BSE

# %%
# loading packages
using BenchmarkTools, JLD2, FileIO, LinearAlgebra, FFTW, Statistics
using Plots, LaTeXStrings
pyplot()
theme(:dark)

BLAS.set_num_threads(1)
FFTW.set_num_threads(1)

push!(LOAD_PATH, "/home/oim3l/Work/Projects/Excitons/Code")
cd("/home/oim3l/Work/Projects/Excitons/Code/BSE_k_ISDF/experiments")

using Revise #remove after debugging
using BSE_k_ISDF

# general problem setup
l = 3.0 # changed from 1.5
V_sp = r -> -100 * exp(-(r - 0.3 * l)^2 / (2 * (0.05 * l)^2)) - 60 * exp(-(r - 0.6 * l)^2 / (2 * (0.05 * l)^2))
V = (r_1, r_2) -> 1 /  sqrt((r_1 - r_2)^2 + 0.01)
W = (r_1, r_2, l, L) -> 0.0625 * (3.0 + sin(2π / l * r_1)) * (3.0 + cos(4π / l * r_2)) *
    exp(-abs(BSE_k_ISDF.supercell_difference(r_1, r_2, L))^2 / (2 * (4 * l)^2)) * V(BSE_k_ISDF.supercell_difference(r_1, r_2, L), 0.)

# %% set parameters

# fixed parameters
N_unit = 128

N_core = 0
N_v = 4
N_c = 5
N_k = 128

# variable parameters
N_μ_vv_vec = 1:40#[10:30]
N_μ_cc_vec = 1:40#[10:30]
N_μ_vc_vec = 1:40#[10:30]

# %% error in M

# set up problem
sp_prob = BSE_k_ISDF.SPProblem(V_sp, l, N_unit, N_k)
prob = BSE_k_ISDF.BSEProblem(sp_prob, N_core, N_v, N_c, V, W)

u_v, u_c = prob.u_v, prob.u_c

M_vv, M_cc, M_vc = BSE_k_ISDF.assemble_M(prob)

errors_M_vv = []
errors_M_cc = []
errors_M_vc = []

@time for N_μ_vv in N_μ_vv_vec
    r_μ_vv_indices = BSE_k_ISDF.find_r_μ(N_unit, N_μ_vv)
    ζ_vv = BSE_k_ISDF.assemble_ζ(u_v, r_μ_vv_indices)
    error_M_vv = norm(M_vv - ζ_vv * M_vv[r_μ_vv_indices, :]) / norm(M_vv)

    push!(errors_M_vv, error_M_vv)
end
@time for N_μ_cc in N_μ_cc_vec
    r_μ_cc_indices = BSE_k_ISDF.find_r_μ(N_unit, N_μ_cc)
    ζ_cc = BSE_k_ISDF.assemble_ζ(u_c, r_μ_cc_indices)
    error_M_cc = norm(M_cc - ζ_cc * M_cc[r_μ_cc_indices, :]) / norm(M_cc)

    push!(errors_M_cc, error_M_cc)
end
@time for N_μ_vc in N_μ_vc_vec
    r_μ_vc_indices = BSE_k_ISDF.find_r_μ(N_unit, N_μ_vc)
    ζ_vc = BSE_k_ISDF.assemble_ζ(u_v, u_c, r_μ_vc_indices)
    error_M_vc = norm(M_vc - ζ_vc * M_vc[r_μ_vc_indices, :]) / norm(M_vc)

    push!(errors_M_vc, error_M_vc)
end

save("results/errors_M_$(N_unit)_$(N_k).jld2", "N_μ_cc_vec", N_μ_cc_vec, "N_μ_vv_vec", N_μ_vv_vec, "N_μ_vc_vec", N_μ_vc_vec, "errors_M_cc", errors_M_cc, "errors_M_vv", errors_M_vv, "errors_M_vc", errors_M_vc)

# %% error in H

# fixed parameters
N_unit = 128

N_core = 0
N_v = 4
N_c = 5
N_k = 128

sp_prob = BSE_k_ISDF.SPProblem(V_sp, l, N_unit, N_k)
prob = BSE_k_ISDF.BSEProblem(sp_prob, N_core, N_v, N_c, V, W)

t_H_entry = @benchmark BSE_k_ISDF.H_entry_fast($prob.v_hat, $prob.w_hat, 4, 1, 15, 3, 1, 60, $prob.E_v, $prob.E_c, $prob.u_v, $prob.u_c, $prob.prob.r_super, $prob.prob.r_unit, $prob.prob.k_bz)

println("estimated time to assemble H_exact is ", mean(t_H_entry).time * (N_v * N_c * N_k)^2 / 2 * 1e-9, " seconds")

# @time H_exact = BSE_k_ISDF.assemble_exact_H(prob)
# save("results/H_exact_$(N_unit)_$(N_v)_$(N_c)_$(N_k).jld2", "H_exact", H_exact)
H_exact = load("results/H_exact_$(N_unit)_$(N_v)_$(N_c)_$(N_k).jld2", "H_exact")

N_μ_vv_vec = []
N_μ_cc_vec = []
N_μ_vc_vec = []
errors_H = []

M_tol_vec = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
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

save("results/errors_H_$(N_unit)_$(N_k).jld2", "M_tol_vec", M_tol_vec, "N_μ_vv_vec", N_μ_vv_vec, "N_μ_cc_vec", N_μ_cc_vec, "N_μ_vc_vec", N_μ_vc_vec, "errors_H", errors_H)

# %% error in absorption spectrum

# %% compute reference absorption spectrum

# fixed parameters
N_unit = 128

N_core = 0
N_v = 4
N_c = 5
N_k = 128

σ = 0.1
g = (ω, σ) -> 1 / π * σ / (ω^2 + σ^2)
E_min, E_max = 2.0, 20.0
Erange = E_min:0.01:E_max
N_iter = 200

#

sp_prob = BSE_k_ISDF.SPProblem(V_sp, l, N_unit, N_k)
prob = BSE_k_ISDF.BSEProblem(sp_prob, N_core, N_v, N_c, V, W)

t_H_entry = @benchmark BSE_k_ISDF.H_entry_fast($prob.v_hat, $prob.w_hat, 1, 1, 1, 1, 1, 1, $prob.E_v, $prob.E_c, $prob.u_v, $prob.u_c, $prob.prob.r_super, $prob.prob.r_unit, $prob.prob.k_bz)

println("estimated time to assemble H_exact is ", mean(t_H_entry).time * (N_v * N_c * N_k)^2 / 2 * 1e-9, " seconds")

@time H_exact = BSE_k_ISDF.assemble_exact_H(prob)
@time F = eigen(H_exact)
ev, ef = F.values, F.vectors
save("results/H_exact_$(N_unit)_$(N_v)_$(N_c)_$(N_k).jld2", "H_exact", H_exact, "ev", ev, "ef", ef)
# H_exact, ev, ef = load("results/H_exact_$(N_unit)_$(N_v)_$(N_c)_$(N_k).jld2", "H_exact", "ev", "ef")

optical_absorption = BSE_k_ISDF.optical_absorption(ev, ef, prob.u_v, prob.u_c, prob.E_v, prob.E_c, prob.prob.r_unit, prob.prob.k_bz, ω -> g(ω, σ), Erange)

d = BSE_k_ISDF.optical_absorption_vector(prob)
optical_absorption_lanc = 8 * π^2 / l * BSE_k_ISDF.lanczos_optical_absorption(H_exact, d, N_iter, ω -> g(ω, σ), Erange)

save("results/optical_absorption_ref_$(N_unit)_$(N_k).jld2", "Erange", Erange, "optical_absorption", optical_absorption)
save("results/optical_absorption_lanc_ref_$(N_unit)_$(N_k)_$(N_iter).jld2", "Erange", Erange, "optical_absorption_lanc", optical_absorption_lanc)

# %% compute absorption spectra for different k

# fixed parameters
N_unit = 128

N_core = 0
N_v = 4
N_c = 5

N_μ_cc = 22
N_μ_vv = 16
N_μ_vc = 21

N_iter = 200
σ = 0.1
g = (ω, σ) -> 1 / π * σ / (ω^2 + σ^2)
E_min, E_max = 2.0, 20.0
Erange = E_min:0.01:E_max

#variable parameters
N_k_vec = 2 .^(4:10)

for N_k in N_k_vec
    sp_prob = BSE_k_ISDF.SPProblem(V_sp, l, N_unit, N_k)
    prob = BSE_k_ISDF.BSEProblem(sp_prob, N_core, N_v, N_c, V, W)

    isdf = BSE_k_ISDF.ISDF(prob, N_μ_vv, N_μ_cc, N_μ_vc)

    H = BSE_k_ISDF.setup_H(prob, isdf)

    optical_absorption_lanc = BSE_k_ISDF.lanczos_optical_absorption(prob, isdf, N_iter, ω -> g(ω, σ), Erange)

    # save results
    save("results/optical_absorption_lanczos_$(N_unit)_$(N_k)_$(N_iter).jld2", "Erange", Erange, "optical_absorption_lanc", optical_absorption_lanc)
end

# %% compute absorption spectra for different N_μ

# fixed parameters
N_unit = 128

N_core = 0
N_v = 4
N_c = 5
N_k = 128

sp_prob = BSE_k_ISDF.SPProblem(V_sp, l, N_unit, N_k)
prob = BSE_k_ISDF.BSEProblem(sp_prob, N_core, N_v, N_c, V, W)

N_μ_cc_vec, N_μ_vv_vec, N_μ_vc_vec, errors_M_cc, errors_M_vv, errors_M_vc =  load("results/errors_M_128_128.jld2", "N_μ_cc_vec", "N_μ_vv_vec", "N_μ_vc_vec", "errors_M_cc", "errors_M_vv", "errors_M_vc")

N_iter = 200
σ = 0.1
g = (ω, σ) -> 1 / π * σ / (ω^2 + σ^2)
E_min, E_max = 2.0, 20.0
Erange = E_min:0.01:E_max

#variable parameters
M_tol_vec = [0.8, 0.5, 0.25]#[1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

for M_tol in M_tol_vec
    N_μ_vv = findfirst(errors_M_vv .<= M_tol)
    N_μ_cc = findfirst(errors_M_cc .<= M_tol)
    N_μ_vc = findfirst(errors_M_vc .<= M_tol)

    isdf = BSE_k_ISDF.ISDF(prob, N_μ_vv, N_μ_cc, N_μ_vc)

    H = BSE_k_ISDF.setup_H(prob, isdf)

    optical_absorption_lanc = BSE_k_ISDF.lanczos_optical_absorption(prob, isdf, N_iter, ω -> g(ω, σ), Erange)

    # save results
    save("results/optical_absorption_lanczos_$(N_unit)_$(N_k)_$(N_iter)_$(M_tol).jld2", "Erange", Erange, "optical_absorption_lanc", optical_absorption_lanc)
end

# %% plotting

# %% plotting of error in M
N_μ_cc_vec, N_μ_vv_vec, N_μ_vc_vec, errors_M_cc, errors_M_vv, errors_M_vc =  load("results/errors_M_128_128.jld2", "N_μ_cc_vec", "N_μ_vv_vec", "N_μ_vc_vec", "errors_M_cc", "errors_M_vv", "errors_M_vc")

plot(title = "Error in " * L"M_{ij}", xlabel = L"N_\mu^{ij}", yscale = :log10, ylims = (1e-10, 1e0))
plot!(N_μ_cc_vec, errors_M_cc, m = :circ, label = L"M_{cc}")
plot!(N_μ_vv_vec, errors_M_vv, m = :square, label = L"M_{vv}")
plot!(N_μ_vc_vec, errors_M_vc, m = :diamond, label = L"M_{vc}")

savefig("results/errors_M.pdf")

# %% plotting of error in H
M_tol_vec, errors_H = load("results/errors_H_128_128.jld2", "M_tol_vec", "errors_H")

plot(title = "Error in " * L"H", xlabel = L"\epsilon_M", xscale = :log10, yscale = :log10, ylims = (1e-11, 1e-5))
plot!(M_tol_vec, errors_H, m = :circ, label = L"H")

savefig("results/errors_H.pdf")


# %% plot optical absorption spectrum

N_k_vec = 2 .^(4:10)
N_unit = 128
N_iter = 200

Erange = load("results/optical_absorption_lanczos_$(N_unit)_$(N_k_vec[1])_$(N_iter).jld2", "Erange")
optical_absorption_lanc = []
optical_absorption_lanc = [load("results/optical_absorption_lanczos_$(N_unit)_$(N_k)_$(N_iter).jld2", "optical_absorption_lanc") for N_k in N_k_vec]



optical_absorption = load("results/optical_absorption_ref_128_128_500.jld2", "optical_absorption_lanc")

errors_optical_absorption = []
for i in 1:length(N_k_vec)
    push!(errors_optical_absorption, norm(optical_absorption_lanc[i] - optical_absorption, 1) / norm(optical_absorption, 1))
end

errors_optical_absorption_M_tol = []
for i in 1:length(M_tol_vec)
    push!(errors_optical_absorption_M_tol, norm(optical_absorption_lanc_M_tol[i] - optical_absorption, 1) / norm(optical_absorption, 1))
end

p_errors_optical_absorption = plot(title = "error in optical absorption spectrum", xlabel = L"N_k", xlims = (N_k_vec[1], N_k_vec[end]), ylims = (5e-3, 1), xscale = :log10, yscale = :log10)
plot!(p_errors_optical_absorption, N_k_vec, errors_optical_absorption, label = "")

savefig("results/errors_optical_absorption.pdf")

plot_indices = [1, 3, 5]
p_optical_absorption = plot(title = "optical absorption spectrum", xlabel = "E")
plot!(p_optical_absorption, Erange, optical_absorption, lw = 3, label = "reference spectrum")
plot!(p_optical_absorption, Erange, optical_absorption_lanc[plot_indices], lw = 2, label = "approximate spectrum for " .* L"N_k = " .* string.(transpose(N_k_vec[plot_indices])))

savefig("results/optical_absorption_spectrum.pdf")

# %% plot error for different N_μ
N_k = 128
N_unit = 128
N_iter = 200
N_unit_ref = 128
N_iter_ref = 200

# M_tol_vec = load("results/errors_H_$(N_unit)_$(N_k).jld2", "M_tol_vec")
M_tol_vec = [0.8, 0.5, 0.25, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
optical_absorption_lanc_M_tol = [load("results/optical_absorption_lanczos_$(N_unit)_$(N_k)_$(N_iter)_$(M_tol).jld2", "optical_absorption_lanc") for M_tol in M_tol_vec]

Erange, optical_absorption_ref = load("results/optical_absorption_lanc_ref_$(N_unit_ref)_$(N_k)_$(N_iter_ref).jld2", "Erange", "optical_absorption_lanc")

plot_indices = [1, 2, 3]#, 4, 5]
p_optical_absorption_M_tol = plot(title = "optical absorption spectrum", xlabel = "E")
plot!(p_optical_absorption_M_tol, Erange, optical_absorption_ref, lw = 3, label = "reference spectrum")
plot!(p_optical_absorption_M_tol, Erange, optical_absorption_lanc_M_tol[plot_indices], lw = 2, label = "approximate spectrum for " .* L"M_{tol} = " .* string.(transpose(M_tol_vec[plot_indices])))

# savefig("results/optical_absorption_spectrum_M_tol.pdf")
