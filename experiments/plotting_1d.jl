# plotting the results of the experiments

import JLD2
import FileIO: load
import BenchmarkTools: Trial, Parameters
using PGFPlotsX, Plots, LaTeXStrings, LinearAlgebra

push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepgfplotslibrary{colorbrewer}")

fig_theme = @pgf {
    "cycle list/Dark2-8",
    mark_options = "solid",
    line_width = "1.5pt",
    grid = "major"}

pyplot()

cd("/home/felix/Work/Research/Code/BSE_k_ISDF/experiments")

# %% plot potentials and solution
# TODO: save eigenfunctions to file

N_unit = 128
N_v = 4
N_c = 5
N_k = 256

r_super, V_grid, W_grid = load("1d_old/example_$(N_unit)_$(N_v)_$(N_c)_$(N_k).jld2", "r_super", "V_grid", "W_grid")

plot_indices = findall(-7.5 .<= r_super .<= 7.5)

fig = @pgf TikzPicture(Axis(
    {fig_theme...,
    xlabel = "\$r\$", ylabel = "",
    xmin = -7.5, xmax = 7.5,
    ymin = 0, ymax = 12,
    legend_pos = "north west",
    "width = 0.4\\textwidth",
    "height = 0.3\\textwidth",
    scale_only_axis = true},
    PlotInc({"no markers"},
        Table(; x = r_super[plot_indices], y = V_grid[plot_indices])),
    LegendEntry("\$V\$"),
    PlotInc({"no markers"},
        Table(; x = r_super[plot_indices], y = W_grid[plot_indices])),
    LegendEntry("\$W\$"),
    PlotInc({"dashed", "no markers"},
        Table(; x = [0.0, 0.0], y = [0.0, 12.0])),
    LegendEntry("\$\\Omega\$"),
    "\\pgfplotsset{cycle list shift=-1}",
    PlotInc({"dashed", "no markers"},
        Table(; x = [1.5, 1.5], y = [0.0, 12.0]))
))

pgfsave("1d_old/1d_potentials.tex", fig, include_preamble = false)
pgfsave("1d_old/1d_potentials.pdf", fig, include_preamble = false)

# %% plot band structure

E_v, E_c, k_bz = load("1d_old/example_$(N_unit)_$(N_v)_$(N_c)_$(N_k).jld2", "E_v", "E_c", "k_bz")
# ev, ef = load("1d_old/H_exact_$(N_unit)_$(N_v)_$(N_c)_$(N_k).jld2", "ev", "ef")
ev, ef = load("1d_old/eigs_$(N_unit)_$(N_v)_$(N_c)_$(N_k)_$(1e-5).jld2", "ev", "ef")

k_1bz = mod.(k_bz .+ 0.5, 1.0) .- 0.5
p = sortperm(k_1bz)

i_gs = findmin(real.(ev))[2]
ef_vck = reshape(abs2.(ef[:, i_gs]), N_v, N_c, N_k)
marker_v = 10 .* sqrt.([sum(ef_vck, dims=2)[iv, 1, ik] for ik in p, iv in 1:N_v])
marker_c = 10 .* sqrt.([sum(ef_vck, dims=1)[1, ic, ik] for ik in p, ic in 1:N_c])

fig = @pgf TikzPicture(Axis(
    {fig_theme...,
    title = "", xlabel = "\$k\$", ylabel = "\$\\epsilon_k\$",
    ylabel_shift = "-3mm",
    xmin = -0.5, xmax = 0.5,
    ymin = -10, ymax = 110,
    legend_pos = "north west",
    "width = 0.4\\textwidth",
    "height = 0.3\\textwidth",
    scale_only_axis = true,
    "colormap/Dark2-8"},
    "\\pgfplotsset{cycle list shift=1}",
    PlotInc({mark = "*", "scatter",
        mark_options = {draw_opacity = 0.0, fill_opacity = 0.3},
        point_meta = "{\\thisrow{markercolor}}",
        "visualization depends on = \\thisrow{markersize} \\as \\perpointmarksize",
        "scatter/@pre marker code/.append style={/tikz/mark size=\\perpointmarksize}"},
        Table(; x = k_1bz[p], y = E_c[1, p], markercolor = [1; 8; 2 * ones(254)], markersize = marker_c[:, 1])),
    LegendEntry("conduction bands"),
    "\\pgfplotsset{cycle list shift=-1}",
    PlotInc({mark = "*", "scatter",
        mark_options = {draw_opacity = 0.0, fill_opacity = 0.3},
        point_meta = "{\\thisrow{markercolor}}",
        "visualization depends on = \\thisrow{markersize} \\as \\perpointmarksize",
        "scatter/@pre marker code/.append style={/tikz/mark size=\\perpointmarksize}"},
        Table(; x = k_1bz[p], y = E_v[4, p], markercolor = [1; 8; 1 * ones(254)], markersize = marker_v[:, 4])),
    LegendEntry("valence bands"),
    "\\pgfplotsset{cycle list shift=-2}",
    PlotInc({"no markers"},
        Table(; x = k_1bz[p], y = E_v[1, p])),
    "\\pgfplotsset{cycle list shift=-3}",
    PlotInc({"no markers"},
        Table(; x = k_1bz[p], y = E_v[2, p])),
    "\\pgfplotsset{cycle list shift=-4}",
    PlotInc({"no markers"},
        Table(; x = k_1bz[p], y = E_v[3, p])),
    "\\pgfplotsset{cycle list shift=-4}",
    PlotInc({"no markers"},
        Table(; x = k_1bz[p], y = E_c[2, p])),
    "\\pgfplotsset{cycle list shift=-5}",
    PlotInc({"no markers"},
        Table(; x = k_1bz[p], y = E_c[3, p])),
    "\\pgfplotsset{cycle list shift=-6}",
    PlotInc({"no markers"},
        Table(; x = k_1bz[p], y = E_c[4, p])),
    "\\pgfplotsset{cycle list shift=-7}",
    PlotInc({"no markers"},
        Table(; x = k_1bz[p], y = E_c[5, p]))
))

pgfsave("1d_old/1d_band_structure.tex", fig, include_preamble = false)
pgfsave("1d_old/1d_band_structure.pdf", fig, include_preamble = false)

#

# %% plot benchmark results

N_k_vec, setup_times, evaluation_times =  load("1d_old/benchmark.jld2", "N_k_vec", "setup_times", "evaluation_times")

fig = @pgf LogLogAxis(
    {fig_theme...,
    title = "Run Time Scaling", xlabel = "\$N_k\$", ylabel = "run time [s]",
    legend_style = "{at={(0.6,0.2)},anchor=west}",
    ymin = 1e-4,
    "width = 0.4\\textwidth",
    "height = 0.3\\textwidth",
    scale_only_axis = true},
    PlotInc({mark = "*"},
        Table(; x = N_k_vec, y = setup_times)),
    LegendEntry("initial setup"),
    PlotInc({mark = "square*"},
        Table(; x = N_k_vec, y = evaluation_times)),
    LegendEntry("matrix vector product"),
    PlotInc({"dashed"},
        Table(; x = N_k_vec, y = 1.5 * 1e-4 * N_k_vec)),
        LegendEntry("\$O(N_k)\$")
)

pgfsave("1d_old/1d_timings_k.tex", fig, include_preamble = false)
pgfsave("1d_old/1d_timings_k.pdf", fig, include_preamble = false)

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

#

N_μ_vec, errors_Z_vv, errors_Z_cc, errors_Z_vc =  load("1d_old/errors_Z.jld2", "N_μ_vec", "errors_Z_vv", "errors_Z_cc", "errors_Z_vc")

fig = @pgf Axis(
    {fig_theme...,
    title = "Error in ISDF", xlabel = "\$N_\\mu^{ij}\$",
    legend_entries = {"\$Z^{cc}\$", "\$Z^{vv}\$", "\$Z^{vc}\$"},
    xmin = N_μ_vec[1], xmax = N_μ_vec[end], ymin = 1e-10, ymax = 1e0,
    ymode = "log",
    "width = 0.4\\textwidth",
    "height = 0.3\\textwidth",
    scale_only_axis = true},
    PlotInc({"solid", mark = "*"},
        Table(; x = N_μ_vec, y = errors_Z_cc)),
    PlotInc({"dashed", mark = "cube*"},
        Table(; x = N_μ_vec, y = errors_Z_vv)),
    PlotInc({"dotted", mark = "triangle*"},
        Table(; x = N_μ_vec, y = errors_Z_vc))
)

pgfsave("1d_old/1d_errors_isdf.tex", fig, include_preamble = false)
pgfsave("1d_old/1d_errors_isdf.pdf", fig, include_preamble = false)

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

Z_tol_vec, errors_optical_absorption, errors_ground_state_energy = load("1d_old/errors_spectrum.jld2", "Z_tol_vec", "errors_optical_absorption", "errors_ground_state_energy")

fig = @pgf LogLogAxis(
    {fig_theme...,
    title = "Error in Spectrum of \$H\$", xlabel = "\$Z_{tol}\$",
    legend_entries = {"error in first eigenvalue", "error in spectral function"},
    xmin = Z_tol_vec[end], xmax = Z_tol_vec[1], ymin = 1e-10, ymax = 5e4,
    # ymode = "log",
    legend_pos = "north west",
    "width = 0.4\\textwidth",
    "height = 0.3\\textwidth",
    scale_only_axis = true},
    PlotInc({"solid", mark = "*"},
        Table(; x = Z_tol_vec, y = errors_optical_absorption)),
    PlotInc({"dashed", mark = "cube*"},
        Table(; x = Z_tol_vec, y = errors_ground_state_energy))
)

pgfsave("1d_old/1d_errors_spectrum.tex", fig, include_preamble = false)
pgfsave("1d_old/1d_errors_spectrum.pdf", fig, include_preamble = false)

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
