# plotting the results of the experiments

import JLD2
import FileIO: load
import BenchmarkTools: Trial, Parameters
using PGFPlotsX, LinearAlgebra

push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepgfplotslibrary{colorbrewer}")

fig_theme = @pgf {
    "cycle list/Dark2-8",
    mark_options = "solid",
    line_width = "1.5pt",
    grid = "major"}

# %% plot potentials and solution

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

# %% plot error

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
