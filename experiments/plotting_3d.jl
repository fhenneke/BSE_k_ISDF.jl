# plotting the results of the experiments

import JLD2
import FileIO: load
import BenchmarkTools
using PGFPlotsX, Plots, LaTeXStrings, LinearAlgebra, Statistics, DelimitedFiles

push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepgfplotslibrary{colorbrewer}")

fig_theme = @pgf {
    "cycle list/Dark2-8",
    mark_options = "solid",
    line_width = "1.5pt",
    grid = "major"}

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

# %% plot interpolation points

# for diamond
r_μ_vv = load("diamond/131313_20/real_space_grid_30.jld2", "r_μ_vv")
N_μ_vv = 50

r_μ_cc = load("diamond/131313_20/real_space_grid_30.jld2", "r_μ_cc")
N_μ_cc = 50

fig = @pgf Axis(
        {fig_theme...,
        title = "interpolation points", xlabel = "\$x\$ in [au]", ylabel = "\$y\$ in [au]", zlabel = "\$z\$ in [au]",
        legend_pos = "north west",
        "width = 0.70\\textwidth"},
        Plot3Inc({mark = "*", "ball color=yellow!80!black", "only marks", "mark size = 4pt", opacity = 0.5},
            Table(; x = r_μ_vv[1, 1:N_μ_vv], y = r_μ_vv[2, 1:N_μ_vv], z = r_μ_vv[3, 1:N_μ_vv])),
        LegendEntry("valence"),
        Plot3Inc({mark = "*", "ball color=yellow!80!black", "only marks", "mark size = 4pt", opacity = 0.5},
            Table(; x = r_μ_cc[1, 1:N_μ_cc], y = r_μ_cc[2, 1:N_μ_cc], z = r_μ_cc[3, 1:N_μ_cc])),
        LegendEntry("conduction"),
        Plot3Inc({mark = "*", "ball color=yellow!80!black", "only marks", "mark size = 4pt"},
            Table(; x = [0.0, 1.68658], y = [0.0, 1.68658], z = [0.0, 1.68658])),
        LegendEntry("positions of nuclei"))

pgfsave("diamond_interpolation_points.tex", fig, include_preamble = false)

# for graphene
r_μ_vv = load("graphene/real_space_grid_30.jld2", "r_μ_vv")
N_μ_vv = 50

r_μ_cc = load("graphene/real_space_grid_30.jld2", "r_μ_cc")
N_μ_cc = 50

fig = @pgf Axis(
        {fig_theme...,
        title = "graphene", xlabel = "\$x\$ in [au]", ylabel = "\$y\$ in [au]", zlabel = "\$z\$ in [au]",
        legend_pos = "north west",
        view = "{25}{10}",
        zmin = -6, zmax = 6,
        "width = 0.70\\textwidth"},
        Plot3Inc({mark = "*", "ball color=yellow!80!black", "only marks", "mark size = 4pt", opacity = 0.5},
            Table(; x = r_μ_vv[1, 1:N_μ_vv], y = r_μ_vv[2, 1:N_μ_vv], z = r_μ_vv[3, 1:N_μ_vv])),
        LegendEntry("valence"),
        Plot3Inc({mark = "*", "ball color=yellow!80!black", "only marks", "mark size = 4pt", opacity = 0.5},
            Table(; x = r_μ_cc[1, 1:N_μ_cc], y = r_μ_cc[2, 1:N_μ_cc], z = r_μ_cc[3, 1:N_μ_cc])),
        LegendEntry("conduction"),
        Plot3Inc({mark = "*", "ball color=yellow!80!black", "only marks", "mark size = 4pt"},
            Table(; x = [0.0, 0.0], y = [0.0, 2.6849674], z = [0.0, 0.0])),
        LegendEntry("positions of nuclei"))

pgfsave("diamond_interpolation_points.tex", fig, include_preamble = false)
# %% plotting of error in spectrum of H

Z_tol_vec, errors_optical_absorption, errors_ground_state_energy = load("diamond/131313_20/errors_spectrum.jld2", "Z_tol_vec", "errors_optical_absorption", "errors_ground_state_energy")

fig = @pgf LogLogAxis(
    {fig_theme...,
    title = "Error in Spectrum of \$H\$", xlabel = "\$Z_{tol}\$",
    legend_entries = {"error in first eigenvalue", "error in spectral function"},
    xmin = Z_tol_vec[end], xmax = Z_tol_vec[1], ymin = 1e-4, ymax = 1e1,
    # ymode = "log",
    legend_pos = "north west",
    "width = 0.55\\textwidth"},
    PlotInc({"solid", mark = "*"},
        Table(; x = Z_tol_vec, y = errors_optical_absorption)),
    PlotInc({"dashed", mark = "cube*"},
        Table(; x = Z_tol_vec, y = errors_ground_state_energy))
)

pgfsave("1d_old/1d_errors_spectrum.tex", fig, include_preamble = false)
pgfsave("1d_old/1d_errors_spectrum.pdf", fig, include_preamble = false)

# %% plot optical absorption spectrum

#diamond
Ha_to_eV = 27.211396641308
Erange, absorption_reference = load("diamond/131313_20/optical_absorption_reference.jld2", "Erange", "absorption")

Z_tol = 0.1
N_iter = 200
absorption_isdf = load("diamond/131313_20/optical_absorption_$(Z_tol)_$(N_iter).jld2", "absorption")

fig = @pgf Axis(
    {fig_theme...,
    "no markers",
    title = "Optical Absorption of Diamond",
    xlabel = "\$\\omega\$ [eV]", ylabel = "\$\\varepsilon_2\$",
    xmin = 0, xmax = 25, ymin = 0, ymax = 38,
    "width = 0.4\\textwidth",
    "height = 0.3\\textwidth",
    scale_only_axis = true},
    PlotInc({"dashed", line_width="2.5pt"},
        Table(; x = Ha_to_eV .* Erange, y = absorption_reference[:, 1])),
    LegendEntry("Reference"),
    PlotInc({},
        Table(; x = Ha_to_eV .* Erange, y = absorption_isdf)),
    LegendEntry("ISDF"),
)

pgfsave("diamond/131313_20/diamond_optical_absorption_spectrum.tex", fig, include_preamble = false)
pgfsave("diamond/131313_20/diamond_optical_absorption_spectrum.pdf", fig, include_preamble = false)

# graphene
Ha_to_eV = 27.211396641308
Erange, absorption_reference = load("graphene/optical_absorption_reference.jld2", "Erange", "absorption")

Z_tol = 0.1
N_iter = 100
absorption_isdf = load("graphene/optical_absorption_$(Z_tol)_$(N_iter).jld2", "absorption")

fig = @pgf Axis(
    {fig_theme...,
    "no markers",
    title = "Optical Absorption of Graphene",
    xlabel = "\$\\omega\$ [eV]", ylabel = "\$\\varepsilon_2\$",
    xmin = 1, xmax = 5, ymin = 0, ymax = 15,
    "width = 0.4\\textwidth",
    "height = 0.3\\textwidth",
    scale_only_axis = true},
    PlotInc({"dashed", line_width="2.5pt"},
        Table(; x = Ha_to_eV .* Erange, y = clamp.(absorption_reference[:, 1], 0, 20))),
    LegendEntry("Reference"),
    PlotInc({},
        Table(; x = Ha_to_eV .* Erange, y = clamp.(absorption_isdf, 0, 20))),
    LegendEntry("ISDF"),
)

pgfsave("graphene/graphene_optical_absorption_spectrum.tex", fig, include_preamble = false)
pgfsave("graphene/graphene_optical_absorption_spectrum.pdf", fig, include_preamble = false)

# %% plot optical absorption spectrum for different N_iter

# example_path = "diamond/131313_20/"

# Ha_to_eV = 27.211396641308

# Erange, absorption_reference = load(example_path * "/optical_absorption_reference.jld2", "Erange", "absorption")

# Z_tol = 0.25
# N_iter_vec = [200]
# optical_absorption_lanc = []
# for N_iter in N_iter_vec
#     absorption = load(example_path * "/optical_absorption_$(Z_tol)_$(N_iter).jld2", "absorption")

#     push!(optical_absorption_lanc, absorption)
# end

# p_optical_absorption_N_k = plot(title = "optical absorption spectrum", xlabel = "E [eV]", xlims = (0, 20), ylims = (0, 35))
# plot!(p_optical_absorption_N_k, Ha_to_eV .* Erange, absorption_reference[:, 1], lw = 4, ls = :dash, label = "reference spectrum")
# plot!(p_optical_absorption_N_k, Ha_to_eV .* Erange, optical_absorption_lanc, lw = 2, label = "approximate spectrum for " .* L"N_{iter} = " .* string.(transpose(N_iter_vec)))

# savefig(example_path * "/optical_absorption_spectrum_lanczos.pdf")

# %% plot optical absorption spctrum for different N_μ

example_path = "diamond/131313_20/"

Ha_to_eV = 27.211396641308

Erange, absorption_reference = load(example_path * "/optical_absorption_reference.jld2", "Erange", "absorption")

N_iter = 200
Z_tol_vec = [0.5, 0.1]
optical_absorption_lanc = []
for Z_tol in Z_tol_vec
    absorption = load(example_path * "/optical_absorption_$(Z_tol)_$(N_iter).jld2", "absorption")

    push!(optical_absorption_lanc, absorption)
end

fig = @pgf Axis(
    {fig_theme...,
    title = "Optical Absorption of Diamond",
    xlabel = "\$\\omega\$ [eV]", ylabel = "\$\\varepsilon_2\$",
    xmin = 0, xmax = 25, ymin = 0, ymax = 38,
    legend_pos = "north east",
    "width = 0.4\\textwidth",
    "height = 0.3\\textwidth",
    scale_only_axis = true},
    PlotInc({"dashed", line_width="3pt"},
        Table(; x = Ha_to_eV .* Erange, y = absorption_reference[:, 1])),
    LegendEntry("Reference"),
    PlotInc({},
        Table(; x = Ha_to_eV .* Erange, y = optical_absorption_lanc[1])),
    LegendEntry("\$Z_{tol} = 0.5\$"),
    PlotInc({},
        Table(; x = Ha_to_eV .* Erange, y = optical_absorption_lanc[2])),
    LegendEntry("\$Z_{tol} = 0.1\$")
)

pgfsave("diamond/131313_20/diamond_optical_absorption_spectrum_isdf.tex", fig, include_preamble = false)
pgfsave("diamond/131313_20/diamond_optical_absorption_spectrum_isdf.pdf", fig, include_preamble = false)

# %% plotting of error in Z

N_μ_vec, errors_Z_vv, errors_Z_cc, errors_Z_vc =  load("diamond/131313_20/errors_Z.jld2", "N_μ_vec", "errors_Z_vv", "errors_Z_cc", "errors_Z_vc")

fig = @pgf Axis(
    {fig_theme...,
    title = "Error in ISDF", xlabel = "\$N_\\mu^{ij}\$",
    legend_entries = {"\$Z^{cc}\$", "\$Z^{vv}\$", "\$Z^{vc}\$"},
    xmin = N_μ_vec[1], xmax = N_μ_vec[end], ymin = 1e-4, ymax = 1e0,
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

pgfsave("diamond/131313_20/diamond_errors_isdf.tex", fig, include_preamble = false)
pgfsave("diamond/131313_20/diamond_errors_isdf.pdf", fig, include_preamble = false)

# %% plot benchmark results for diamond

N_μ_vv = 70
N_μ_cc = 220
N_μ_vc = 100

N_ks_vec, setup_times, evaluation_times =  load("diamond/benchmark_$(N_μ_vv)_$(N_μ_cc)_$(N_μ_vc).jld2", "N_ks_vec", "setup_times", "evaluation_times")

fig = @pgf LogLogAxis(
    {fig_theme...,
    title = "Run Time Scaling", xlabel = "\$N_k\$", ylabel = "run time [s]",
    legend_pos = "south east",
    legend_style = "{at={(0.6,0.2)},anchor=west}",
    "width = 0.4\\textwidth",
    "height = 0.3\\textwidth",
    scale_only_axis = true},
    PlotInc({mark = "*"},
        Table(; x = prod.(N_ks_vec), y = setup_times)),
    LegendEntry("initial setup"),
    PlotInc({mark = "square*"},
        Table(; x = prod.(N_ks_vec), y = evaluation_times)),
    LegendEntry("matrix vector product"),
    PlotInc({"dashed"},
        Table(; x = prod.(N_ks_vec), y = 2e-2 * prod.(N_ks_vec))),
    LegendEntry("\$O(N_k)\$")
)

pgfsave("diamond/diamond_timings_k.tex", fig, include_preamble = false)
pgfsave("diamond/diamond_timings_k.pdf", fig, include_preamble = false)
