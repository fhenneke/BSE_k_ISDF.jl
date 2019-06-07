# plotting the results of the experiments

import JLD2
import FileIO: load
import BenchmarkTools
using PGFPlotsX, LinearAlgebra, Statistics, DelimitedFiles
using BSE_k_ISDF

push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepgfplotslibrary{colorbrewer}")

fig_theme = @pgf {
    "cycle list/Dark2-8",
    mark_options = "solid",
    line_width = "1.5pt",
    grid = "major"}

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
