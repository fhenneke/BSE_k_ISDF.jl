# plotting the results of the experiments

import JLD2
import FileIO: load
import BenchmarkTools
using PGFPlotsX, Plots, LaTeXStrings, LinearAlgebra, Statistics, DelimitedFiles

push!(PGFPlotsX.CUSTOM_PREAMBLE, "\\usepgfplotslibrary{colorbrewer}")

# pyplot()
pgfplots()

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

plot(size = (200, 150), title = "run time scaling", xlabel = L"N_k", ylabel = "time [s]")
plot!(prod.(N_ks_vec), setup_times, m = :circ, labels = "initial setup", xscale = :log10, yscale = :log10)
plot!(prod.(N_ks_vec), evaluation_times, m = :square, labels = "matrix vector product")
# plot!(N_k_vec, (0.5 * 80 * 1e-6 * 20^2) * N_k_vec.^2, ls = :dash, labels = "entrywise assembly of Hamiltonian")
plot!(prod.(N_ks_vec), 4e-3 * prod.(N_ks_vec), ls = :dash, labels = L"O(N_k)")

savefig(example_path * "/timings_k.tex")

# same plot with PGFPlotsX

example_path = "diamond/"

N_μ_vvs = ((2, 2, 2), (4, 4, 4))
N_μ_ccs = ((4, 4, 4), (7, 7, 7))
N_μ_vcs = ((4, 4, 4), (5, 5, 5))

N_ks_vec, setup_times, evaluation_times =  load(example_path * "/benchmark_$(N_μ_vvs)_$(N_μ_ccs)_$(N_μ_vcs).jld2", "N_ks_vec", "setup_times", "evaluation_times")

fig_theme = @pgf {
    "cycle list/Dark2-8",
    mark_options = "solid",
    line_width = "1.5pt",
    grid = "major"}

fig = @pgf LogLogAxis(
        {fig_theme...,
        title = "run time scaling", xlabel = "\$N_k\$", ylabel = "time [s]",
        legend_pos = "south east",
        "width = 0.98\\textwidth"},
        PlotInc({mark = "*"},
            Table(; x = prod.(N_ks_vec), y = setup_times)),
        LegendEntry("initial setup"),
        PlotInc({mark = "square*"},
            Table(; x = prod.(N_ks_vec), y = evaluation_times)),
        LegendEntry("matrix vector product"),
        PlotInc({"dashed"},
            Table(; x = prod.(N_ks_vec), y = 3e-3 * prod.(N_ks_vec))),
        LegendEntry("\$O(N_k)\$"))

pgfsave("diamond_timings_k.tex", fig, include_preamble = false)

# %% plotting of error in M
example_path = "diamond/131313_20"

N_μ_vec, errors_Z_vv, errors_Z_cc, errors_Z_vc =  load(example_path * "/errors_Z.jld2", "N_μ_vec", "errors_Z_vv", "errors_Z_cc", "errors_Z_vc")

# N_μ_vec = [prod(N_μs[1]) + N_μs[2] for N_μs in N_μs_vec]

plot(title = "Error in ISDF", xlabel = L"N_\mu^{ij}", xscale = :log10, yscale = :log10, xlims = (N_μ_vec[1], N_μ_vec[end]), ylims = (1e-5, 2e0))
plot!(N_μ_vec, errors_Z_cc, m = :circ, label = L"Z^{cc}")
plot!(N_μ_vec, errors_Z_vv, m = :square, label = L"Z^{vv}")
plot!(N_μ_vec, errors_Z_vc, m = :diamond, label = L"Z^{vc}")

savefig(example_path * "/errors_M.pdf")

# same lot with PGFPlotsX
example_path = "diamond/131313_20"

N_μ_vec, errors_Z_vv, errors_Z_cc, errors_Z_vc =  load(example_path * "/errors_Z.jld2", "N_μ_vec", "errors_Z_vv", "errors_Z_cc", "errors_Z_vc")

fig = @pgf Axis(
        {fig_theme...,
        title = "Error in ISDF", xlabel = "\$N_\\mu^{ij}\$", ylabel = "time [s]",
        legend_entries = {"\$Z^{cc}\$", "\$Z^{vv}\$", "\$Z^{vc}\$"},
        xmin = N_μ_vec[1], xmax = N_μ_vec[end], ymin = 3e-4, ymax = 1e0,
        ymode = "log",
        # legend_pos = "south east",
        "width = 0.98\\textwidth"},
        PlotInc({"solid", mark = "*"},
            Table(; x = N_μ_vec, y = errors_Z_cc)),
        PlotInc({"dashed", mark = "cube*"},
            Table(; x = N_μ_vec, y = errors_Z_vv)),
        PlotInc({"dotted", mark = "triangle*"},
            Table(; x = N_μ_vec, y = errors_Z_vc)))

pgfsave("diamond/131313_20/errors_isdf.tex", fig, include_preamble = false)

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

# same plot with PGFPlotsX
#diamond
Ha_to_eV = 27.211396641308
Erange, absorption_reference = load("diamond/131313_20/optical_absorption_reference.jld2", "Erange", "absorption")

M_tol = 0.2
N_iter = 200
absorption_isdf = load("diamond/131313_20/optical_absorption_$(M_tol)_$(N_iter).jld2", "absorption")

fig = @pgf Axis(
    {theme...,
    "no markers",
    title = "Optical Absorption of Diamond",
    xlabel = "E [eV]", ylabel = "",
    xmin = 0, xmax = 20, ymin = 0, ymax = 35,
    legend_pos = "north east",
    "width = 0.55\\textwidth"},
    PlotInc({"dashed", line_width="3pt"},
        Table(; x = Ha_to_eV .* Erange, y = absorption_reference[:, 1])),
    LegendEntry("Reference"),
    PlotInc({},
        Table(; x = Ha_to_eV .* Erange, y = absorption_isdf)),
    LegendEntry("ISDF"),
)

pgfsave("diamond/131313_20/diamond_optical_absorption_spectrum.tex", fig, include_preamble = false)

# graphene
Ha_to_eV = 27.211396641308
Erange, absorption_reference = load("graphene/optical_absorption_reference.jld2", "Erange", "absorption")

M_tol = 0.2
N_iter = 200
absorption_isdf = load("graphene/optical_absorption_$(M_tol)_$(N_iter).jld2", "absorption")

fig = @pgf Axis(
    {theme...,
    "no markers",
    title = "Optical Absorption of Graphene",
    xlabel = "E [eV]", ylabel = "",
    xmin = 1, xmax = 5, ymin = 0, ymax = 15,
    # legend_pos = "north east",
    "width = 0.55\\textwidth"},
    PlotInc({"dashed", line_width="3pt"},
        Table(; x = Ha_to_eV .* Erange, y = clamp.(absorption_reference[:, 1], 0, 20))),
    LegendEntry("Reference"),
    PlotInc({},
        Table(; x = Ha_to_eV .* Erange, y = clamp.(absorption_isdf, 0, 20))),
    LegendEntry("ISDF"),
)

pgfsave("graphene/graphene_optical_absorption_spectrum.tex", fig, include_preamble = false)

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

# same plot with PGFPlotsX

fig = @pgf Axis(
    {theme...,
    title = "Optical Absorption of Diamond",
    xlabel = "E [eV]", ylabel = "",
    xmin = 0, xmax = 20, ymin = 0, ymax = 38,
    legend_pos = "north east",
    "width = 0.95\\textwidth"},
    PlotInc({"dashed", line_width="3pt"},
        Table(; x = Ha_to_eV .* Erange, y = absorption_reference[:, 1])),
    LegendEntry("Reference Spectrum"),
    PlotInc({},
        Table(; x = Ha_to_eV .* Erange, y = optical_absorption_lanc[1])),
    LegendEntry("Approximate Spectrum for \$Z_{tol} = 0.5\$"),
    PlotInc({},
        Table(; x = Ha_to_eV .* Erange, y = optical_absorption_lanc[2])),
    LegendEntry("Approximate Spectrum for \$Z_{tol} = 0.25\$")
)

pgfsave("diamond/131313_20/diamond_optical_absorption_spectrum_isdf.tex", fig, include_preamble = false)
