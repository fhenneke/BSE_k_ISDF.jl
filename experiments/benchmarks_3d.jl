# benchmarks for ISDF in BSE

# %%
# loading packages
using BenchmarkTools, JLD2, FileIO, LinearAlgebra, FFTW
using Plots, LaTeXStrings

BLAS.set_num_threads(1)
FFTW.set_num_threads(1)

push!(LOAD_PATH, "/home/felix/Work/Research/Code")
cd("/home/felix/Work/Research/Code/BSE_k_ISDF/experiments")

using Revise #remove after debugging
using BSE_k_ISDF

# %% benchmark different values of N_k

# fixed parameters
path = "/home/felix/Work/Research/Code/BSE_k_ISDF/experiments/diamond/"
N_1d = 20
N_rs = (N_1d, N_1d, N_1d)
N_core = 0
N_v = 4
N_c = 5
N_μ_irs = (3, 3, 3)
N_μ_mt = 2 * 3^3

#variable parameters
N_ks_vec = [(2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (7, 7, 7), (9, 9, 9)]
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 20
for N_ks in N_ks_vec
    prob = BSE_k_ISDF.BSEProblemExciting(N_core, N_v, N_c, N_ks, N_rs, path * "$(N_ks...)_$(N_1d)/")

    isdf = BSE_k_ISDF.ISDF(prob, N_μ_mt, N_μ_irs)
    t_isdf = @benchmark BSE_k_ISDF.ISDF(prob, N_μ_mt, N_μ_irs)

    H = BSE_k_ISDF.setup_H(prob, isdf)
    t_H_setup = @benchmark BSE_k_ISDF.setup_H($prob, $isdf)

    x = rand(N_v * N_c * prod(N_ks)) + im * rand(N_v * N_c * prod(N_ks))
    H * x
    t_H_x = @benchmark $H * $x

    save(path * "$(N_ks...)_$(N_1d)/benchmark_$(N_μ_irs...)_$(N_μ_mt).jld2", "t_isdf", t_isdf, "t_H_setup", t_H_setup, "t_H_x", t_H_x)
end
