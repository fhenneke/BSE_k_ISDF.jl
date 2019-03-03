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
example_path = "diamond/"
N_1d = 20
N_rs = (N_1d, N_1d, N_1d)
N_core = 0
N_v = 4
N_c = 5

# N_μ_vvs = ((3, 3, 3), 53)
# N_μ_ccs = ((4, 4, 4), 66)
# N_μ_vcs = ((3, 3, 3), 73)
N_μ_vvs = ((2, 2, 2), 42)
N_μ_ccs = ((3, 3, 3), 93)
N_μ_vcs = ((3, 3, 3), 53)

#variable parameters
N_ks_vec = [(2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (7, 7, 7), (9, 9, 9), (13, 13, 13)]

for N_ks in N_ks_vec
    prob = BSE_k_ISDF.BSEProblemExciting(N_core, N_v, N_c, N_ks, N_rs, example_path * "$(N_ks...)_$(N_1d)/")

    isdf = BSE_k_ISDF.ISDF(prob, N_μ_vvs, N_μ_ccs, N_μ_vcs)
    function f()
        BSE_k_ISDF.ISDF(prob, N_μ_vvs, N_μ_ccs, N_μ_vcs)
    end
    t_isdf = @benchmark $f()

    function g()
        return BSE_k_ISDF.setup_H(prob, isdf)
    end
    t_H_setup = @benchmark $g()

    H = BSE_k_ISDF.setup_H(prob, isdf);
    x = rand(N_v * N_c * prod(N_ks)) + im * rand(N_v * N_c * prod(N_ks))
    function h()
        return H * x
    end
    t_H_x = @benchmark $h()

    save(example_path * "$(N_ks...)_$(N_1d)/benchmark_$(N_μ_vvs)_$(N_μ_ccs)_$(N_μ_vcs).jld2", "t_isdf", t_isdf, "t_H_setup", t_H_setup, "t_H_x", t_H_x)
end

# %% save results

results = [load("diamond/$(N_ks...)_$(N_1d)/benchmark_$(N_μ_vvs)_$(N_μ_ccs)_$(N_μ_vcs).jld2") for N_ks in N_ks_vec]
timings = ["t_isdf", "t_H_setup", "t_H_x"]

setup_times = sum(1e-9 .* time.(minimum.([res[t] for res in results, t in timings[1:2]])); dims = 2)
evaluation_times = 1e-9 .* time.(minimum.([res[t] for res in results, t in timings[3:3]]))

save(example_path * "/benchmark_$(N_μ_vvs)_$(N_μ_ccs)_$(N_μ_vcs).jld2", "N_ks_vec", N_ks_vec, "setup_times", setup_times, "evaluation_times", evaluation_times)
