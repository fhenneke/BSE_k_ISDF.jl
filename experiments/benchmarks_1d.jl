# benchmarks for ISDF in BSE

# %%
# loading packages
using BenchmarkTools, JLD2, FileIO, LinearAlgebra, FFTW
using BSE_k_ISDF

BLAS.set_num_threads(1)
FFTW.set_num_threads(1)

# general problem setup

l = 1.5
V_sp = r -> 20 * cos(4π / l * r) + 0.2 * sin(2π / l * r)

V_func = (r_1, r_2) -> 1 /  sqrt((r_1 - r_2)^2 + 0.01)
W_func = (r_1, r_2, l, L) -> 0.0625 * (3.0 + sin(2π / l * r_1)) * (3.0 + cos(4π / l * r_2)) *
    exp(-abs(BSE_k_ISDF.supercell_difference(r_1, r_2, L))^2 / (2 * (4 * l)^2)) * V_func(BSE_k_ISDF.supercell_difference(r_1, r_2, L), 0.)

# %% benchmark different values of N_k

# fixed parameters
N_unit = 128

N_core = 0
N_v = 4
N_c = 5

N_μ_vv = 17
N_μ_cc = 23
N_μ_vc = 21

# variable parameters
N_k_vec = 2 .^(4:12)

for N_k in N_k_vec
    sp_prob = BSE_k_ISDF.SPProblem1D(V_sp, l, N_unit, N_k)
    prob = BSE_k_ISDF.BSEProblem1D(sp_prob, N_core, N_v, N_c, V_func, W_func)

    isdf = BSE_k_ISDF.ISDF(prob, N_μ_vv, N_μ_cc, N_μ_vc)
    t_isdf = @benchmark BSE_k_ISDF.ISDF($prob, $N_μ_vv, $N_μ_cc, $N_μ_vc)

    H = BSE_k_ISDF.setup_H(prob, isdf)
    t_H_setup = @benchmark BSE_k_ISDF.setup_H($prob, $isdf)

    x = rand(N_v * N_c * N_k) + im * rand(N_v * N_c * N_k)
    H * x
    t_H_x = @benchmark $H * $x

    save("1d_old/benchmark_$(N_k).jld2", "t_isdf", t_isdf, "t_H_setup", t_H_setup, "t_H_x", t_H_x)
end

# %% save results

results = [load("1d_old/benchmark_$(N_k).jld2") for N_k in N_k_vec]
timings = ["t_isdf", "t_H_setup", "t_H_x"]

setup_times = sum(1e-9 .* time.(minimum.([res[t] for res in results, t in timings[1:2]])); dims = 2)
evaluation_times = 1e-9 .* time.(minimum.([res[t] for res in results, t in timings[3:3]]))

save("1d_old/benchmark.jld2", "N_k_vec", N_k_vec, "setup_times", setup_times, "evaluation_times", evaluation_times)
