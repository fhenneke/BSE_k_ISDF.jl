using Test

push!(LOAD_PATH, "/home/felix/Work/Research/Code")
cd("/home/felix/Work/Research/Code/BSE_k_ISDF/experiments")

using Revise # remove after debugging
using BSE_k_ISDF, LinearAlgebra

# %% bse problem type
example_path = "diamond/333_20/"

N_1d = 20 # TODO: read from file
N_rs = (N_1d, N_1d, N_1d)
N_r = prod(N_rs)
N_core = 0
N_v = 4
N_c = 5
N_ks = (3, 3, 3)
N_k = 27
prob = BSE_k_ISDF.BSEProblemExciting(N_core, N_v, N_c, N_ks, N_rs, example_path)

@test typeof(prob) <: BSE_k_ISDF.AbstractBSEProblem

E_v, E_c = BSE_k_ISDF.energies(prob)
u_v, u_c = BSE_k_ISDF.orbitals(prob)

@test BSE_k_ISDF.size(prob) == (N_v, N_c, N_k)
@test BSE_k_ISDF.size_k(prob) == N_ks
@test BSE_k_ISDF.size_r(prob) == N_rs
@test size(E_v) == (N_v, N_k)
@test size(E_c) == (N_c, N_k)
@test size(u_v) == (N_r, N_v, N_k)
@test size(u_c) == (N_r, N_c, N_k)

a_mat = BSE_k_ISDF.lattice_matrix(prob)
rl = BSE_k_ISDF.r_lattice(prob)
rc = BSE_k_ISDF.r_cartesian(prob)

@test size(a_mat) == (3, 3)
@test size(rl) == (3, N_r)
@test size(rc) == (3, N_r)
@test rc[:, 1] == a_mat * rl[:, 1]

shift = [0.016666666666666, 0.05, 0.0833333333333]
b_mat = 2 * π * inv(a_mat)
kl = BSE_k_ISDF.k_lattice(prob, shift)
kc = BSE_k_ISDF.k_cartesian(prob, b_mat * shift)

@test size(kl) == (3, N_k)
@test size(kc) == (3, N_k)
@test kc[:, 1] ≈ b_mat * kl[:, 1]

fft_frequencies = BSE_k_ISDF.fftfreq(2, 3, 4)

@test size(fft_frequencies) == (3, 2 * 3 * 4)
@test fft_frequencies[:, 1] == [0, 0, 0]
@test fft_frequencies[:, 2] == [-1, 0, 0]
@test fft_frequencies[:, 3] == [0, 1, 0]
@test fft_frequencies[:, end - 1] == [0, -1, -1]
@test fft_frequencies[:, end] == [-1, -1, -1]

# ISDF
N_μ_vvs = ((3, 3, 3), 50)
N_μ_ccs = ((4, 4, 4), 60)
N_μ_vcs = ((3, 3, 3), 70)
isdf = BSE_k_ISDF.ISDF(prob, N_μ_vvs, N_μ_ccs, N_μ_vcs)
ζ_vv, ζ_cc, ζ_vc = BSE_k_ISDF.interpolation_vectors(isdf)
u_v_vv, u_c_cc, u_v_vc, u_c_vc = BSE_k_ISDF.interpolation_coefficients(isdf)
N_μ_vv, N_μ_cc, N_μ_vc = BSE_k_ISDF.size(isdf)

@test (N_μ_vv, N_μ_cc, N_μ_vc) == (78, 122, 92)
@test size(ζ_vv) == (N_r, N_μ_vv)
@test size(ζ_cc) == (N_r, N_μ_cc)
@test size(ζ_vc) == (N_r, N_μ_vc)
@test size(u_v_vv) == (N_μ_vv, N_v, N_k)
@test size(u_c_cc) == (N_μ_cc, N_c, N_k)
@test size(u_v_vc) == (N_μ_vc, N_v, N_k)
@test size(u_c_vc) == (N_μ_vc, N_c, N_k)

# Hamiltonian
D = BSE_k_ISDF.setup_D(prob)
@test size(D) == (N_v * N_c * N_k, N_v * N_c * N_k)
@test isdiag(D)
@test D[1, 1] == E_c[1, 1] - E_v[1, 1]
@test D[2, 2] == E_c[1, 1] - E_v[2, 1]
@test D[1, 2] == 0

V = BSE_k_ISDF.setup_V(prob, isdf)
V_dense = Matrix(V)

@test size(V) == (N_v * N_c * N_k, N_v * N_c * N_k)
@test ishermitian(V)
@test V_dense ≈ V_dense'

W = BSE_k_ISDF.setup_W(prob, isdf)
# W_dense = Matrix(W)
W_dense_new = Matrix(W)
@test W_dense ≈ W_dense_new

@test size(W) == (N_v * N_c * N_k, N_v * N_c * N_k)
@test ishermitian(W)
@test all(abs.(W_dense .- W_dense') .< 2e-4) #TODO: find out why this error is so large!
