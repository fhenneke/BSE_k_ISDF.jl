using Test

push!(LOAD_PATH, "/home/felix/Work/Research/Code")
cd("/home/felix/Work/Research/Code/BSE_k_ISDF/experiments")

using Revise # remove after debugging
using BSE_k_ISDF, LinearAlgebra

# %% bse problem type
example_path = "diamond/333_20_test/"

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
@test sqrt(BSE_k_ISDF.Ω0_volume(prob) / N_r) * norm(u_v[:, end, 1]) ≈ 1 atol=0.001 # relatively large error, should they be normalized by hand?
@test sqrt(BSE_k_ISDF.Ω0_volume(prob) / N_r) * norm(u_c[:, 1, 1]) ≈ 1 atol=0.001

# test periodicity, test if boundary value is close to mean over the neighbors
for u_i in [u_v, u_c]
    u_i_reshaped = reshape(u_i[:, 1, 1], N_rs)
    N_rs_position = div.(N_rs, 2)

    @test isapprox(u_i_reshaped[1, N_rs_position[2], N_rs_position[3]], 0.5 * (u_i_reshaped[2, N_rs_position[2], N_rs_position[3]] + u_i_reshaped[end, N_rs_position[2], N_rs_position[3]]), rtol = 1e-1)
    @test isapprox(u_i_reshaped[N_rs_position[1], 1, N_rs_position[3]], 0.5 * (u_i_reshaped[N_rs_position[1], 2, N_rs_position[3]] + u_i_reshaped[N_rs_position[1], end, N_rs_position[3]]), rtol = 1e-1)
    @test isapprox(u_i_reshaped[N_rs_position[1], N_rs_position[2], 1], 0.5 * (u_i_reshaped[N_rs_position[1], N_rs_position[2], 2] + u_i_reshaped[N_rs_position[1], N_rs_position[2], end]), rtol = 2e-1)
end

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

fft_frequencies = BSE_k_ISDF.fftfreq(N_rs...)

@test size(fft_frequencies) == (3, N_r)
@test fft_frequencies[:, 1] == [0, 0, 0]
@test fft_frequencies[:, N_rs[1]] == [-1, 0, 0]
@test fft_frequencies[:, N_rs[1] + 1] == [0, 1, 0]
@test fft_frequencies[:, end] == [-1, -1, -1]
for iG in 1:N_r
    @test BSE_k_ISDF.G_vector_to_index(Int.(fft_frequencies[:, iG]), N_rs) == iG
end

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

V = BSE_k_ISDF.setup_V(prob, isdf)
V_dense = Matrix(V)
V_exact = BSE_k_ISDF.assemble_exact_V(prob)

@test size(V) == (N_v * N_c * N_k, N_v * N_c * N_k)
@test ishermitian(V)
@test V_dense ≈ V_dense'
@test norm(V_dense - V_exact) / norm(V_exact) < 0.05

W = BSE_k_ISDF.setup_W(prob, isdf)
W_dense = Matrix(W)
W_exact = BSE_k_ISDF.assemble_exact_W(prob)

@test size(W) == (N_v * N_c * N_k, N_v * N_c * N_k)
@test ishermitian(W)
@test isapprox(W_dense, W_dense', atol = 1e-2) # TODO: find out why this error is so large!
@test isapprox(W_dense, W_exact, atol=3e-2) # TODO: check how this error becomes smaller for increased number of interpolation points

H = BSE_k_ISDF.setup_H(prob, isdf)
H_dense = Matrix(H)
H_exact = D + 2 * V_exact - W_exact

@test size(H) == (N_v * N_c * N_k, N_v * N_c * N_k)
@test ishermitian(H)
@test isapprox(H_dense, H_dense', atol = 1e-2)
@test H_dense ≈ D + 2 * V_dense - W_dense
@test isapprox(H_dense, H_exact, atol=4e-2)
