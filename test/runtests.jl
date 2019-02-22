using Test

push!(LOAD_PATH, "/home/felix/Work/Research/Code")
cd("/home/felix/Work/Research/Code/BSE_k_ISDF/experiments")

using Revise # remove after debugging
using BSE_k_ISDF, LinearAlgebra, FFTW

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

# test energies and orbitals
E_v, E_c = BSE_k_ISDF.energies(prob)
u_v, u_c = BSE_k_ISDF.orbitals(prob)
Ω0_vol = BSE_k_ISDF.Ω0_volume(prob)

@test BSE_k_ISDF.size(prob) == (N_v, N_c, N_k)
@test BSE_k_ISDF.size_k(prob) == N_ks
@test BSE_k_ISDF.size_r(prob) == N_rs
@test size(E_v) == (N_v, N_k)
@test size(E_c) == (N_c, N_k)
@test size(u_v) == (N_r, N_v, N_k)
@test size(u_c) == (N_r, N_c, N_k)
@test sqrt(Ω0_vol / N_r) * norm(u_v[:, end, 1]) ≈ 1 atol=0.001 # relatively large error, should they be normalized by hand?
@test sqrt(Ω0_vol / N_r) * norm(u_c[:, 1, 1]) ≈ 1 atol=0.001

# test periodicity, test if boundary value is close to mean over the neighbors
for u_i in [u_v, u_c]
    u_i_reshaped = reshape(u_i[:, 1, 1], N_rs)
    N_rs_position = div.(N_rs, 2)

    @test isapprox(u_i_reshaped[1, N_rs_position[2], N_rs_position[3]], 0.5 * (u_i_reshaped[2, N_rs_position[2], N_rs_position[3]] + u_i_reshaped[end, N_rs_position[2], N_rs_position[3]]), rtol = 1e-1)
    @test isapprox(u_i_reshaped[N_rs_position[1], 1, N_rs_position[3]], 0.5 * (u_i_reshaped[N_rs_position[1], 2, N_rs_position[3]] + u_i_reshaped[N_rs_position[1], end, N_rs_position[3]]), rtol = 1e-1)
    @test isapprox(u_i_reshaped[N_rs_position[1], N_rs_position[2], 1], 0.5 * (u_i_reshaped[N_rs_position[1], N_rs_position[2], 2] + u_i_reshaped[N_rs_position[1], N_rs_position[2], end]), rtol = 2e-1)
end

# test lattice
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
N_μ_ccs = ((4, 4, 4), 130)
N_μ_vcs = ((3, 3, 3), 51)
N_μ_vv, N_μ_cc, N_μ_vc = 3^3 + 50, 4^3 + 130, 3^3 + 51
@test BSE_k_ISDF.find_r_μ(prob, N_μ_vvs[1],  N_μ_vvs[2]) == BSE_k_ISDF.find_r_μ(prob, N_μ_vv)
@test BSE_k_ISDF.find_r_μ(prob, N_μ_ccs[1],  N_μ_ccs[2]) == BSE_k_ISDF.find_r_μ(prob, N_μ_cc)
@test BSE_k_ISDF.find_r_μ(prob, N_μ_vcs[1],  N_μ_vcs[2]) == BSE_k_ISDF.find_r_μ(prob, N_μ_vc)

@test BSE_k_ISDF.find_r_μ_uniform(9, 9) == 1:9
@test BSE_k_ISDF.find_r_μ_uniform(10, 5) == 1:2:10
for i in 1:11
    @test length(BSE_k_ISDF.find_r_μ_uniform(11, i)) == i
end

isdf = BSE_k_ISDF.ISDF(prob, N_μ_vv, N_μ_cc, N_μ_vc)
ζ_vv, ζ_cc, ζ_vc = BSE_k_ISDF.interpolation_vectors(isdf)
u_v_vv, u_c_cc, u_v_vc, u_c_vc = BSE_k_ISDF.interpolation_coefficients(isdf)
N_μ_vv, N_μ_cc, N_μ_vc = BSE_k_ISDF.size(isdf)

@test BSE_k_ISDF.size(isdf) == (N_μ_vv, N_μ_cc, N_μ_vc)
@test size(ζ_vv) == (N_r, N_μ_vv)
@test size(ζ_cc) == (N_r, N_μ_cc)
@test size(ζ_vc) == (N_r, N_μ_vc)
@test size(u_v_vv) == (N_μ_vv, N_v, N_k)
@test size(u_c_cc) == (N_μ_cc, N_c, N_k)
@test size(u_v_vc) == (N_μ_vc, N_v, N_k)
@test size(u_c_vc) == (N_μ_vc, N_c, N_k)

# test the convolution with padding
function w_conv_reference!(b, w, a)
    dim = ndims(a)
    sa = size(a)
    sw = size(w)

    for i in CartesianIndices(sa)
        b[i] = zero(eltype(b))
        for j in CartesianIndices(sa)
            b[i] += w[CartesianIndex(mod1.((i - j).I .+ 1, sw))] * a[j]
        end
    end

    return b
end
conv_size = (9, 11, 13)
a = rand(Complex{Float64}, conv_size)
w = rand(Complex{Float64}, 2 .* conv_size)
b = rand(Complex{Float64}, conv_size)
ap = zeros(Complex{Float64}, 2 .* conv_size)
bp = zeros(Complex{Float64}, 2 .* conv_size)
cp = zeros(Complex{Float64}, 2 .* conv_size)
w_hat = zeros(Complex{Float64}, 2 .* conv_size)
p = plan_fft(bp)
p_back = plan_bfft(bp)
@test w_conv_reference!(b, w, a) ≈ BSE_k_ISDF.w_conv!(b, w, a, ap, bp, cp, w_hat, p, p_back)

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
@test isapprox(V_dense, V_exact, atol = 2e-2)
@test isapprox(V_dense, V_exact, rtol = 7e-2)

W = BSE_k_ISDF.setup_W(prob, isdf)
W_dense = Matrix(W)
W_exact = BSE_k_ISDF.assemble_exact_W(prob)

@test size(W) == (N_v * N_c * N_k, N_v * N_c * N_k)
@test ishermitian(W)
@test isapprox(W_dense, W_dense', atol = 9e-3) # TODO: find out why this error is so large!
@test isapprox(W_dense, W_dense', rtol = 2e-2)
@test isapprox(W_dense, W_exact, atol=2e-2) # TODO: check how this error becomes smaller for increased number of interpolation points
@test isapprox(W_dense, W_exact, rtol=2e-2)

H = BSE_k_ISDF.setup_H(prob, isdf)
H_dense = Matrix(H)
H_exact = D + 2 * V_exact - W_exact

@test size(H) == (N_v * N_c * N_k, N_v * N_c * N_k)
@test ishermitian(H)
@test isapprox(H_dense, H_dense', atol = 9e-3)
@test isapprox(H_dense, H_dense', rtol = 5e-4)
@test H_dense ≈ D + 2 * V_dense - W_dense
@test isapprox(H_dense, H_exact, atol=5e-2)
@test isapprox(H_dense, H_exact, rtol=3e-3)

# absorption spectrum

direction = 1
N_iter = 50
σ = 0.0055
g = ω -> 1 / π * σ / (ω^2 + σ^2)
E_min, E_max = 0.0, 1.0
Erange = E_min:0.001:E_max

d = BSE_k_ISDF.optical_absorption_vector(prob, direction)
scaling = norm(d)^2 * 8 * π^2 / (Ω0_vol * N_k)

optical_absorption_lanc = BSE_k_ISDF.lanczos_optical_absorption(prob, isdf, direction, N_iter, g, Erange)
optical_absorption_dense_lanc = BSE_k_ISDF.lanczos_optical_absorption(H_dense, d, N_iter, g, Erange, scaling)
optical_absorption_exact_lanc = BSE_k_ISDF.lanczos_optical_absorption(H_exact, d, N_iter, g, Erange, scaling)

ev_dense, ef_dense = eigen(H_dense)
ev_dense = real.(ev_dense)
oscillator_strength_dense = [dot(d, ef_dense[:, i]) / norm(d) for i in 1:size(ef_dense, 2)]
optical_absorption_dense_eig = BSE_k_ISDF.lanczos_optical_absorption(ev_dense, abs2.(oscillator_strength_dense), g, Erange, scaling)

ev_exact, ef_exact = eigen(H_exact)
ev_exact = real.(ev_exact)
oscillator_strength_exact = [dot(d, ef_exact[:, i]) / norm(d) for i in 1:size(ef_exact, 2)]
optical_absorption_exact_eig = BSE_k_ISDF.lanczos_optical_absorption(ev_exact, abs2.(oscillator_strength_exact), g, Erange, scaling)

@test optical_absorption_lanc ≈ optical_absorption_dense_lanc
@test isapprox(optical_absorption_lanc, optical_absorption_exact_lanc, rtol = 5e-2)
@test isapprox(optical_absorption_lanc, optical_absorption_dense_eig, rtol = 5e-2)
@test isapprox(optical_absorption_dense_eig, optical_absorption_exact_eig, rtol = 2e-2)
