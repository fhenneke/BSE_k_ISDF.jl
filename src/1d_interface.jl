# 1D model problem to test BSE code

# single particle code
# single particle problem
using LinearAlgebra, SparseArrays

#TODO: remove this type? just use it in setup of the bse problem?
abstract type SPProblem end

struct SPProblem1D <: SPProblem
    V_sp::Function
    l::Float64
    r_unit::Vector{Float64}
    r_super::Vector{Float64}
    k_bz::Vector{Float64}
end

struct SPSolution1D
    prob
    ev
    ef
end

function SPProblem1D(V_sp, l, N_unit, N_k)
    r_unit = range(0, stop = l, length = N_unit + 1)[1:(end-1)]
    r_super = range(-div(N_k, 2) * l, stop = div(N_k + 1, 2) * l, length = N_unit * N_k + 1)[1:(end-1)]
    k_bz = range(-pi / l, stop = pi / l, length = N_k + 1)[1:(end - 1)]

    return SPProblem1D(V_sp, l, r_unit, r_super, k_bz)
end

"""
function to compute the modified momentum operator \$-i \nabla + k\$ in 1D
used to assemble k-dependent Hamiltonians; uses periodic boundary conditions where the function values at r[1] and r[N + 1] are identified

input
r: vector; containing a 1D space grid with N + 1 points
k: number;
bc: 1 for periodic boundary conditions, 0 for homogeneous dirichlet boundary conditions

output
sparse matrix of size N x N
"""
function modified_momentum_operator(r, k) # with periodic boundary conditions
    N = length(r) - 1

    Δr_inv = 1 ./ vcat(diff(r))
    nabla = spdiagm(0 => -Δr_inv, 1 => Δr_inv[1:(end - 1)])
    nabla[end, 1] = Δr_inv[end]
    return -im * nabla + k * I
end

function modified_momentum_operator(r, l, k) # with periodic boundary conditions
    return modified_momentum_operator(vcat(r, l), k)
end

function potential_energy(r, V_fun)
    V = spdiagm(0 => V_fun.(r))
    return V
end

"""
assemble single particle hamiltonian with crystal momentum k
"""
function single_particle_hamiltonian(r, l, k, V_fun)
    momentum_k = modified_momentum_operator(r, l, k)
    H_k = 1 / 2 * momentum_k * momentum_k' + potential_energy(r, V_fun)
end

"""
solve single particle problem
"""
function solve(prob::SPProblem1D)
    V_sp = prob.V_sp
    l = prob.l
    r = prob.r_unit
    k_bz = prob.k_bz

    ev = []
    ef = []
    for k in k_bz
        H_k = Hermitian(Matrix(single_particle_hamiltonian(r, l, k, V_sp)))
        sol = eigen(H_k)
        push!(ev, sol.values)
        push!(ef, sol.vectors)
    end

    return SPSolution1D(prob, ev, ef)
end

# BSE problem

struct BSEProblem1D <: AbstractBSEProblem
    prob::SPProblem1D
    N_core::Int
    N_v::Int
    N_c::Int
    N_k::Int
    E_v::Array{Float64, 2}
    E_c::Array{Float64, 2}
    u_v::Array{Complex{Float64}, 3}
    u_c::Array{Complex{Float64}, 3}
    v_hat::Array{Complex{Float64}, 2}
    w_hat::Array{Complex{Float64}, 3}
end

function BSEProblem1D(sp_prob::SPProblem1D, N_core, N_v, N_c, V, W)
    sp_sol = solve(sp_prob)
    return BSEProblem1D(sp_sol, N_core, N_v, N_c, V, W)
end

function BSEProblem1D(sp_sol::SPSolution1D, N_core, N_v, N_c, V, W)
    sp_prob = sp_sol.prob
    l = sp_prob.l
    N_unit = length(sp_prob.r_unit)
    N_k = length(sp_prob.k_bz)

    E_v = [sp_sol.ev[ik][N_core + iv] for iv in 1:N_v, ik in 1:N_k]
    E_c = [sp_sol.ev[ik][N_core + N_v + ic] for ic in 1:N_c, ik in 1:N_k]
    u_v = [sqrt(N_unit / l) * sp_sol.ef[ik][ir, N_core + iv] for ir in 1:N_unit, iv in 1:N_v, ik in 1:N_k]
    u_c = [sqrt(N_unit / l) * sp_sol.ef[ik][ir, N_core + N_v + ic] for ir in 1:N_unit, ic in 1:N_c, ik in 1:N_k]

    # fourier representations of V and W
    v_hat = compute_hat_Gq(V, sp_prob.r_super, sp_prob.r_unit)
    w_hat = compute_hat_GGq(W, sp_prob.r_super, sp_prob.r_unit)

    return BSEProblem1D(sp_prob, N_core, N_v, N_c, N_k, E_v, E_c, u_v, u_c, v_hat, w_hat)
end

function compute_hat_Gq(V, r_super, r_unit)
    N_unit = length(r_unit)
    N_k = div(length(r_super), N_unit)
    Δr = r_unit[2] - r_unit[1]
    l = N_unit * Δr

    v = V.(r_super, 0.)
    v_hat_vec = (-1).^(0:(length(r_super) - 1)) .* fft(v) * (l / N_unit)
    v_hat = ifftshift(transpose(reshape(circshift(v_hat_vec, div(N_k, 2)), N_k, N_unit)), 2)

    return v_hat
end

function compute_hat_GGq(W, r_super, r_unit)
    N_unit = length(r_unit)
    N_k = div(length(r_super), N_unit)
    Δr = r_unit[2] - r_unit[1]
    l = N_unit * Δr
    L = N_k * l

    w = W.(r_super .+ r_unit', r_unit', l, L)
    w_hat_vec = (-1).^((0:(N_k * N_unit - 1))) .* fft(w) * (l / N_unit)^2;

    A = (iG, iGp, iq) -> begin
        if iq > div(N_k, 2)
            iq_res = iq - N_k
        else
            iq_res = iq
        end
        if iG > div(N_unit, 2)
            iG_res = iG - N_unit
        else
            iG_res = iG
        end
        if iGp > div(N_unit, 2)
            iGp_res = iGp - N_unit
        else
            iGp_res = iGp
        end
        iqG_r = mod1(iq_res + N_k * (iG_res - 1), N_k * N_unit)
        iG_c = mod1(iG_res - iGp_res + 1, N_unit)

        return (iqG_r, iG_c)
    end

    w_hat = [w_hat_vec[A(iG, iGp, iq)...]
             for iG in 1:N_unit, iGp in 1:N_unit, iq in 1:N_k]

    return w_hat
end

function supercell_difference(r_1, r_2, L)
    r_1 - r_2 - round((r_1 - r_2) / L) * L
end

# reference implementations for testing

function V_entry_realspace(V, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit)
    N_unit = size(u_v, 1)
    N_cells = div(length(r_super), N_unit)
    Δr = r_super[2] - r_super[1]
    L = N_cells * N_unit * Δr

    v = complex(0.)
    for (ir, r_1) in enumerate(r_super)
        for (jr, r_2) in enumerate(r_unit)
            diff_r = supercell_difference(r_1, r_2, L)
            v += (conj(u_c[mod1(ir, N_unit), ic, ik]) * u_v[mod1(ir, N_unit), iv, ik]) *
                V(diff_r, 0.) *
                (conj(u_v[jr, jv, jk]) * u_c[jr, jc, jk])
        end
    end
    return Δr^2 * v / N_cells
end

function V_entry_fast(v_hat, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit)
    N_unit = length(r_unit)
    N_cells = div(length(r_super), N_unit)
    Δr = r_unit[2] - r_unit[1]
    l = N_unit * Δr
    L = N_cells * l

    U_vc_1_hat = fft(u_c[:, ic, ik] .* conj.(u_v[:, iv, ik])) * (l / N_unit)
    U_vc_2_hat = fft(u_c[:, jc, jk] .* conj.(u_v[:, jv, jk])) * (l / N_unit)

    v = 1 / L * U_vc_1_hat' * (v_hat[:, 1] .* U_vc_2_hat)

    return v
end

function W_entry_realspace(W, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit, k_bz)
    N_unit = size(u_v, 1)
    N_cells = div(length(r_super), N_unit)
    Δr = r_super[2] - r_super[1]
    l = N_unit * Δr
    L = N_cells * l

    w = complex(0.)
    for (ir, r_1) in enumerate(r_super)
        for (jr, r_2) in enumerate(r_unit)
            w += (conj(u_c[mod1(ir, N_unit), ic, ik]) * u_c[mod1(ir, N_unit), jc, jk]) *
                exp(-im * (k_bz[ik] - k_bz[jk]) * (r_1 - r_2)) * W(r_1, r_2, l, L) *
                (conj(u_v[jr, jv, jk]) * u_v[jr, iv, ik])
        end
    end
    return Δr^2 * w / N_cells
end

function W_entry_fast(w_hat, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit, k_bz)
    N_unit = length(r_unit)
    N_cells = div(length(r_super), N_unit)
    N_k = length(k_bz)
    Δr = r_unit[2] - r_unit[1]
    l = N_unit * Δr
    L = N_cells * l

    ijk = mod1(ik - jk + 1, N_k)
    if ik - jk >= div(N_k, 2)
        G_shift = -1
    elseif ik - jk < -div(N_k, 2)
        G_shift = 1
    else
        G_shift = 0
    end

    U_c_hat_shift = circshift(fft(u_c[:, ic, ik] .* conj.(u_c[:, jc, jk])), -G_shift) * (l / N_unit)
    U_v_hat_shift = circshift(fft(u_v[:, iv, ik] .* conj.(u_v[:, jv, jk])), -G_shift) * (l / N_unit)

    w = 1 / (l * L) * U_c_hat_shift' * (@view(w_hat[:, :, ijk]) * U_v_hat_shift)

    return w
end

function H_entry_realspace(V, W, iv, ic, ik, jv, jc, jk, E_v, E_c, u_v, u_c, r_super, r_unit, k_bz)
    (E_c[ic, ik] - E_v[iv, ik]) * (iv == jv) * (ic == jc) * (ik == jk) +
        2 * V_entry_realspace(V, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit) -
        W_entry_realspace(W, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit, k_bz)
end

function H_entry_fast(v_hat, w_hat, iv, ic, ik, jv, jc, jk, E_v, E_c, u_v, u_c, r_super, r_unit, k_bz)
    (E_c[ic, ik] - E_v[iv, ik]) * (iv == jv) * (ic == jc) * (ik == jk) +
        2 * V_entry_fast(v_hat, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit) -
        W_entry_fast(w_hat, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit, k_bz)
end

function assemble_exact_V_1d(prob)
    r_super = prob.prob.r_super
    r_unit = prob.prob.r_unit
    N_v = prob.N_v
    N_c = prob.N_c
    N_k = prob.N_k
    u_v = prob.u_v
    u_c = prob.u_c
    v_hat = prob.v_hat

    V_reshaped = zeros(Complex{Float64}, N_v, N_c, N_k, N_v, N_c, N_k)
    for jk in 1:N_k, jc in 1:N_c, jv in 1:N_v, ik in 1:N_k, ic in 1:N_c, iv in 1:N_v
        V_reshaped[iv, ic, ik, jv, jc, jk] =
            V_entry_fast(v_hat, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit)
    end
    V = reshape(V_reshaped, N_v * N_c * N_k, N_v * N_c * N_k)
    for i in 1:(N_v * N_c * N_k)
        V[i, i] = real(V[i, i])
    end

    return V
end

function assemble_exact_W_1d(prob)
    r_super = prob.prob.r_super
    r_unit = prob.prob.r_unit
    k_bz = prob.prob.k_bz
    N_v = prob.N_v
    N_c = prob.N_c
    N_k = prob.N_k
    u_v = prob.u_v
    u_c = prob.u_c
    w_hat = prob.w_hat

    W_reshaped = zeros(Complex{Float64}, N_v, N_c, N_k, N_v, N_c, N_k)
    for jk in 1:N_k, jc in 1:N_c, jv in 1:N_v, ik in 1:N_k, ic in 1:N_c, iv in 1:N_v
        W_reshaped[iv, ic, ik, jv, jc, jk] =
            W_entry_fast(w_hat, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit, k_bz)
    end
    W = reshape(W_reshaped, N_v * N_c * N_k, N_v * N_c * N_k)
    for i in 1:(N_v * N_c * N_k)
        W[i, i] = real(W[i, i])
    end

    return W
end

function assemble_exact_H_1d(prob)
    r_super = prob.prob.r_super
    r_unit = prob.prob.r_unit
    k_bz = prob.prob.k_bz
    N_v = prob.N_v
    N_c = prob.N_c
    N_k = prob.N_k
    E_v = prob.E_v
    E_c = prob.E_c
    u_v = prob.u_v
    u_c = prob.u_c
    v_hat = prob.v_hat
    w_hat = prob.w_hat

    H_reshaped = zeros(Complex{Float64}, N_v, N_c, N_k, N_v, N_c, N_k)
    for jk in 1:N_k, jc in 1:N_c, jv in 1:N_v, ik in 1:N_k, ic in 1:N_c, iv in 1:N_v
        if N_v * N_c * (ik - 1) + N_v * (ic - 1) + iv <= N_v * N_c * (jk - 1) + N_v * (jc - 1) + jv # exploit symmetry
            H_reshaped[iv, ic, ik, jv, jc, jk] =
                H_entry_fast(v_hat, w_hat, iv, ic, ik, jv, jc, jk, E_v, E_c, u_v, u_c, r_super, r_unit, k_bz)
        end
    end
    H = reshape(H_reshaped, N_v * N_c * N_k, N_v * N_c * N_k)
    for i in 1:(N_v * N_c * N_k)
        H[i, i] = real(H[i, i])
    end

    return Hermitian(H)
end

# TODO: remove this special case for 1D?
function ISDF(prob::BSEProblem1D, N_μ_vv::Int, N_μ_cc::Int, N_μ_vc::Int)
    N_unit = length(prob.prob.r_unit)
    r_μ_vv_indices = find_r_μ(N_unit, N_μ_vv)
    r_μ_cc_indices = find_r_μ(N_unit, N_μ_cc)
    r_μ_vc_indices = find_r_μ(N_unit, N_μ_vc)

    return ISDF(r_μ_vv_indices, r_μ_cc_indices, r_μ_vc_indices, prob.u_v, prob.u_c)
end


# TODO: can those methods be combined?
function setup_V(prob::BSEProblem1D, isdf)
    v_hat = prob.v_hat
    r_super = prob.prob.r_super
    r_unit = prob.prob.r_unit
    N_v = prob.N_v
    N_c = prob.N_c
    N_k = prob.N_k
    N_μ = isdf.N_μ_vc
    ζ_vc = isdf.ζ_vc

    V_tilde = assemble_V_tilde1d(v_hat[:, 1], ζ_vc, r_super, r_unit)
    V_workspace = create_V_workspace(N_v, N_c, N_k, N_μ)

    u_v_vc_conj = conj.(isdf.u_v_vc)
    u_c_vc = isdf.u_c_vc

    V = LinearMap{Complex{Float64}}(
        x -> V_times_vector(x, V_tilde, u_v_vc_conj, u_c_vc, V_workspace),
        N_v * N_c * N_k; ishermitian=true)

    return V
end

#TODO: combine with 3d version
function assemble_V_tilde1d(v_hat, ζ_vc, r_super, r_unit)
    N_unit = length(r_unit)
    N_cells = div(length(r_super), N_unit)
    N_μ = size(ζ_vc, 2)
    N_ν = N_μ
    Δr = r_unit[2] - r_unit[1]
    l = N_unit * Δr
    L = N_cells * l

    ζ_vc_hat = fft(ζ_vc, 1) * (l / N_unit)

    V_tilde = 1 / L * ζ_vc_hat' * (v_hat .* ζ_vc_hat)

    return V_tilde
end

#TODO: can this be combined with the 1D version?
function setup_W(prob::BSEProblem1D, isdf)
    w_hat = prob.w_hat

    r_super = prob.prob.r_super
    r_unit = prob.prob.r_unit
    k_bz = prob.prob.k_bz
    N_v = prob.N_v
    N_c = prob.N_c
    N_k = prob.N_k

    N_μ = isdf.N_μ_cc
    N_ν = isdf.N_μ_vv
    ζ_vv = isdf.ζ_vv
    ζ_cc = isdf.ζ_cc
    u_v_vv_conj = conj.(isdf.u_v_vv)
    u_c_cc = isdf.u_c_cc

    W_tilde = assemble_W_tilde1d(w_hat, ζ_vv, ζ_cc, r_super, r_unit, k_bz)
    W_workspace = create_W_workspace1d(N_v, N_c, N_k, N_ν, N_μ)
    W_tilde_hat = fft(W_tilde, 1)


    W = LinearMap{Complex{Float64}}(
        x -> W_times_vector(x, W_tilde_hat, u_v_vv_conj, u_c_cc, W_workspace),
        N_v * N_c * N_k; ishermitian=true)

    return W
end

function assemble_W_tilde1d(w_hat, ζ_vv, ζ_cc, r_super, r_unit, k_bz)
    N_unit = length(r_unit)
    N_cells = div(length(r_super), N_unit)
    N_μ = size(ζ_cc, 2)
    N_ν = size(ζ_vv, 2)
    N_k = length(k_bz)
    Δr = r_unit[2] - r_unit[1]
    l = N_unit * Δr
    L = N_cells * l

    ζ_cc_hat = fft(ζ_cc, 1) * (l / N_unit)
    ζ_vv_hat = fft(ζ_vv, 1) * (l / N_unit)
    ζ_cc_hat_shift = circshift(ζ_cc_hat, (1, 0))
    ζ_vv_hat_shift = circshift(ζ_vv_hat, (1, 0))

    k_bz_shifted = range(0.0, stop = k_bz[end] - k_bz[1], length = N_k)

    W_tilde_shifted = complex(zeros(N_k, N_μ, N_ν))

    for ik in 1:div(N_k, 2)
        W_tilde_shifted[ik, :, :] .= W_tilde_at_q(@view(w_hat[:, :, ik]), ζ_cc_hat, ζ_vv_hat, l, L)
    end
    for ik in (div(N_k, 2) + 1):N_k
        W_tilde_shifted[ik, :, :] .= W_tilde_at_q(@view(w_hat[:, :, ik]), ζ_cc_hat_shift, ζ_vv_hat_shift, l, L)
    end

    # W_tilde = cat(1, W_tilde_shifted[2:end, :, :], zeros(1, N_μ, N_ν), conj.(W_tilde_shifted[end:-1:1, :, :]))
    W_tilde = cat(W_tilde_shifted[1:end, :, :], zeros(1, N_μ, N_ν), conj.(W_tilde_shifted[end:-1:2, :, :]); dims = 1)

    return W_tilde
end

function W_tilde_at_q(w_k_hat::AbstractMatrix, ζ_1_hat, ζ_2_hat, l, L)
    return 1 / (l * L) * (ζ_1_hat' * (w_k_hat * ζ_2_hat))
end

function create_W_workspace1d(N_v, N_c, N_k, N_ν, N_μ)
    P = plan_fft(complex(zeros(2 * N_k, N_μ, N_ν)), 1)
    P_inv = inv(P)
    X = complex(zeros(N_v, N_c, N_k))
    B = complex(zeros(N_ν, N_c, N_k))
    C = complex(zeros(N_k, N_μ, N_ν))
    C_padded = complex(zeros(2 * N_k, N_μ, N_ν))
    C_transformed = complex(zeros(2 * N_k, N_μ, N_ν))
    D_large = complex(zeros(2 * N_k, N_μ, N_ν))
    D_small = complex(zeros(N_k, N_μ, N_ν))
    E = complex(zeros(N_ν, N_c, N_k))
    F = complex(zeros(N_v, N_c, N_k))
    W_workspace = (P, P_inv, X, B, C, C_padded, C_transformed, D_large, D_small, E, F)

    return W_workspace
end

function create_W_workspace_fast(N_v, N_c, N_k, N_ν, N_μ)
    P = plan_fft(complex(zeros(2 * N_k, N_μ, N_ν)), 1)
    P_back = plan_bfft(complex(zeros(2 * N_k, N_μ, N_ν)), 1)
    X = complex(zeros(N_v, N_c))
    B = complex(zeros(N_ν, N_c))
    C = complex(zeros(N_μ, N_ν))
    C_padded = complex(zeros(2 * N_k, N_μ, N_ν))
    C_transformed = complex(zeros(2 * N_k, N_μ, N_ν))
    D_large = complex(zeros(2 * N_k, N_μ, N_ν))
    D = complex(zeros(N_μ, N_ν))
    E = complex(zeros(N_ν, N_c))
    F = complex(zeros(N_v, N_c))
    W_workspace = (P, P_back, X, B, C, C_padded, C_transformed, D_large, D, E, F)

    return W_workspace
end

# matrix free W TODO: combine with 3d version?
function W_times_vector1d(x, W_tilde_hat, u_v_vv_conj, u_c_cc, W_workspace)
    N_k = size(u_v_vv_conj, 3)
    N_v = size(u_v_vv_conj, 2)
    N_c = size(u_c_cc, 2)
    P, P_inv, X, B, C, C_padded, C_transformed, D_large, D_small, E, F = W_workspace

    X[:] .= x
    @views for jk in 1:N_k
        mul!(B[:, :, jk], u_v_vv_conj[:, :, jk], X[:, :, jk])
        mul!(C[jk, :, :], u_c_cc[:, :, jk], transpose(B[:, :, jk]))
    end
    C_padded[1:N_k, :, :] .= C
    mul!(C_transformed, P, C_padded)
    C_transformed .*= W_tilde_hat
    mul!(D_large, P_inv, C_transformed)
    D_small .= @view D_large[1:N_k, :, :]
    @views for ik in 1:N_k
        mul!(E[:, :, ik], transpose(D_small[ik, :, :]), conj.(u_c_cc[:, :, ik]))
        mul!(F[:, :, ik], adjoint(u_v_vv_conj[:, :, ik]), E[:, :, ik])
    end

    return copy(vec(F))
end

#TODO: remove/combine with optimizations in 3d version
function W_times_vector_fast!(y, x, W_tilde_hat, u_v_vv_conj, u_c_cc, u_c_cc_conj, W_workspace)
    N_μ = size(W_tilde_hat, 2)
    N_ν = size(W_tilde_hat, 3)
    N_k = length(u_v_vv_conj)
    N_v = size(u_v_vv_conj[1], 2)
    N_c = size(u_c_cc[1], 2)

    P, P_back, X, B, C, C_padded, C_transformed, D_large, D, E, F = W_workspace

    for jk in 1:N_k
        X[:] .= @view(x[(N_v * N_c * (jk - 1) + 1):(N_v * N_c * jk)])
        mul!(B, u_v_vv_conj[jk], X)
        mul!(C, u_c_cc[jk], transpose(B))
        C_padded[jk, :, :] .= C
    end
    mul!(C_transformed, P, C_padded)
    C_transformed .*= (1 / (2 * N_k)) .* W_tilde_hat
    mul!(D_large, P_back, C_transformed)
    for ik in 1:N_k
        D .= @view(D_large[mod1(ik - 1, 2 * N_k), :, :])
        mul!(E, transpose(D), u_c_cc_conj[ik])
        mul!(F, adjoint(u_v_vv_conj[ik]), E)
        y[(N_v * N_c * (ik - 1) + 1):(N_v * N_c * ik)] .= @view(F[:])
    end

    return y
end

function optical_absorption_vector(prob::BSEProblem1D)
    return optical_absorption_vector(prob.u_v, prob.u_c, prob.E_v, prob.E_c, prob.prob.r_unit, prob.prob.k_bz)
end

function optical_absorption_vector(u_v, u_c, E_v, E_c, r_unit, k_bz) # TODO: make general? only works in 1d atm
    N_unit = length(r_unit)
    N_v = size(u_v, 2)
    N_c = size(u_c, 2)
    N_k = length(k_bz)
    l = N_unit * (r_unit[2] - r_unit[1])

    d = vec([im * dot(u_c[:, ic, ik],
                  modified_momentum_operator(r_unit, l, k_bz[ik]) *
                    u_v[:, iv, ik]) /
            (E_c[ic, ik] - E_v[iv, ik])
            for iv in 1:N_v, ic in 1:N_c, ik in 1:N_k])

    return d
end

function optical_absorption(prob::BSEProblem1D, g, Erange)
    H = assemble_exact_H(prob)
    F = eigen(Hermitian(H))
    ev, ef = F.values, F.vectors

    absorption = optical_absorption(ev, ef, prob.u_v, prob.u_c, prob.E_v, prob.E_c, prob.prob.r_unit, prob.prob.k_bz, g, Erange)

    return absorption
end

function optical_absorption(ev, ef, u_v, u_c, E_v, E_c, r_unit, k_bz, g, Erange)
    N_v = size(u_v, 2)
    N_c = size(u_c, 2)
    N_k = length(k_bz)
    l = (r_unit[2] - r_unit[1]) * length(r_unit)

    d = optical_absorption_vector(u_v, u_c, E_v, E_c, r_unit, k_bz)
    osc = [abs2(dot(d, ef[:, n])) for n in 1:(N_v * N_c * N_k)]

    absorption = zeros(length(Erange))
    for j in 1:(N_v * N_c * N_k)
        absorption .+= osc[j] * g.(Erange .- ev[j])
    end
    absorption .*= 8 * π^2 / l

    return absorption
end
