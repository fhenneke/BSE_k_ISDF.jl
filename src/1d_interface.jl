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

function V_entry(V, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit)
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

# TODO: remove this special case for 1D?
function ISDF(prob::BSEProblem1D, N_μ_vv::Int, N_μ_cc::Int, N_μ_vc::Int)
    N_unit = length(prob.prob.r_unit)
    r_μ_vv_indices = find_r_μ(N_unit, N_μ_vv)
    r_μ_cc_indices = find_r_μ(N_unit, N_μ_cc)
    r_μ_vc_indices = find_r_μ(N_unit, N_μ_vc)

    return ISDF(r_μ_vv_indices, r_μ_cc_indices, r_μ_vc_indices, prob.u_v, prob.u_c)
end
