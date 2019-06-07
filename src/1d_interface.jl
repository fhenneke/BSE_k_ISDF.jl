# 1D model problem to test BSE code

# single particle code
# single particle problem
using LinearAlgebra, SparseArrays, ProgressMeter

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
    k_bz = range(0.0, stop = 1.0, length = N_k + 1)[1:(end - 1)]

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
    kc = 2 * pi / l * (mod.(prob.k_bz .+ 0.5, 1.0) .- 0.5)

    ev = []
    ef = []
    for k in kc
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
    w_hat_vec = (-1).^((0:(N_k * N_unit - 1))) .* fft(w) * (l / N_unit^2);

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

# 

function size(prob::BSEProblem1D)
    return (prob.N_v, prob.N_c, prob.N_k)
end

function size_k(prob::BSEProblem1D)
    return (prob.N_k, 1, 1)
end

function size_r(prob::BSEProblem1D)
    return (length(prob.prob.r_unit), 1, 1)
end

function energies(prob::BSEProblem1D)
    return prob.E_v, prob.E_c
end

function orbitals(prob::BSEProblem1D)
    return prob.u_v, prob.u_c
end

function lattice_matrix(prob::BSEProblem1D)
    return [prob.prob.l 0 0; 0 1 0; 0 0 1]
end

function compute_v_hat(prob::BSEProblem1D)
    return prob.v_hat[:, 1]
end

function compute_w_hat(prob::BSEProblem1D)
    N_k = prob.N_k
    w_hat = [(prob.w_hat[:, :, iq], (0:127)' .* [1, 0, 0]) for iq in 1:N_k]
    q_2bz_ind = [1:N_k; 1:N_k]
    q_2bz_shift = hcat(zeros(Int, 3, div(N_k, 2)), ones(Int, div(N_k, 2))' .* [1, 0, 0], -ones(Int, div(N_k, 2))' .* [1, 0, 0], zeros(Int, 3, div(N_k, 2)))
    return w_hat, q_2bz_ind, q_2bz_shift
end

function find_r_μ(prob::BSEProblem1D, N_μ::Int) # copy of this function in exciting_interface.jl
    r_μ_indices = round.(Int, range(1, stop = size_r(prob)[1] + 1, length = N_μ + 1)[1:(end - 1)])
    return r_μ_indices
end

function optical_absorption_vector(prob::BSEProblem1D, direction)
    return optical_absorption_vector(prob.u_v, prob.u_c, prob.E_v, prob.E_c, prob.prob.r_unit, prob.prob.k_bz)
end

function optical_absorption_vector(u_v, u_c, E_v, E_c, r_unit, k_bz) # TODO: make general? only works in 1d atm
    N_unit = length(r_unit)
    N_v = size(u_v, 2)
    N_c = size(u_c, 2)
    N_k = length(k_bz)
    Δr = r_unit[2] - r_unit[1]
    l = N_unit * Δr

    d = vec([im * Δr * dot(u_c[:, ic, ik],
                  modified_momentum_operator(r_unit, l, 2 * pi / l * k_bz[ik]) *
                    u_v[:, iv, ik]) / (E_c[ic, ik] - E_v[iv, ik])
            for iv in 1:N_v, ic in 1:N_c, ik in 1:N_k])

    return d
end

# reference implementations for testing

function V_entry_realspace(V_func, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit)
    N_unit = size(u_v, 1)
    N_cells = div(length(r_super), N_unit)
    Δr = r_super[2] - r_super[1]
    L = N_cells * N_unit * Δr

    v = complex(0.)
    for (ir, r_1) in enumerate(r_super)
        for (jr, r_2) in enumerate(r_unit)
            diff_r = supercell_difference(r_1, r_2, L)
            v += (conj(u_c[mod1(ir, N_unit), ic, ik]) * u_v[mod1(ir, N_unit), iv, ik]) *
                V_func(diff_r, 0.) *
                (conj(u_v[jr, jv, jk]) * u_c[jr, jc, jk])
        end
    end
    return Δr^2 * v / N_cells
end
function assemble_exact_V_realspace(prob::BSEProblem1D, V_func)
    r_super = prob.prob.r_super
    r_unit = prob.prob.r_unit
    N_v = prob.N_v
    N_c = prob.N_c
    N_k = prob.N_k
    u_v = prob.u_v
    u_c = prob.u_c

    V_reshaped = zeros(Complex{Float64}, N_v, N_c, N_k, N_v, N_c, N_k)
    @showprogress 1 "Assemble exact V ..." for jk in 1:N_k, jc in 1:N_c, jv in 1:N_v, ik in 1:N_k, ic in 1:N_c, iv in 1:N_v
        V_reshaped[iv, ic, ik, jv, jc, jk] =
            V_entry_realspace(V_func, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit)
    end
    V = reshape(V_reshaped, N_v * N_c * N_k, N_v * N_c * N_k)
    for i in 1:(N_v * N_c * N_k)
        V[i, i] = real(V[i, i])
    end

    return V
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

    w = 1 / L * U_c_hat_shift' * (@view(w_hat[:, :, ijk]) * U_v_hat_shift)

    return w
end
function W_entry_realspace(W_func, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit, k_bz)
    N_unit = size(u_v, 1)
    N_cells = div(length(r_super), N_unit)
    Δr = r_super[2] - r_super[1]
    l = N_unit * Δr
    L = N_cells * l

    w = complex(0.)
    for (ir, r_1) in enumerate(r_super)
        for (jr, r_2) in enumerate(r_unit)
            w += (conj(u_c[mod1(ir, N_unit), ic, ik]) * u_c[mod1(ir, N_unit), jc, jk]) *
                exp(-im * 2 * pi / l * (k_bz[ik] - k_bz[jk]) * (r_1 - r_2)) * W_func(r_1, r_2, l, L) *
                (conj(u_v[jr, jv, jk]) * u_v[jr, iv, ik])
        end
    end
    return Δr^2 * w / N_cells
end
function assemble_exact_W_realspace(prob::BSEProblem1D, W_func)
    r_super = prob.prob.r_super
    r_unit = prob.prob.r_unit
    k_bz = prob.prob.k_bz
    N_v = prob.N_v
    N_c = prob.N_c
    N_k = prob.N_k
    u_v = prob.u_v
    u_c = prob.u_c

    W_reshaped = zeros(Complex{Float64}, N_v, N_c, N_k, N_v, N_c, N_k)
    @showprogress 1 "Assemble exact W ..." for jk in 1:N_k, jc in 1:N_c, jv in 1:N_v, ik in 1:N_k, ic in 1:N_c, iv in 1:N_v
        W_reshaped[iv, ic, ik, jv, jc, jk] =
            W_entry_realspace(W_func, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit, k_bz)
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
