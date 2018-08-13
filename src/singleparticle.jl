# single particle problem

function SPProblem(V_sp, l, N_unit, N_k)
    r_unit = linspace(0, l, N_unit + 1)[1:(end-1)]
    r_super = linspace(-div(N_k, 2) * l, div(N_k + 1, 2) * l, N_unit * N_k + 1)[1:(end-1)]
    k_bz = linspace(-pi / l, pi / l, N_k + 1)[1:(end - 1)]

    return SPProblem(V_sp, l, r_unit, r_super, k_bz)
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
    nabla = spdiagm((-Δr_inv, Δr_inv[1:(end - 1)]), (0, 1))
    nabla[end, 1] = Δr_inv[end]
    return -im * nabla + k * I
end

function modified_momentum_operator(r, l, k) # with periodic boundary conditions
    return modified_momentum_operator(vcat(r, l), k)
end

function potential_energy(r, V_fun)
    V = spdiagm(V_fun.(r))
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
function solve(prob::SPProblem)
    V_sp = prob.V_sp
    l = prob.l
    r = prob.r_unit
    k_bz = prob.k_bz

    ev = []
    ef = []
    for k in k_bz
        H_k = Hermitian(full(single_particle_hamiltonian(r, l, k, V_sp)))
        sol = eig(H_k)
        push!(ev, sol[1])
        push!(ef, sol[2])
    end

    return SPSolution(prob, ev, ef)
end
