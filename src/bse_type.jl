# type for abstract bse problem. mainly contains infos on methods
# which have to implemented by this type

import Base.size

"""
abstract type for AbstractBSEProblems

    concrete BSE problem types should implement the methods
    -   size
    -   size_k
    -   size_r
    -   energies
    -   orbitals
    -   lattice_matrix
    -   compute_w_hat
    -   optical_absorption_vector
"""
abstract type AbstractBSEProblem end

"""
    size(prob)

Return the size of the problem in the form of a tuple (N_v, N_c, N_k).
"""
function size(prob::AbstractBSEProblem)
    error("not implemented yet")
end

"""
    size_k(prob)

Return the size of the k grid problem in the form of a tuple.
"""
function size_k(prob::AbstractBSEProblem)
    error("not implemented yet")
end

"""
    size_r(prob)

Return the size of the problem in the form of a tuple (N_v, N_c, N_k).
"""
function size_r(prob::AbstractBSEProblem)
    error("not implemented yet")
end

"""
    energies(prob)

Return the single particle energies in the form of a tuple of 2d arrays of sizes (N_v, N_k) and (N_c, N_k).
"""
function energies(prob::AbstractBSEProblem)
    error("not implemented yet")
end

"""
    orbitals(prob)

Return the single particle orbitals in the form of a tuple of 3d-arrays of sizes (N_r, N_v, N_k) and (N_r, N_c, N_k).
"""
function orbitals(prob::AbstractBSEProblem)
    error("not implemented yet")
end

"""
    lattice_matrix(prob)

Return a 2d array consisting of the 3 lattice vectors in the different columns.
"""
function lattice_matrix(prob::AbstractBSEProblem)
    error("not implemented yet")
end

"""
    compute_w_hat(prob)

Return a representation of the screening in the form of a three-tuple `w_hat, q_2bz_ind, q_2bz_shift`. Here
-   `w_hat` is a vector of length `N_k` of tuples consiting of a matrix of ceofficients and a matrix whos columns represent G vectors;
-   `q_2bz_ind` is a vector of integers of length `N_q` such that `q_2bz_ind[iq]` is the index of the restricted q-point corresponding to the q-point at `iq`.
-   `q_2bz_shift` is a matrix whos columns correspond to the shift of q-points to restricted q-points.
"""
function lattice_matrix(prob::AbstractBSEProblem)
    error("not implemented yet")
end

"""
    optical_absorption_vector(prob, direction)

Compute the optical absorption vector in along the the direction `a_mat[:, direction]`.
"""
function optical_absorption_vector(prob::AbstractBSEProblem, direction)
    error("not implemented yet")
end
