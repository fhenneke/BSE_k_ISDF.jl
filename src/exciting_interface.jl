# code for interoperability with exciting

import DelimitedFiles: readdlm
import HDF5: h5open, readmmap
import LightXML: parse_file, root, attribute
import Random: MersenneTwister, shuffle
import Mmap

mutable struct BSEProblemExciting <: AbstractBSEProblem
    input_xml

    N_rs
    r_lattice
    r_cartesian
    a_mat
    Ω0_vol
    atoms

    N_ks
    k_bz
    b_mat
    BZ_vol

    E_v
    E_c
    u_v
    u_c

    N_k_diffs
    q_bz
    q_2bz
    q_2bz_ind
    q_2bz_shift
    w_hat
    gqmax

    pmat
    #TODO: add scissor shift here?
end

function BSEProblemExciting(N_core, N_v, N_c, N_ks, N_rs, path)
    input_xml = parse_file(path * "/input.xml")

    r_lattice, r_cartesian, a_mat, Ω0_vol, k_bz, b_mat, BZ_vol = read_r_points_k_points(N_rs, path)

    atoms = []
    for species in root(input_xml)["structure"][1]["species"]
        species_name = attribute(species, "speciesfile")[1:(end - 4)]
        mt_radius = parse(Float64, attribute(species, "rmt"))
        for atom in species["atom"]
            position = [parse(Float64, x) for x in split(attribute(atom, "coord"))]
            push!(atoms, (species = species_name, position = position, mt_radius = mt_radius))
        end
    end

    N_k = size(k_bz, 2)
    E_v, E_c, u_v, u_c = read_eigenvalues_eigenfunctions(N_core, N_v, N_c, N_k, N_rs, k_bz, path)

    N_ks, N_k_diffs, q_bz, q_2bz, q_2bz_ind, q_2bz_shift, w_hat = read_q_points_screenedcoulomb(N_ks, Ω0_vol, path)
    gqmax = parse(Float64, attribute(root(input_xml)["xs"][1], "gqmax"))

    pmat = read_pmat(N_core, N_v, N_c, N_k, path)

    return BSEProblemExciting(input_xml, N_rs, r_lattice, r_cartesian, a_mat, Ω0_vol, atoms, N_ks, k_bz, b_mat, BZ_vol, E_v, E_c, u_v, u_c, N_k_diffs, q_bz, q_2bz, q_2bz_ind, q_2bz_shift, w_hat, gqmax, pmat)
end

function read_r_points_k_points(N_rs, path) # TODO: make into type constructor?
    lattice_out = readdlm(path * "/LATTICE.OUT")
    a_mat = zeros(3, 3)
    a_mat .= lattice_out[8:10, 1:3]
    b_mat = zeros(3, 3)
    b_mat .= lattice_out[23:25, 1:3]

    Ω0_vol = abs(det(a_mat))
    BZ_vol = abs(det(b_mat))

    rl_1d_1 = range(0.0, stop = 1.0, length = N_rs[1] + 1)[1:N_rs[1]]
    rl_1d_2 = range(0.0, stop = 1.0, length = N_rs[2] + 1)[1:N_rs[2]]
    rl_1d_3 = range(0.0, stop = 1.0, length = N_rs[3] + 1)[1:N_rs[3]]
    rl_1 = kron(ones(N_rs[3]), ones(N_rs[2]), rl_1d_1)
    rl_2 = kron(ones(N_rs[3]), rl_1d_2, ones(N_rs[1]))
    rl_3 = kron(rl_1d_3, ones(N_rs[2]), ones(N_rs[1]))
    r_lattice = transpose(hcat(rl_1, rl_2, rl_3))
    r_cartesian = zero(r_lattice)
    for ir in 1:size(r_lattice, 2)
        r_cartesian[:, ir] = a_mat * r_lattice[:, ir]
    end

    k_points_out = readdlm(path * "/KPOINTS_QMT001.OUT")

    k_bz = float.(transpose(k_points_out[2:end, 2:4]))

    r_lattice, r_cartesian, a_mat, Ω0_vol, k_bz, b_mat, BZ_vol
end

function read_eigenvalues_eigenfunctions(N_core, N_v, N_c, N_k, N_rs, k_bz, path)
    N_r = prod(N_rs)

    eigval_out = readdlm(path * "/EIGVAL_QMT001.OUT")

    N_states = eigval_out[2, 1]
    eigval = zeros(N_states, N_k)
    for ik in 1:N_k
        eigval[:, ik] = eigval_out[(4 + 1 + (N_states + 2) * (ik - 1)):(4 + N_states + (N_states + 2) * (ik - 1)), 2]
    end

    E_v = eigval[(N_core + 1):(N_core + N_v), :]
    E_c = eigval[(N_core + N_v + 1):(N_core + N_v + N_c), :]

    mask_boundary = trues(N_rs[1] + 1, N_rs[2] + 1, N_rs[3] + 1)
    mask_boundary[N_rs[1] + 1, :, :] .= false
    mask_boundary[:, N_rs[2] + 1, :] .= false
    mask_boundary[:, :, N_rs[3] + 1] .= false
    mask_boundary = vec(mask_boundary)

    file = h5open(path * "/bse_output.h5", "r")

    s = open(path * "/u_v.bin", "w+")
    for ik in 1:N_k
        phase = kron(
            exp.(-2 * pi * im * range(0.0, stop = 1.0, length = N_rs[3] + 1)[1:N_rs[3]] .* k_bz[3, ik]),
            exp.(-2 * pi * im * range(0.0, stop = 1.0, length = N_rs[2] + 1)[1:N_rs[2]] .* k_bz[2, ik]),
            exp.(-2 * pi * im * range(0.0, stop = 1.0, length = N_rs[1] + 1)[1:N_rs[1]] .* k_bz[1, ik]))
        for iv in 1:N_v
            c = read(file, "wfplot/" * lpad(string(N_core + iv), 4, string(0)) * "/" * lpad(string(ik), 4, string(0)) * "/data")

            write(s, (c[1, mask_boundary] .+ im * c[2, mask_boundary]) .* phase)
        end
    end
    close(s)
    s_v = open(path * "/u_v.bin")
    u_v = Mmap.mmap(s_v, Array{Complex{Float64}, 3}, (N_r, N_v, N_k))

    s = open(path * "/u_c.bin", "w+")
    for ik in 1:N_k
        phase = kron(
            exp.(-2 * pi * im * range(0.0, stop = 1.0, length = N_rs[3] + 1)[1:N_rs[3]] .* k_bz[3, ik]),
            exp.(-2 * pi * im * range(0.0, stop = 1.0, length = N_rs[2] + 1)[1:N_rs[2]] .* k_bz[2, ik]),
            exp.(-2 * pi * im * range(0.0, stop = 1.0, length = N_rs[1] + 1)[1:N_rs[1]] .* k_bz[1, ik]))
        for ic in 1:N_c
            c = read(file, "wfplot/" * lpad(string(N_core + N_v + ic), 4, string(0)) * "/" * lpad(string(ik), 4, string(0)) * "/data")

            write(s, (c[1, mask_boundary] .+ im * c[2, mask_boundary]) .* phase)
        end
    end
    close(s)
    s_c = open(path * "/u_c.bin")
    u_c = Mmap.mmap(s_c, Array{Complex{Float64}, 3}, (N_r, N_c, N_k))

    close(file)

    return E_v, E_c, u_v, u_c
end

function read_q_points_screenedcoulomb(N_ks, Ω0_vol, path)
    q_points_out = readdlm(path * "/QPOINTS_SCR.OUT")
    q_bz = float.(transpose(q_points_out[2:end, 2:4]))

    N_k_diffs = size_q(N_ks)

    # q grid [0, 1[ ∪ [-1, 0[
    q_2bz = q_lattice(N_ks)

    N_k_diff = size(q_2bz, 2)

    q_2bz_ind = zeros(Int, N_k_diff)
    q_2bz_shift = zeros(Int, 3, N_k_diff)

    for iq in 1:N_k_diff
        q_2bz_ind[iq] = findfirst(vec(prod(mod.(q_bz .- q_2bz[:, iq] .+ 1e-5, 1) .< 1e-3; dims=1)))
        q_2bz_shift[:, iq] = round.(Int, q_bz[:, q_2bz_ind[iq]] - q_2bz[:, iq])
    end

    G_vec = Array{Int64, 2}[]
    for iq in 1:size(q_bz, 2)
        G = round.(Int, readdlm(path * "/GQPOINTS/GQPOINTS_SCR_Q" * lpad(string(iq), 5, string(0)) * ".OUT")[2:end, 2:4]' .- q_bz[:, iq])
        push!(G_vec, G)
    end

    w_hat = []
    file = h5open(path * "/bse_output.h5", "r")
    for iq in 1:size(q_bz, 2)
        wqq = read(file, "screenedpotential/" * lpad(string(iq), 4, string(0)) * "/wqq")
        G_len = size(G_vec[iq], 2)
        wqq_complex = wqq[1, 1:G_len, 1:G_len] + im * wqq[2, 1:G_len, 1:G_len]
        push!(w_hat, (wqq_complex, G_vec[iq])) # TODO: permutedim necessary?
    end
    close(file)

    return N_ks, N_k_diffs, q_bz, q_2bz, q_2bz_ind, q_2bz_shift, w_hat
end

function read_pmat(N_core, N_v, N_c, N_k, path)
    pmat = zeros(Complex{Float64}, N_v * N_c * N_k, 3)
    pmat_reshaped = reshape(pmat, N_v, N_c, N_k, 3)

    file = h5open(path * "/bse_output.h5", "r")
    for ik in 1:N_k
        pmat_block = permutedims(read(file, "pmat/" * lpad(string(ik), 4, string(0)) * "/pmat/"), [3, 4, 2, 1])
        pmat_reshaped[:, :, ik, :] = pmat_block[(N_core + 1):(N_core + N_v), (N_core + N_v + 1):(N_core + N_v + N_c), :, 1] -
                                im * pmat_block[(N_core + 1):(N_core + N_v), (N_core + N_v + 1):(N_core + N_v + N_c), :, 2]
    end
    close(file)

    return pmat
end

function size(prob::BSEProblemExciting)
    return (size(prob.E_v, 1), size(prob.E_c, 1), prod(prob.N_ks))
end

function size_k(prob::BSEProblemExciting)
    return prob.N_ks
end

function size_r(prob::BSEProblemExciting)
    return prob.N_rs
end

function energies(prob::BSEProblemExciting)
    return prob.E_v, prob.E_c
end

function orbitals(prob::BSEProblemExciting)
    return prob.u_v, prob.u_c
end

function lattice_matrix(prob::BSEProblemExciting)
    return prob.a_mat
end

function compute_v_hat(prob::BSEProblemExciting)
    return compute_v_hat(prob, prob.gqmax)
end

function compute_w_hat(prob::BSEProblemExciting)
    return prob.w_hat, prob.q_2bz_ind, prob.q_2bz_shift
end

function optical_absorption_vector(prob::BSEProblemExciting, direction)
    N_v, N_c, N_k = size(prob)
    ev = vec([prob.E_c[ic, ik] - prob.E_v[iv, ik] for iv in 1:N_v, ic in 1:N_c, ik in 1:N_k])
    return prob.pmat[:, direction] ./ ev
end

"""
    find_r_μ(prob::BSEProblemExciting, N_μ_irs::Tuple, N_μ_mt::Int)

Chooses interpolation points as union of a uniform grid of size
`N_μ_irs` and `N_μ_mt` random points in the muffin tins around the
atoms.
"""
function find_r_μ(prob::BSEProblemExciting, N_μ_irs::Tuple, N_μ_mt::Int)
    # uniform grid (interstitial region and muffin tin region)
    N_rs = size_r(prob)
    N_r = prod(N_rs)

    grid_1 = find_r_μ_uniform(N_rs[1], N_μ_irs[1])
    grid_2 = find_r_μ_uniform(N_rs[2], N_μ_irs[2])
    grid_3 = find_r_μ_uniform(N_rs[3], N_μ_irs[3])
    r_μ_mask_1 = zeros(N_rs[1])
    r_μ_mask_2 = zeros(N_rs[2])
    r_μ_mask_3 = zeros(N_rs[3])
    r_μ_mask_1[grid_1] .= 1.0
    r_μ_mask_2[grid_2] .= 1.0
    r_μ_mask_3[grid_3] .= 1.0
    r_μ_mask = Bool.(kron(r_μ_mask_3, r_μ_mask_2, r_μ_mask_1))

    # additional points in muffin tin regions
    r_mask_mt = zeros(Bool, N_r)
    for atom in prob.atoms
        for ir in 1:N_r
            for is in -1:1, js in -1:1, ms in -1:1
                if norm(prob.r_cartesian[:, ir] - prob.a_mat * (atom[:position] + [is, js, ms])) < atom[:mt_radius]
                    if r_μ_mask[ir] == false
                        r_mask_mt[ir] = true
                    end
                end
            end
        end
    end

    rng = MersenneTwister(0) # for reproducibility

    r_μ_indices_mt = shuffle(rng, findall(r_mask_mt))[1:N_μ_mt]
    r_μ_mask[r_μ_indices_mt] .= true

    r_μ_indices = findall(r_μ_mask)

    return r_μ_indices
end

function find_r_μ(prob::BSEProblemExciting, N_μ::Int)
    # splits the points roughly equal among uniform grid and each of the atoms
    N_atoms = length(prob.atoms)
    N_1d = 1
    while (N_1d + 1)^3 * N_atoms < N_μ
        N_1d += 1
    end
    N_μ_irs = (N_1d, N_1d, N_1d)
    N_μ_mt = N_μ - N_1d^3

    return find_r_μ(prob, N_μ_irs, N_μ_mt)
end

"""
    find_r_μ_uniform(N_r, N_μ)

Helper function to select `N_μ` points uniformly from the range `1:N_r`.
"""
function find_r_μ_uniform(N_r::Int, N_μ::Int)
    r_μ_indices = round.(Int, range(1, stop = N_r + 1, length = N_μ + 1)[1:(end - 1)])
    return r_μ_indices
end

# method to have more control over how the interpolations points are chosen
function ISDF(prob::BSEProblemExciting, N_μ_vvs::Tuple, N_μ_ccs, N_μ_vcs)
    r_μ_vv_indices = find_r_μ(prob, N_μ_vvs[1], N_μ_vvs[2])
    r_μ_cc_indices = find_r_μ(prob, N_μ_ccs[1], N_μ_ccs[2])
    r_μ_vc_indices = find_r_μ(prob, N_μ_vcs[1], N_μ_vcs[2])

    return ISDF(prob, r_μ_vv_indices, r_μ_cc_indices, r_μ_vc_indices)
end

#TODO: maybe include this in the tests
function read_reference(prob)
    N_v, N_c, N_k = size(prob)
    f = h5open(example_path * "bse_matrix.h5") #TODO: fix this: path is not known here!

    reordered_indices = vec(permutedims(reshape(1:(N_c * N_v * N_k), N_c, N_v, N_k), [2, 1, 3]))

    H_reference = read(f["H_BSE"])[1, reordered_indices, reordered_indices] - im * read(f["H_BSE"])[2, reordered_indices, reordered_indices]

    V_reference_reordered = zeros(Complex{Float64}, N_c * N_v * N_k, N_c * N_v * N_k)
    V_reshaped = reshape(V_reference_reordered, N_c * N_v, N_k, N_c * N_v, N_k)
    n = 1
    for ik in 1:N_k
        for jk in ik:N_k
            V_reshaped[:, ik, :, jk] = read(f, "0001/EXCLI_BSE-BAR_QMT001.OUT/" * lpad(string(n), 6, string(0)))[1, :, :] - im * read(f, "0001/EXCLI_BSE-BAR_QMT001.OUT/" * lpad(string(n), 6, string(0)))[2, :, :]
            n += 1
        end
    end
    V_reference = V_reference_reordered[reordered_indices, reordered_indices]
    for i in 1:size(V_reference, 1)
        V_reference[i, i] = real(V_reference[i, i])
    end

    W_reference_reordered = zeros(Complex{Float64}, N_c * N_v * N_k, N_c * N_v * N_k)
    W_reshaped = reshape(W_reference_reordered, N_c * N_v, N_k, N_c * N_v, N_k)
    n = 1
    for ik in 1:N_k
        for jk in ik:N_k
            W_reshaped[:, ik, :, jk] = read(f, "0001/SCCLI_QMT001.OUT/" * lpad(string(n), 6, string(0)))[1, :, :] - im * read(f, "0001/SCCLI_QMT001.OUT/" * lpad(string(n), 6, string(0)))[2, :, :]
            n += 1
        end
    end
    W_reference = W_reference_reordered[reordered_indices, reordered_indices]
    for i in 1:size(W_reference, 1)
        W_reference[i, i] = real(W_reference[i, i])
    end

    close(f)

    return H_reference, Hermitian(V_reference), Hermitian(W_reference)
end
