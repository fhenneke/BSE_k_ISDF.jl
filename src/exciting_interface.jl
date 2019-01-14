# code for interoperability with exciting

import DelimitedFiles: readdlm
import HDF5: h5open, readmmap
import LightXML: parse_file, root, attribute
import Random: MersenneTwister, randsubseq

struct BSEProblemExciting
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

    return BSEProblemExciting(input_xml, N_rs, r_lattice, r_cartesian, a_mat, Ω0_vol, atoms, N_ks, k_bz, b_mat, BZ_vol, E_v, E_c, u_v, u_c, N_k_diffs, q_bz, q_2bz, q_2bz_ind, q_2bz_shift, w_hat)
end

function read_r_points_k_points(N_rs, path) # TODO: make into type constructor
    N_unit = prod(N_rs)

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
    N_unit = prod(N_rs)

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


    u_v = zeros(Complex{Float64}, N_unit, N_v, N_k)

    file = h5open(path * "/bse_output.h5", "r")
    for ik in 1:N_k
        phase = kron(
            exp.(-2 * pi * im * range(0.0, stop = 1.0, length = N_rs[3] + 1)[1:N_rs[3]] .* k_bz[3, ik]),
            exp.(-2 * pi * im * range(0.0, stop = 1.0, length = N_rs[2] + 1)[1:N_rs[2]] .* k_bz[2, ik]),
            exp.(-2 * pi * im * range(0.0, stop = 1.0, length = N_rs[1] + 1)[1:N_rs[1]] .* k_bz[1, ik]))
        for iv in 1:N_v
            c = read(file, lpad(string(N_core + iv), 4, string(0)) * "/" * lpad(string(ik), 4, string(0)) * "/data")

            u_v[:, iv, ik] .= (c[1, mask_boundary] .+ im * c[2, mask_boundary]) .* phase
        end
    end

    u_c = zeros(Complex{Float64}, N_unit, N_c, N_k)
    for ik in 1:N_k
        phase = kron(
            exp.(-2 * pi * im * range(0.0, stop = 1.0, length = N_rs[3] + 1)[1:N_rs[3]] .* k_bz[3, ik]),
            exp.(-2 * pi * im * range(0.0, stop = 1.0, length = N_rs[2] + 1)[1:N_rs[2]] .* k_bz[2, ik]),
            exp.(-2 * pi * im * range(0.0, stop = 1.0, length = N_rs[1] + 1)[1:N_rs[1]] .* k_bz[1, ik]))
        for ic in 1:N_c
            c = read(file, lpad(string(N_core + N_v + ic), 4, string(0)) * "/" * lpad(string(ik), 4, string(0)) * "/data")

            u_c[:, ic, ik] .= (c[1, mask_boundary] .+ im * c[2, mask_boundary]) .* phase
        end
    end
    close(file)

    return E_v, E_c, u_v, u_c
end

function read_q_points_screenedcoulomb(N_ks, Ω0_vol, path)
    q_points_out = readdlm(path * "/QPOINTS_SCR.OUT")
    q_bz = float.(transpose(q_points_out[2:end, 2:4]))

    N_k_diffs = 2 .* N_ks

    # alternative q grid [0, 1[ ∪ [-1, 0[
    q_grid_1 = vcat(range(0, stop = 1, length = N_ks[1] + 1)[1:(end - 1)], range(-1, stop = 0, length = N_ks[1] + 1)[1:(end - 1)])
    q_grid_2 = vcat(range(0, stop = 1, length = N_ks[2] + 1)[1:(end - 1)], range(-1, stop = 0, length = N_ks[2] + 1)[1:(end - 1)])
    q_grid_3 = vcat(range(0, stop = 1, length = N_ks[3] + 1)[1:(end - 1)], range(-1, stop = 0, length = N_ks[3] + 1)[1:(end - 1)])
    q_2bz = transpose(hcat(kron(ones(2 * N_ks[3]), ones(2 * N_ks[2]), q_grid_1),
                           kron(ones(2 * N_ks[3]), q_grid_2, ones(2 * N_ks[1])),
                           kron(q_grid_3, ones(2 * N_ks[2]), ones(2 * N_ks[1])))
                     )
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
        wqq_complex = Ω0_vol * (wqq[1, 1:G_len, 1:G_len] + im * wqq[2, 1:G_len, 1:G_len])
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
        pmat_reshaped[:, :, ik, :] = pmat_block[(N_core + 1):(N_core + N_v), (N_core + N_v + 1):(N_core + N_v + N_c), :, 1] +
                                im * pmat_block[(N_core + 1):(N_core + N_v), (N_core + N_v + 1):(N_core + N_v + N_c), :, 2]
    end
    close(file)

    return pmat
end

function find_r_μ(prob::BSEProblemExciting, N_μ_mt, N_μ_irs)
    # uniform grid (interstitial region and muffin tin region)
    grid_1 = find_r_μ(prob.N_rs[1], N_μ_irs[1])
    grid_2 = find_r_μ(prob.N_rs[2], N_μ_irs[2])
    grid_3 = find_r_μ(prob.N_rs[3], N_μ_irs[3])
    r_μ_mask_1 = zeros(prob.N_rs[1])
    r_μ_mask_2 = zeros(prob.N_rs[2])
    r_μ_mask_3 = zeros(prob.N_rs[3])
    r_μ_mask_1[grid_1] .= 1.0
    r_μ_mask_2[grid_2] .= 1.0
    r_μ_mask_3[grid_3] .= 1.0
    r_μ_mask = Bool.(kron(r_μ_mask_3, r_μ_mask_2, r_μ_mask_1))

    # aditional points in muffin tin regions
    N_r = prod(prob.N_rs)
    r_mask_mt = zeros(Bool, N_r)
    for atom in prob.atoms
        for ir in 1:N_r
            for is in -1:1, js in -1:1, ms in -1:1
                if norm(prob.r_cartesian[:, ir] - prob.a_mat * (atom[:position] + [is, js, ms])) < atom[:mt_radius]
                    r_mask_mt[ir] = true
                end
            end
        end
    end

    rng = MersenneTwister(0) # for reproducibility

    r_μ_indices_mt = randsubseq(rng, findall(r_mask_mt), N_μ_mt / length(findall(r_mask_mt)))
    r_μ_mask[r_μ_indices_mt] .= true

    r_μ_indices = findall(r_μ_mask)

    return r_μ_indices
end

function ISDF(prob::BSEProblemExciting, N_μ_mt, N_μ_irs)
    r_μ_indices = find_r_μ(prob, N_μ_mt, N_μ_irs)

    return ISDF(r_μ_indices, r_μ_indices, r_μ_indices, prob.u_v, prob.u_c)
end

"""
input: 2d array of k points (size (d, N_k))
       2d array of k point differences (size (d, 2^d * N_k))

output: 2d array of integers

    resultgin matrix such that the entry l = ikkp2iq[i, j] satisfies k_i - k_j == q_l

    not linear in the number of k points, only used for the consistency check
    which use the elementwise W
"""
function ikkp2iq_matrix(k_bz, q_2bz)
    N_k = size(k_bz, 2)
    ikkp2iq = zeros(Int, N_k, N_k)
    for ik in 1:N_k
        for jk in 1:N_k
            k_diff = k_bz[:, ik] - k_bz[:, jk]
            ikkp2iq[ik, jk] = findfirst(vec(prod(abs.(k_diff .- q_2bz) .< 1e-3; dims=1)))
        end
    end
    return ikkp2iq
end
