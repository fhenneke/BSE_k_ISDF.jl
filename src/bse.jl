# BSE Hamiltonian
using FFTW, LinearMaps, ProgressMeter

# only for consitency checks

function V_entry(v_hat, iv, ic, ik, jv, jc, jk, u_v, u_c, Ω0_vol, N_rs, N_ks)
    N_r = prod(N_rs)
    N_k = prod(N_ks)

    U_vc_1_hat = Ω0_vol / N_r * vec(fft(reshape(u_c[:, ic, ik] .* conj.(u_v[:, iv, ik]), N_rs)))
    U_vc_2_hat = Ω0_vol / N_r * vec(fft(reshape(u_c[:, jc, jk] .* conj.(u_v[:, jv, jk]), N_rs)))

    v = 1 / (Ω0_vol * N_k) * U_vc_1_hat' * (v_hat .* U_vc_2_hat)

    return v
end

function assemble_exact_V(prob)
    N_v, N_c, N_k = size(prob)
    N_rs = size_r(prob)
    N_ks = size_k(prob)
    Ω0_vol = Ω0_volume(prob)
    u_v, u_c = orbitals(prob)

    v_hat = compute_v_hat(prob)

    V_reshaped = zeros(Complex{Float64}, N_v, N_c, N_k, N_v, N_c, N_k)
    @showprogress 1 "Assemble exact V ..." for jk in 1:N_k, jc in 1:N_c, jv in 1:N_v, ik in 1:N_k, ic in 1:N_c, iv in 1:N_v
        V_reshaped[iv, ic, ik, jv, jc, jk] =
            V_entry(v_hat, iv, ic, ik, jv, jc, jk, u_v, u_c, Ω0_vol, N_rs, N_ks)
    end
    V = reshape(V_reshaped, N_v * N_c * N_k, N_v * N_c * N_k)
    for i in 1:(N_v * N_c * N_k)
        V[i, i] = real(V[i, i])
    end

    return V
end

function W_entry(w_hat, iv, ic, ik, jv, jc, jk, u_v, u_c, Ω0_vol, N_rs, ikkp2iq, q_2bz_ind, q_2bz_shift)
    N_r = prod(N_rs)
    N_k = size(ikkp2iq, 1)

    iq = ikkp2iq[ik, jk]

    U_c_hat = Ω0_vol / N_r * vec(fft(reshape(u_c[:, ic, ik] .* conj.(u_c[:, jc, jk]), N_rs)))
    U_v_hat = Ω0_vol / N_r * vec(fft(reshape(u_v[:, iv, ik] .* conj.(u_v[:, jv, jk]), N_rs)))

    G_shift_indices = vec(mapslices(G -> G_vector_to_index(G, N_rs), w_hat[q_2bz_ind[iq]][2] .+ q_2bz_shift[:, iq]; dims=1))
    w = 1 / (Ω0_vol^2 * N_k) *
        U_c_hat[G_shift_indices]' * (w_hat[q_2bz_ind[iq]][1] * U_v_hat[G_shift_indices])

    return w
end

function assemble_exact_W(prob)
    N_v, N_c, N_k = size(prob)
    N_rs = size_r(prob)
    N_ks = size_k(prob)
    Ω0_vol = Ω0_volume(prob)
    u_v, u_c = orbitals(prob)

    w_hat = prob.w_hat
    ikkp2iq = ikkp2iq_matrix(prob.k_bz, prob.q_2bz)
    q_2bz_ind = prob.q_2bz_ind
    q_2bz_shift = prob.q_2bz_shift

    W_reshaped = zeros(Complex{Float64}, N_v, N_c, N_k, N_v, N_c, N_k)
    @showprogress 1 "Assemble exact W ..." for jk in 1:N_k, jc in 1:N_c, jv in 1:N_v, ik in 1:N_k, ic in 1:N_c, iv in 1:N_v
        W_reshaped[iv, ic, ik, jv, jc, jk] =
            W_entry(w_hat, iv, ic, ik, jv, jc, jk, u_v, u_c, Ω0_vol, N_rs, ikkp2iq, q_2bz_ind, q_2bz_shift)
    end
    W = reshape(W_reshaped, N_v * N_c * N_k, N_v * N_c * N_k)
    for i in 1:(N_v * N_c * N_k)
        W[i, i] = real(W[i, i])
    end

    return W
end

###############################################################################

# methods for setting up the matrix free Hamiltonian
"""
    setup_D(prob)

Assemble a sparse matrix of size `(N_v * N_c * N_K, N_v * N_c * N_k)`
for the diagonal part of the BSE Hamiltonian. The entries on the
diagonal are given by the single particle energy differences.
"""
function setup_D(prob)
    E_v, E_c = energies(prob)
    N_v, N_c, N_k = size(prob)

    D = spdiagm(0 => vec([E_c[ic, ik] - E_v[iv, ik] for iv in 1:N_v, ic in 1:N_c, ik in 1:N_k]))
    return D
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

"""
    setup_V(prob, isdf)

Create a linear operator for the application of `V` to a vector.
The result is a hermitian `LinearMap` of size
`(N_v * N_c * N_K, N_v * N_c * N_k)`.
"""
function setup_V(prob::BSEProblemExciting, isdf::ISDF)
    N_rs = size_r(prob)
    N_v, N_c, N_k = size(prob)
    Ω0_vol = Ω0_volume(prob)

    N_μ = size(isdf)[3]
    ζ_vc = interpolation_vectors(isdf)[3]

    v_hat = compute_v_hat(prob)

    V_tilde = assemble_V_tilde(v_hat, ζ_vc, Ω0_vol, N_rs, N_k)
    V_workspace = create_V_workspace(N_v, N_c, N_k, N_μ)

    u_v_vc_conj = conj.(interpolation_coefficients(isdf)[3])
    u_c_vc = interpolation_coefficients(isdf)[4]

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

"""
    assemble_V_tilde(v_hat, ζ_vc, Ω0_vol, N_rs, N_k)

Assembles a Matrix representation of ``V`` in the basis given by interpolation vectors of the ISDF.
"""
function assemble_V_tilde(v_hat, ζ_vc, Ω0_vol, N_rs, N_k)
    N_r = size(ζ_vc, 1)

    ζ_vc_hat = zeros(Complex{Float64}, size(ζ_vc))
    for iμ in 1:size(ζ_vc, 2)
        ζ_vc_hat[:, iμ] = Ω0_vol / N_r * vec(fft(reshape(ζ_vc[:, iμ], N_r)))
    end

    V_tilde = 1 / (Ω0_vol * N_k) * ζ_vc_hat' * (v_hat .* ζ_vc_hat)

    return V_tilde
end

"""
    create_V_workspace(N_v, N_c, N_k, N_μ)

Allocates momory for efficiently computing the application of ``V`` to
a vector. The output is a 5-tuple of arrays of different dimension and
sizes.
"""
function create_V_workspace(N_v, N_c, N_k, N_μ)
    X = complex(zeros(N_v, N_c, N_k))
    B = complex(zeros(N_μ, N_c, N_k))
    C = complex(zeros(N_μ))
    D = complex(zeros(N_μ))
    E = complex(zeros(N_v, N_c, N_k))
    V_workspace = (X, B, C, D, E)

    return V_workspace
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

"""
    setup_W(prob, isdf)

Create a linear operator for the application of `W` to a vector.
The result is a hermitian `LinearMap` of size
`(N_v * N_c * N_K, N_v * N_c * N_k)`.
"""
function setup_W(prob::BSEProblemExciting, isdf)
    w_hat = prob.w_hat
    q_2bz_ind = prob.q_2bz_ind
    q_2bz_shift = prob.q_2bz_shift

    N_rs = size_r(prob)
    N_v, N_c, N_k = size(prob)
    N_ks = size_k(prob)
    Ω0_vol = Ω0_volume(prob)
    N_k_diffs = prob.N_k_diffs

    N_ν = size(isdf)[1]
    N_μ = size(isdf)[2]
    ζ_vv = interpolation_vectors(isdf)[1]
    ζ_cc = interpolation_vectors(isdf)[2]
    u_v_vv_conj = conj.(interpolation_coefficients(isdf)[1]) #TODO: test whether the conjugation is really necessary for performance
    u_c_cc = interpolation_coefficients(isdf)[2]

    W_tilde = assemble_W_tilde(w_hat, ζ_vv, ζ_cc, Ω0_vol, N_rs, N_k, q_2bz_ind, q_2bz_shift)
    W_workspace = create_W_workspace(N_v, N_c, N_ks, N_k_diffs, N_ν, N_μ)

    W = LinearMap{Complex{Float64}}(
        x -> W_times_vector(x, W_tilde, u_v_vv_conj, u_c_cc, W_workspace),
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

"""
    G_vector_to_index(G, N_rs)

Return the index `i` such that `fftfreq(N_rs...)[:, i] == G`.
"""
function G_vector_to_index(G, N_rs) # TODO: make fast for all dimensions
    if length(G) == 3
        ind = mod1(G[1] + 1, N_rs[1]) + N_rs[1] * (mod1(G[2] + 1, N_rs[2]) - 1) + N_rs[1] * N_rs[2] * (mod1(G[3] + 1, N_rs[3]) - 1)
    else
        ind = mod1(G[1] + 1, N_rs[1])
        for id in 2:length(G)
            ind += prod(N_rs[1:(id - 1)]) * (mod1(G[id] + 1, N_rs[id]) - 1)
        end
    end
    return ind
end

"""
    assemble_W_tildew_hat, ζ_vv, ζ_cc, Ω0_vol, N_rs, N_k, q_2bz_ind, q_2bz_shift)

Assembles a Matrix representation of ``W`` in the basis given by interpolation vectors of the ISDF.
"""
function assemble_W_tilde(w_hat, ζ_vv, ζ_cc, Ω0_vol, N_rs, N_k, q_2bz_ind, q_2bz_shift)
    N_r = size(ζ_vv, 1)
    N_ν = size(ζ_vv, 2)
    N_μ = size(ζ_cc, 2)
    N_q = length(q_2bz_ind)

    ζ_vv_hat = zeros(Complex{Float64}, size(ζ_vv))
    for iμ in 1:size(ζ_vv, 2)
        ζ_vv_hat[:, iμ] = Ω0_vol / N_r * vec(fft(reshape(ζ_vv[:, iμ], N_rs)))
    end
    ζ_cc_hat = zeros(Complex{Float64}, size(ζ_cc))
    for iμ in 1:size(ζ_cc, 2)
        ζ_cc_hat[:, iμ] = Ω0_vol / N_r * vec(fft(reshape(ζ_cc[:, iμ], N_rs)))
    end

    W_tilde = complex(zeros(N_q, N_μ, N_ν))

    # TODO: can this be written in terms of more general data structures as W_tilde[iq, :, :] = 1 / (Ω0_vol^2 * N_k) * ζ_cc_hat' * w_hat[iq] * ζ_vv_hat?
    for iq in 1:N_q
        G_shift_indices = vec(mapslices(G -> G_vector_to_index(G, N_rs), w_hat[q_2bz_ind[iq]][2] .+ q_2bz_shift[:, iq]; dims=1))
        W_tilde[iq, :, :] = 1 / (Ω0_vol^2 * N_k) *
            ζ_cc_hat[G_shift_indices, :]' * w_hat[q_2bz_ind[iq]][1] * ζ_vv_hat[G_shift_indices, :]
    end

    return W_tilde
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

function create_W_workspace(N_v, N_c, N_ks, N_qs, N_ν, N_μ)
    N_k = prod(N_ks)
    p = plan_fft(zeros(Complex{Float64}, N_qs); flags = FFTW.PATIENT, timelimit=60)
    p_back = plan_bfft(zeros(Complex{Float64}, N_qs); flags = FFTW.PATIENT, timelimit=60)
    X = complex(zeros(N_v, N_c, N_k))
    B = complex(zeros(N_ν, N_c, N_k))
    C = complex(zeros(N_k, N_μ, N_ν))
    D = complex(zeros(N_k, N_μ, N_ν))
    E = complex(zeros(N_ν, N_c, N_k))
    F = complex(zeros(N_v, N_c, N_k))
    c_reshaped = complex(zeros(N_ks))
    c_padded = complex(zeros(N_qs))
    c_transformed = complex(zeros(N_qs))
    d_reshaped = complex(zeros(N_ks))
    d_padded = complex(zeros(N_qs))
    w_tilde_reshaped = complex(zeros(N_qs))
    w_tilde_hat = complex(zeros(N_qs))
    W_workspace = (p, p_back, X, B, C, D, E, F, c_reshaped, c_padded, d_reshaped, d_padded, c_transformed, w_tilde_reshaped, w_tilde_hat, N_ks, N_qs)

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

# MIT licensed via julia
function w_conv!(b, w, a, ap, bp, cp, w_hat, p, p_back)
    dim = ndims(a)
    sa = size(a)
    sw = size(w)

    all(sa .<= sw) || throw(ArgumentError("size of w needs to be larger than the size of a"))
    size(ap) == sw || throw(ArgumentError("size of a buffer needs to equal the size of w"))
    size(bp) == sw || throw(ArgumentError("size of b buffer needs to equal the size of w"))
    size(w_hat) == sw || throw(ArgumentError("size of w_hat buffer needs to equal the size of w"))

    indices = CartesianIndices(sa)

    copyto!(ap, indices, a, indices)

    mul!(cp, p, ap)
    mul!(w_hat, p, w)
    cp .*= (1 / length(w)) .* w_hat

    mul!(bp, p_back, cp)

    copyto!(b, indices, bp, indices)

    return b
end
function w_conv!(b, w, a)
    return w_conv!(b, w, a, zeros(w), zeros(w), zeros(w), zeros(w), plan_fft(zeros(w)), plan_bfft(zeros(w)))
end
function w_conv(w, a)
    return w_conv!(similar(a), w, a)
end

# TODO what to do with the reference implementation? move to interface code?
function w_conv_reference!(b, w, a)
    dim = ndims(a)
    sa = size(a)
    sw = size(w)

    all(sa .<= sw) || throw(ArgumentError("size of w needs to be larger than the size of a"))

    for i in CartesianIndices(sa)
        b[i] = zero(eltype(b))
        for j in CartesianIndices(sa)
            b[i] += w[CartesianIndex(mod1.((i - j).I .+ 1, sw))] * a[j]
        end
    end

    return b
end
function w_conv_reference(w, a)
    b = similar(a)
    return w_conv_reference!(b, w, a)
end

# matrix free V
function V_times_vector(x, V_tilde, u_v_vc_conj, u_c_vc, V_workspace)
    N_μ = size(V_tilde, 1)
    N_k = size(u_v_vc_conj, 3)
    N_v = size(u_v_vc_conj, 2)
    N_c = size(u_c_vc, 2)
    X, B, C, D, E = V_workspace

    X[:] .= x
    @views for jk in 1:N_k
        mul!(B[:, :, jk], u_v_vc_conj[:, :, jk], X[:, :, jk])
    end
    for jr in 1:N_μ
        C[jr] = 0.0
        for jc in 1:N_c
            for jk in 1:N_k
                C[jr] += u_c_vc[jr, jc, jk] * B[jr, jc, jk]
            end
        end
    end
    mul!(D, V_tilde, C)
    for ik in 1:N_k
        for ic in 1:N_c
            for iv in 1:N_v
                E[iv, ic, ik] = 0.
                for iμ in 1:N_μ
                    E[iv, ic, ik] += conj(u_c_vc[iμ, ic, ik] * u_v_vc_conj[iμ, iv, ik]) * D[iμ]
                end
            end
        end
    end

    return vec(E)
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

function W_times_vector(x, W_tilde, u_v_vv_conj, u_c_cc, W_workspace)
    N_v = size(u_v_vv_conj, 2)
    N_c = size(u_c_cc, 2)
    N_k = size(u_v_vv_conj, 3)
    N_ν = size(u_v_vv_conj, 1)
    N_μ = size(u_c_cc, 1)
    p, p_back, X, B, C, D, E, F, c_reshaped, c_padded, d_reshaped, d_padded, c_transformed, w_tilde_reshaped, w_tilde_hat, N_ks, N_qs = W_workspace

    X[:] .= x
    @views for jk in 1:N_k
        mul!(B[:, :, jk], u_v_vv_conj[:, :, jk], X[:, :, jk])
        mul!(C[jk, :, :], u_c_cc[:, :, jk], transpose(B[:, :, jk]))
    end
    for iμ in 1:N_μ
        for iν in 1:N_ν
            c_reshaped[:] .= @view(C[:, iμ, iν])
            w_tilde_reshaped[:] .= @view(W_tilde[:, iμ, iν])
            w_conv!(d_reshaped, w_tilde_reshaped, c_reshaped, c_padded, d_padded, c_transformed, w_tilde_hat, p, p_back)
            D[:, iμ, iν] .= @view(d_reshaped[:])
        end
    end
    @views for ik in 1:N_k
        mul!(E[:, :, ik], transpose(D[ik, :, :]), conj.(u_c_cc[:, :, ik]))
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

"""
    setup_H(prob, isdf)

Create a linear operator for the application of ``H = D + 2 V - W``
to a vector. The result is a hermitian `LinearMap` of size
`(N_v * N_c * N_K, N_v * N_c * N_k)`.
"""
function setup_H(prob, isdf)
    D = setup_D(prob)
    V = setup_V(prob, isdf)
    W = setup_W(prob, isdf)
    return D + 2 * V - W
end
