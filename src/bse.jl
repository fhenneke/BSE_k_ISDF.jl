# BSE Hamiltonian
using FFTW, LinearMaps, ProgressMeter

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

    V_tilde = assemble_V_tilde(prob, isdf)
    V_workspace = create_V_workspace(prob, isdf)

    u_v_vc_conj = conj.(interpolation_coefficients(isdf)[3])
    u_c_vc = interpolation_coefficients(isdf)[4]

    V = LinearMap{Complex{Float64}}(
        x -> V_times_vector(x, V_tilde, u_v_vc_conj, u_c_vc, V_workspace),
        N_v * N_c * N_k; ishermitian=true)

    return V
end

"""
    assemble_V_tilde(prob, isdf)

Assembles a Matrix representation of ``V`` in the basis given by interpolation vectors of the ISDF.
"""
function assemble_V_tilde(prob, isdf)
    N_rs = size_r(prob)
    N_r = prod(N_rs)
    N_k = size(prob)[3]
    Ω0_vol = Ω0_volume(prob)
    ζ_vc = interpolation_vectors(isdf)[3]

    v_hat = compute_v_hat(prob)

    ζ_vc_hat = zeros(Complex{Float64}, size(ζ_vc))
    for iμ in 1:size(ζ_vc, 2)
        ζ_vc_hat[:, iμ] = Ω0_vol / N_r * vec(fft(reshape(ζ_vc[:, iμ], N_rs)))
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
function create_V_workspace(prob, isdf)
    N_v, N_c, N_k = size(prob)
    N_μ = size(isdf)[3]

    X = complex(zeros(N_v, N_c, N_k))
    B = complex(zeros(N_μ, N_c, N_k))
    C = complex(zeros(N_μ))
    D = complex(zeros(N_μ))
    E = complex(zeros(N_v, N_c, N_k))
    V_workspace = (X, B, C, D, E)

    return V_workspace
end

"""
    setup_W(prob, isdf)

Create a linear operator for the application of `W` to a vector.
The result is a hermitian `LinearMap` of size
`(N_v * N_c * N_K, N_v * N_c * N_k)`.
"""
function setup_W(prob::BSEProblemExciting, isdf)
    N_v, N_c, N_k = size(prob)
    u_v_vv_conj = conj.(interpolation_coefficients(isdf)[1]) #TODO: test whether the conjugation is really necessary for performance
    u_c_cc = interpolation_coefficients(isdf)[2]

    W_tilde_hat = assemble_W_tilde(prob, isdf)
    W_workspace = create_W_workspace(prob, isdf)

    W = LinearMap{Complex{Float64}}(
        x -> W_times_vector(x, W_tilde_hat, u_v_vv_conj, u_c_cc, W_workspace),
        N_v * N_c * N_k; ishermitian=true)

    return W
end

"""
    assemble_W_tilde(prob, isdf)

Assembles a Matrix representation of ``W`` in the basis given by interpolation vectors of the ISDF.
""" #TODO: better dispatch on problem type
function assemble_W_tilde(prob::AbstractBSEProblem, isdf::ISDF)
    w_hat = prob.w_hat
    q_2bz_ind = prob.q_2bz_ind
    q_2bz_shift = prob.q_2bz_shift

    N_rs = size_r(prob)
    N_k = size(prob)[3]
    N_qs = prob.N_k_diffs
    Ω0_vol = Ω0_volume(prob)

    ζ_vv = interpolation_vectors(isdf)[1]
    ζ_cc = interpolation_vectors(isdf)[2]

    return assemble_W_tilde(w_hat, ζ_vv, ζ_cc, Ω0_vol, N_rs, N_k, N_qs, q_2bz_ind, q_2bz_shift)
end
function assemble_W_tilde(w_hat, ζ_vv, ζ_cc, Ω0_vol, N_rs, N_k, N_qs, q_2bz_ind, q_2bz_shift)
    N_r = size(ζ_vv, 1)
    N_ν = size(ζ_vv, 2)
    N_μ = size(ζ_cc, 2)
    N_q = prod(N_qs)

    ζ_vv_hat = zeros(Complex{Float64}, size(ζ_vv))
    for iμ in 1:size(ζ_vv, 2)
        ζ_vv_hat[:, iμ] = Ω0_vol / N_r * vec(fft(reshape(ζ_vv[:, iμ], N_rs)))
    end
    ζ_cc_hat = zeros(Complex{Float64}, size(ζ_cc))
    for iμ in 1:size(ζ_cc, 2)
        ζ_cc_hat[:, iμ] = Ω0_vol / N_r * vec(fft(reshape(ζ_cc[:, iμ], N_rs)))
    end

    W_tilde = complex(zeros(N_qs[1], N_qs[2], N_qs[3], N_μ, N_ν))
    W_tilde_reshaped = reshape(W_tilde, N_q, N_μ, N_ν)

    # TODO: can this be written in terms of more general data structures as W_tilde[iq, :, :] = 1 / (Ω0_vol^2 * N_k) * ζ_cc_hat' * w_hat[iq] * ζ_vv_hat?
    for iq in 1:N_q
        G_shift_indices = vec(mapslices(G -> G_vector_to_index(G, N_rs), w_hat[q_2bz_ind[iq]][2] .+ q_2bz_shift[:, iq]; dims=1))
        W_tilde_reshaped[iq, :, :] = 1 / (Ω0_vol^2 * N_k) *
            ζ_cc_hat[G_shift_indices, :]' * w_hat[q_2bz_ind[iq]][1] * ζ_vv_hat[G_shift_indices, :]
    end

    return fft!(W_tilde, (1, 2, 3))
end

"""
    create_W_workspace(prob, isdf)

Allocates momory for efficiently computing the application of ``W`` to
a vector. The output is a 17-tuple of FFTW plans, arrays of different
dimension and sizes, and tuples.
""" #TODO: better dispatch on problem type
function create_W_workspace(prob::AbstractBSEProblem, isdf::ISDF)
    N_v, N_c, N_k = size(prob)
    N_ks = size_k(prob)
    N_k = prod(N_ks)
    Ω0_vol = Ω0_volume(prob)
    N_qs = prob.N_k_diffs

    N_ν = size(isdf)[1]
    N_μ = size(isdf)[2]

    p = plan_fft(zeros(Complex{Float64}, N_qs); flags = FFTW.PATIENT, timelimit=60)
    p_back = plan_bfft(zeros(Complex{Float64}, N_qs); flags = FFTW.PATIENT, timelimit=60)
    X = zeros(Complex{Float64}, N_v, N_c, N_k)
    B = zeros(Complex{Float64}, N_ν, N_c, N_k)
    C = zeros(Complex{Float64}, N_k, N_μ, N_ν)
    D = zeros(Complex{Float64}, N_k, N_μ, N_ν)
    E = zeros(Complex{Float64}, N_ν, N_c, N_k)
    F = zeros(Complex{Float64}, N_v, N_c, N_k)
    c_reshaped = zeros(Complex{Float64}, N_ks..., Threads.nthreads())
    c_padded = zeros(Complex{Float64}, N_qs..., Threads.nthreads())
    c_transformed = zeros(Complex{Float64}, N_qs..., Threads.nthreads())
    d_reshaped = zeros(Complex{Float64}, N_ks..., Threads.nthreads())
    d_padded = zeros(Complex{Float64}, N_qs..., Threads.nthreads())
    w_tilde_reshaped = zeros(Complex{Float64}, N_qs..., Threads.nthreads())
    w_tilde_hat = zeros(Complex{Float64}, N_qs..., Threads.nthreads())
    W_workspace = (p, p_back, X, B, C, D, E, F, c_reshaped, c_padded, d_reshaped, d_padded, c_transformed, w_tilde_reshaped, w_tilde_hat, N_ks, N_qs)

    return W_workspace
end

# MIT licensed via julia
"""
    w_conv!(b, w, a, ap, bp, cp, w_hat, p, p_back)

Compute the convolution of `w` and `a` and store it in `b`. This means

``b_i = \\sum_j a_j w_{i - j + 1}``

with periodic `w`. The function uses Fourier transforms in the form
of FFTW forward and backward plans `p` and `p_back`. The array `a` is
zero padded to the size of `w`.
"""
function w_conv!(b, w, a, ap, bp, cp, w_hat, p, p_back)
    dim = ndims(a)
    sa = size(a)
    sw = size(w)

    indices = CartesianIndices(sa)

    copyto!(ap, indices, a, indices)

    mul!(cp, p, ap)
    # mul!(w_hat, p, w)
    cp .*= (1 / length(w)) .* w_hat

    mul!(bp, p_back, cp)

    copyto!(b, indices, bp, indices)

    return b
end

"""
    V_times_vector(x, V_tilde, u_v_vc_conj, u_c_vc, V_workspace)

Compute the product of ``V`` and `x`.
"""
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

"""
    W_times_vector(x, W_tilde_hat, u_v_vv_conj, u_c_cc, W_workspace)

Compute the product of ``W`` and `x`.
"""
function W_times_vector(x, W_tilde_hat, u_v_vv_conj, u_c_cc, W_workspace)
    N_v = size(u_v_vv_conj, 2)
    N_c = size(u_c_cc, 2)
    N_k = size(u_v_vv_conj, 3)
    N_ν = size(u_v_vv_conj, 1)
    N_μ = size(u_c_cc, 1)
    p, p_back, X, B, C, D, E, F, c_reshaped_threads, c_padded_threads, d_reshaped_threads, d_padded_threads, c_transformed_threads, w_tilde_reshaped_threads, w_tilde_hat_threads, N_ks, N_qs = W_workspace

    X[:] .= x
    Threads.@threads for jk in 1:N_k
        mul!(@view(B[:, :, jk]), @view(u_v_vv_conj[:, :, jk]), @view(X[:, :, jk]))
        mul!(@view(C[jk, :, :]), @view(u_c_cc[:, :, jk]), transpose(@view(B[:, :, jk])))
    end
    Threads.@threads for iν in 1:N_ν
        @views for iμ in 1:N_μ
            c_reshaped_threads[:, :, :, Threads.threadid()] .= reshape(C[:, iμ, iν], N_ks)
            w_tilde_hat_threads[:, :, :, Threads.threadid()] .= W_tilde_hat[:, :, :, iμ, iν]
            w_conv!(d_reshaped_threads[:, :, :, Threads.threadid()], w_tilde_reshaped_threads[:, :, :, Threads.threadid()], c_reshaped_threads[:, :, :, Threads.threadid()], c_padded_threads[:, :, :, Threads.threadid()], d_padded_threads[:, :, :, Threads.threadid()], c_transformed_threads[:, :, :, Threads.threadid()], w_tilde_hat_threads[:, :, :, Threads.threadid()], p, p_back)
            D[:, iμ, iν] .= vec(d_reshaped_threads[:, :, :, Threads.threadid()])
        end
    end
    Threads.@threads for ik in 1:N_k
        mul!(@view(E[:, :, ik]), transpose(@view(D[ik, :, :])), conj.(@view(u_c_cc[:, :, ik])))
        mul!(@view(F[:, :, ik]), adjoint(@view(u_v_vv_conj[:, :, ik])), @view(E[:, :, ik]))
    end

    return copy(vec(F))
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
    return D + 2 * V - W # TODO: add scissor shift here?
end

###############################################################################

# Some methods which are only used for consitency checks in the tests.

"""
    V_entry(v_hat, iv, ic, ik, jv, jc, jk, u_v, u_c, Ω0_vol, N_rs, N_ks)

Compute the matrix entry in ``V`` corresponding to the multiindex
`(iv ic ik, jv jc jk)`.
"""
function V_entry(v_hat, iv, ic, ik, jv, jc, jk, u_v, u_c, Ω0_vol, N_rs, N_ks)
    N_r = prod(N_rs)
    N_k = prod(N_ks)

    U_vc_1_hat = Ω0_vol / N_r * vec(fft(reshape(u_c[:, ic, ik] .* conj.(u_v[:, iv, ik]), N_rs)))
    U_vc_2_hat = Ω0_vol / N_r * vec(fft(reshape(u_c[:, jc, jk] .* conj.(u_v[:, jv, jk]), N_rs)))

    v = 1 / (Ω0_vol * N_k) * U_vc_1_hat' * (v_hat .* U_vc_2_hat)

    return v
end

"""
    assemble_exact_V(prob)

Assemble the full matrix for ``V``. The entries are computed using the function `V_entry`.
This should only be used for very small problems.
"""
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

"""
    W_entry(w_hat, iv, ic, ik, jv, jc, jk, u_v, u_c, Ω0_vol, N_rs, ikkp2iq, q_2bz_ind, q_2bz_shift)

Compute the matrix entry in ``V`` corresponding to the multiindex
`(iv ic ik, jv jc jk)`. This method is not effient for large problems
since `ikkp2iq` is quadratic in `N_k`.
"""
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

"""
    assemble_exact_W(prob)

Assemble the full matrix for ``W``. The entries are computed using the function `W_entry`.
This should only be used for very small problems.
"""
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

"""
    ikkp2iq_matrix(k_bz, q_2bz)

Compute the all indices in `q_bz` corresponding to differences of points in `k_bz`. The resulting matrix is such that the entry `l = ikkp2iq[i, j]` satisfies `k_bz[i] - k_bz[j] == q_2bz[l]`.

This method is not effient for large problems since it is quadratic in `N_k`.
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
