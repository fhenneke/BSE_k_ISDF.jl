# BSE Hamiltonian
using FFTW, LinearMaps

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

# only for consitency checks
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

function assemble_exact_V(prob)
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

function W_entry(W, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit, k_bz)
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

function W_entry_3d(w_hat, iv, ic, ik, jv, jc, jk, u_v, u_c, Ω0_vol, N_rs, ikkp2iq, q_2bz_ind, q_2bz_shift)
    N_unit = prod(N_rs)
    N_k = size(ikkp2iq, 1)

    iq = ikkp2iq[ik, jk]

    U_c_hat = Ω0_vol / N_unit * vec(fft(reshape(u_c[:, ic, ik] .* conj.(u_c[:, jc, jk]), N_rs)))
    U_v_hat = Ω0_vol / N_unit * vec(fft(reshape(u_v[:, iv, ik] .* conj.(u_v[:, jv, jk]), N_rs)))

    G_shift_indices = vec(mapslices(G -> G_vector_to_index(G, N_rs), w_hat[q_2bz_ind[iq]][2] .+ q_2bz_shift[:, iq]; dims=1))
    w = 1 / (Ω0_vol^2 * N_k) *
        U_c_hat[G_shift_indices]' * (w_hat[q_2bz_ind[iq]][1] * U_v_hat[G_shift_indices])

    return w
end

function assemble_exact_W(prob)
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

function H_entry(V, W, iv, ic, ik, jv, jc, jk, E_v, E_c, u_v, u_c, r_super, r_unit, k_bz)
    (E_c[ic, ik] - E_v[iv, ik]) * (iv == jv) * (ic == jc) * (ik == jk) +
        2 * V_entry(V, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit) -
        W_entry(W, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit, k_bz)
end

function H_entry_fast(v_hat, w_hat, iv, ic, ik, jv, jc, jk, E_v, E_c, u_v, u_c, r_super, r_unit, k_bz)
    (E_c[ic, ik] - E_v[iv, ik]) * (iv == jv) * (ic == jc) * (ik == jk) +
        2 * V_entry_fast(v_hat, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit) -
        W_entry_fast(w_hat, iv, ic, ik, jv, jc, jk, u_v, u_c, r_super, r_unit, k_bz)
end

function assemble_exact_H(prob)
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

###############################################################################

# methods for setting up the matrix free Hamiltonian
function setup_D(prob)
    E_v, E_c = prob.E_v, prob.E_c
    N_v = size(E_v, 1)
    N_c = size(E_c, 1)
    N_k = size(E_c, 2)

    D = spdiagm(0 => vec([E_c[ic, ik] - E_v[iv, ik] for iv in 1:N_v, ic in 1:N_c, ik in 1:N_k]))
    return D
end

function setup_V(prob::BSEProblem1D, isdf)
    v_hat = prob.v_hat
    r_super = prob.prob.r_super
    r_unit = prob.prob.r_unit
    N_v = prob.N_v
    N_c = prob.N_c
    N_k = prob.N_k
    N_μ = isdf.N_μ_vc
    ζ_vc = isdf.ζ_vc

    V_tilde = assemble_V_tilde(v_hat[:, 1], ζ_vc, r_super, r_unit)
    V_workspace = create_V_workspace(N_v, N_c, N_k, N_μ)

    u_v_vc_conj = conj.(isdf.u_v_vc)
    u_c_vc = isdf.u_c_vc

    V = LinearMap{Complex{Float64}}(
        x -> V_times_vector(x, V_tilde, u_v_vc_conj, u_c_vc, V_workspace),
        N_v * N_c * N_k; ishermitian=true)

    return V
end

function setup_V(prob::BSEProblemExciting, isdf)
    N_rs = prob.N_rs
    N_v = size(prob.E_v, 1)
    N_c = size(prob.E_c, 1)
    N_k = prod(prob.N_ks)
    Ω0_vol = prob.Ω0_vol

    N_μ = isdf.N_μ_vc
    ζ_vc = isdf.ζ_vc

    v_hat = 4 * pi * vec(mapslices(x -> norm(x) < 1e-10 ? 0. : 1 / norm(x)^2, fftfreq(N_rs...); dims = 1))

    V_tilde = assemble_V_tilde3d(v_hat, ζ_vc, Ω0_vol, N_rs[1], N_rs[2], N_rs[3], N_k)
    V_workspace = create_V_workspace(N_v, N_c, N_k, N_μ)

    u_v_vc_conj = conj.(isdf.u_v_vc)
    u_c_vc = isdf.u_c_vc

    V = LinearMap{Complex{Float64}}(
        x -> V_times_vector(x, V_tilde, u_v_vc_conj, u_c_vc, V_workspace),
        N_v * N_c * N_k; ishermitian=true)

    return V
end

function assemble_V_tilde(v_hat, ζ_vc, r_super, r_unit)
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

function assemble_V_tilde3d(v_hat, ζ_vc, Ω0_vol, N_r_1, N_r_2, N_r_3, N_k)
    N_unit = size(ζ_vc, 1)

    ζ_vc_hat = zeros(Complex{Float64}, size(ζ_vc))
    for iμ in 1:size(ζ_vc, 2)
        ζ_vc_hat[:, iμ] = Ω0_vol / N_unit * vec(fft(reshape(ζ_vc[:, iμ], N_r_1, N_r_2, N_r_3)))
    end

    V_tilde = 1 / (Ω0_vol * N_k) * ζ_vc_hat' * (v_hat .* ζ_vc_hat)

    return V_tilde
end

function create_V_workspace(N_v, N_c, N_k, N_μ)
    X = complex(zeros(N_v, N_c, N_k))
    B = complex(zeros(N_μ, N_c, N_k))
    C = complex(zeros(N_μ))
    D = complex(zeros(N_μ))
    E = complex(zeros(N_v, N_c, N_k))
    V_workspace = (X, B, C, D, E)

    return V_workspace
end

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

    W_tilde = assemble_W_tilde(w_hat, ζ_vv, ζ_cc, r_super, r_unit, k_bz)
    W_workspace = create_W_workspace(N_v, N_c, N_k, N_ν, N_μ)
    W_tilde_hat = fft(W_tilde, 1)


    W = LinearMap{Complex{Float64}}(
        x -> W_times_vector(x, W_tilde_hat, u_v_vv_conj, u_c_cc, W_workspace),
        N_v * N_c * N_k; ishermitian=true)

    return W
end

function setup_W(prob::BSEProblemExciting, isdf)
    w_hat = prob.w_hat
    q_2bz_ind = prob.q_2bz_ind
    q_2bz_shift = prob.q_2bz_shift

    N_rs = prob.N_rs
    N_v = size(prob.E_v, 1)
    N_c = size(prob.E_c, 1)
    N_ks = prob.N_ks
    N_k = prod(N_ks)
    k_bz = prob.k_bz
    Ω0_vol = prob.Ω0_vol
    N_k_diffs = prob.N_k_diffs

    N_μ = isdf.N_μ_cc
    N_ν = isdf.N_μ_vv
    ζ_vv = isdf.ζ_vv
    ζ_cc = isdf.ζ_cc
    u_v_vv_conj = conj.(isdf.u_v_vv)
    u_c_cc = isdf.u_c_cc

    W_tilde = assemble_W_tilde3d(w_hat, ζ_vv, ζ_cc, Ω0_vol, N_rs, N_k, q_2bz_ind, q_2bz_shift)
    W_workspace = create_W_workspace3d(N_v, N_c, N_ks, N_k_diffs, N_μ, N_ν)

    W = LinearMap{Complex{Float64}}(
        x -> W_times_vector3d(x, W_tilde, u_v_vv_conj, u_c_cc, W_workspace),
        N_v * N_c * N_k; ishermitian=true)

    return W
end


function assemble_W_tilde(w_hat, ζ_vv, ζ_cc, r_super, r_unit, k_bz)
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

function assemble_W_tilde3d(w_hat, ζ_vv, ζ_cc, Ω0_vol, N_rs, N_k, q_2bz_ind, q_2bz_shift)
    N_unit = size(ζ_vv, 1)
    N_ν = size(ζ_vv, 2)
    N_μ = size(ζ_cc, 2)
    N_q = length(q_2bz_ind)

    ζ_vv_hat = zeros(Complex{Float64}, size(ζ_vv))
    for iμ in 1:size(ζ_vv, 2)
        ζ_vv_hat[:, iμ] = Ω0_vol / N_unit * vec(fft(reshape(ζ_vv[:, iμ], N_rs)))
    end
    ζ_cc_hat = zeros(Complex{Float64}, size(ζ_cc))
    for iμ in 1:size(ζ_cc, 2)
        ζ_cc_hat[:, iμ] = Ω0_vol / N_unit * vec(fft(reshape(ζ_cc[:, iμ], N_rs)))
    end

    W_tilde = complex(zeros(N_q, N_μ, N_ν))

    for iq in 1:N_q
        G_shift_indices = vec(mapslices(G -> G_vector_to_index(G, N_rs), w_hat[q_2bz_ind[iq]][2] .+ q_2bz_shift[:, iq]; dims=1))
        W_tilde[iq, :, :] = 1 / (Ω0_vol^2 * N_k) *
            ζ_cc_hat[G_shift_indices, :]' * w_hat[q_2bz_ind[iq]][1] * ζ_vv_hat[G_shift_indices, :]
    end

    return W_tilde
end

function create_W_workspace(N_v, N_c, N_k, N_ν, N_μ)
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

function create_W_workspace3d(N_v, N_c, N_ks, N_qs, N_ν, N_μ)
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

function fftfreq(n1, n2, n3)
    f1 = vcat(collect(0:(div(n1, 2) - (1 - n1 % 2))), collect(-div(n1, 2):-1))
    f2 = vcat(collect(0:(div(n2, 2) - (1 - n2 % 2))), collect(-div(n2, 2):-1))
    f3 = vcat(collect(0:(div(n3, 2) - (1 - n3 % 2))), collect(-div(n3, 2):-1))

    F1 = kron(ones(n3), ones(n2), f1)
    F2 = kron(ones(n3), f2, ones(n1))
    F3 = kron(f3, ones(n2), ones(n1))

    return vcat(F1', F2', F3')
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
                for μ in 1:N_μ
                    E[iv, ic, ik] += conj(u_c_vc[μ, ic, ik] * u_v_vc_conj[μ, iv, ik]) * D[μ]
                end
            end
        end
    end

    return vec(E)
end

# matrix free W
function W_times_vector(x, W_tilde_hat, u_v_vv_conj, u_c_cc, W_workspace)
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

function W_times_vector3d(x, W_tilde, u_v_vv_conj, u_c_cc, W_workspace)
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

function setup_H(prob, isdf)
    D = setup_D(prob)
    V = setup_V(prob, isdf)
    W = setup_W(prob, isdf)
    return D + 2 * V - W
end
