# ISDF

function ISDF(prob::BSEProblem, N_μ_vv::Int, N_μ_cc::Int, N_μ_vc::Int)
    N_unit = length(prob.prob.r_unit)
    r_μ_vv_indices = find_r_μ(N_unit, N_μ_vv)
    r_μ_cc_indices = find_r_μ(N_unit, N_μ_cc)
    r_μ_vc_indices = find_r_μ(N_unit, N_μ_vc)

    ζ_vv = assemble_ζ(prob.u_v, r_μ_vv_indices)
    ζ_cc = assemble_ζ(prob.u_c, r_μ_cc_indices)
    ζ_vc = assemble_ζ(prob.u_v, prob.u_c, r_μ_vc_indices)

    return ISDF(N_μ_vv, N_μ_cc, N_μ_vc,
        r_μ_vv_indices, r_μ_cc_indices, r_μ_vc_indices,
        prob.u_v[r_μ_vv_indices, :, :], prob.u_c[r_μ_cc_indices, :, :],
        prob.u_v[r_μ_vc_indices, :, :], prob.u_c[r_μ_vc_indices, :, :],
        ζ_vv, ζ_cc, ζ_vc)
end

function find_r_μ(N_unit::Int, N_μ::Int)
    r_μ_indices = Int.(round.(range(1, stop = N_unit + 1, length = N_μ + 1)[1:(end - 1)]))
    return r_μ_indices
end

function assemble_ζ(u_i, r_μ_indices)
    N_unit, N_i, N_k = size(u_i)
    U_i = reshape(u_i, N_unit, N_i * N_k)
    P_i = conj.(U_i * U_i[r_μ_indices, :]')
    P_ii = abs2.(P_i)
    A = P_ii[:, :]
    B = P_ii[r_μ_indices, :]
    ζ = copy((qr(B', Val(true)) \ A')')
    return ζ
end

function assemble_ζ(u_i, u_j, r_μ_indices)
    N_unit, N_i, N_k = size(u_i)
    N_unit, N_j, N_k = size(u_j)
    U_i = reshape(u_i, N_unit, N_i * N_k)
    U_j = reshape(u_j, N_unit, N_j * N_k)
    P_i = conj.(U_i * U_i[r_μ_indices, :]')
    P_j = conj.(U_j * U_j[r_μ_indices, :]')
    P_ij = P_j .* conj(P_i)
    A = P_ij[:, :]
    B = P_ij[r_μ_indices, :]
    ζ = copy((qr(B', Val(true)) \ A')')
    return ζ
end

function estimate_error(prob, isdf, N_k_samples = 20)
    error_M_vv = isdf_error_estimate(prob.u_v, isdf.ζ_vv, isdf.r_μ_vv_indices, N_k_samples)
    error_M_cc = isdf_error_estimate(prob.u_c, isdf.ζ_cc, isdf.r_μ_cc_indices, N_k_samples)
    error_M_vc = isdf_error_estimate(prob.u_v, prob.u_c, isdf.ζ_vc, isdf.r_μ_vc_indices, N_k_samples)
    return error_M_vv, error_M_cc, error_M_vc
end

function isdf_error_estimate(u_i, ζ, r_μ_indices, N_k_samples)
    N_unit, N_i, N_k = size(u_i)
    ik_samples = rand(1:N_k, N_k_samples)
    M_sample_reshaped = [u_i[ir, jj, jk] * conj(u_i[ir, ii, ik]) for ir in 1:N_unit, ii in 1:N_i, ik in ik_samples, jj in 1:N_i, jk in ik_samples]
    M_sample = reshape(M_sample_reshaped, N_unit, (N_i * N_k_samples)^2)
    vecnorm(M_sample - ζ * M_sample[r_μ_indices, :]) / vecnorm(M_sample)
end

function isdf_error_estimate(u_v, u_c, ζ, r_μ_indices, N_k_samples)
    N_unit, N_c, N_k = size(u_c)
    N_unit, N_v, N_k = size(u_v)
    ik_samples = rand(1:N_k, N_k_samples)
    M_sample_reshaped = [u_c[ir, ic, ik] * conj(u_v[ir, iv, ik]) for ir in 1:N_unit, iv in 1:N_v, ic in 1:N_c, ik in ik_samples]
    M_sample = reshape(M_sample_reshaped, N_unit, N_v * N_c * N_k_samples)
    vecnorm(M_sample - ζ * M_sample[r_μ_indices, :]) / vecnorm(M_sample)
end

function assemble_M(prob::BSEProblem)
    M_vv = assemble_M(prob.u_v)
    M_cc = assemble_M(prob.u_c)
    M_vc = assemble_M(prob.u_v, prob.u_c)
    return M_vv, M_cc, M_vc
end

function assemble_M(u_i::Array)
    N_unit, N_i, N_k = size(u_i)
    M_reshaped = [u_i[ir, jj, jk] * conj(u_i[ir, ii, ik]) for ir in 1:N_unit, ii in 1:N_i, ik in 1:N_k, jj in 1:N_i, jk in 1:N_k]
    M = reshape(M_reshaped, N_unit, (N_i * N_k)^2)
end

function assemble_M(u_v::Array, u_c::Array)
    N_unit, N_c, N_k = size(u_c)
    N_unit, N_v, N_k = size(u_v)
    M_reshaped = [u_c[ir, ic, ik] * conj(u_v[ir, iv, ik]) for ir in 1:N_unit, iv in 1:N_v, ic in 1:N_c, ik in 1:N_k]
    M = reshape(M_reshaped, N_unit, N_v * N_c * N_k)
end

function assemble_M(isdf::ISDF)
    M_vv = isdf.ζ_vv * assemble_M(isdf.u_v_vv)
    M_cc = isdf.ζ_cc * assemble_M(isdf.u_c_cc)
    M_vc = isdf.ζ_vc * assemble_M(isdf.u_v_vc, isdf.u_c_vc)
    return M_vv, M_cc, M_vc
end
