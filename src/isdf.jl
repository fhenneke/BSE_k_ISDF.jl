# ISDF

"""
    ISDF type

Stores information on interpolations vectors and interpolated values.
The arrays `ζ_ij` and `u_ij` are chosen such that approximately
`u_i u_j ≈ ζ_ij * u_i_ij u_j_ij`.
"""
struct ISDF
    N_μ_vv #TODO: remove this information?
    N_μ_cc
    N_μ_vc
    r_μ_vv_indices #TODO: remove this information?
    r_μ_cc_indices
    r_μ_vc_indices
    u_v_vv
    u_c_cc
    u_v_vc
    u_c_vc
    ζ_vv
    ζ_cc
    ζ_vc
end

"""
    size(isdf)

Return the size as tuple (N_μ_v, N_μ_c, N_μ_vc).
"""
function size(isdf::ISDF)
    return (size(isdf.u_v_vv, 1), size(isdf.u_c_cc, 1), size(
    isdf.u_v_vc, 1))
end

"""
    interpolation_vectors(isdf)

Returns a 3-tuple of interpolation vectors of sizes `(N_r, N_μ_v)`, `(N_r, N_μ_c)`, and `(N_r, N_μ_vc)`.
"""
function interpolation_vectors(isdf::ISDF)
    return isdf.ζ_vv, isdf.ζ_cc, isdf.ζ_vc
end

"""
    interpolation_coefficients(isdf)

Returns a 4-tuple of interpolation coefficients of sizes `(N_μ_v, N_v, N_k)`, `(N_μ_c, N_c, N_k)`, `(N_μ_vc, N_v, N_k)` and `(N_μ_vc, N_c, N_k)`.
"""
function interpolation_coefficients(isdf::ISDF)
    return isdf.u_v_vv, isdf.u_c_cc, isdf.u_v_vc, isdf.u_c_vc
end

function ISDF(prob::AbstractBSEProblem, r_μ_vv_indices::AbstractVector, r_μ_cc_indices::AbstractVector, r_μ_vc_indices::AbstractVector)
    u_v, u_c = orbitals(prob)

    ζ_vv = assemble_ζ(u_v, r_μ_vv_indices)
    ζ_cc = assemble_ζ(u_c, r_μ_cc_indices)
    ζ_vc = assemble_ζ(u_v, u_c, r_μ_vc_indices)

    return ISDF(length(r_μ_vv_indices), length(r_μ_cc_indices), length(r_μ_vc_indices),
        r_μ_vv_indices, r_μ_cc_indices, r_μ_vc_indices,
        u_v[r_μ_vv_indices, :, :], u_c[r_μ_cc_indices, :, :],
        u_v[r_μ_vc_indices, :, :], u_c[r_μ_vc_indices, :, :],
        ζ_vv, ζ_cc, ζ_vc)
end

function ISDF(prob::AbstractBSEProblem, N_μ_vv::Int, N_μ_cc::Int, N_μ_vc::Int)
    u_v, u_c = orbitals(prob)

    r_μ_vv_indices = find_r_μ(prob, N_μ_vv)
    r_μ_cc_indices = find_r_μ(prob, N_μ_cc)
    r_μ_vc_indices = find_r_μ(prob, N_μ_vc)

    return ISDF(prob, r_μ_vv_indices, r_μ_cc_indices, r_μ_vc_indices)
end

"""
    find_r_μ(prob, N_μ)

Return an array of `N_μ` indices corresponding to interpolation
points in the unit cell.
"""
function find_r_μ(prob::AbstractBSEProblem, N_μ::Int)
    # generic implementation chooses pseudo random points
    N_rs = size_r(prob)
    N_r = prod(N_rs)

    r_mask = zeros(Bool, N_rs)
    shifts = (sqrt(2), sqrt(3), sqrt(5))
    for i in 1:N_μ
        r_mask[CartesianIndex(ceil.(Int, mod.(i .* shifts, 1.0) .* N_rs))] = true
    end
    r_μ_indices = findall(vec(r_mask))
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

# routines for estimating errors

function isdf_error(prob, isdf)
    error_M_vv = isdf_error(prob.u_v, isdf.ζ_vv, isdf.r_μ_vv_indices)
    error_M_cc = isdf_error(prob.u_c, isdf.ζ_cc, isdf.r_μ_cc_indices)
    error_M_vc = isdf_error(prob.u_v, prob.u_c, isdf.ζ_vc, isdf.r_μ_vc_indices, N_k_samples)
    return error_M_vv, error_M_cc, error_M_vc
end

"""
should be equivalent to

    M_i = assemble_M(u_i)
    norm(M_i - ζ * M_i[r_μ_indices, :]) / norm(M_i)
"""
function isdf_error(u_i, ζ, r_μ_indices)
    N_unit, N_i, N_k = size(u_i)
    error2 = 0.0
    normalization2 = 0.0
    col = zeros(Complex{Float64}, N_unit)
    for ii in 1:N_i, ik in 1:N_k, jj in 1:N_i, jk in 1:N_k
        @views col[:] .= u_i[:, jj, jk] .* conj.(u_i[:, ii, ik])
        normalization2 += norm(col)^2
        error2 += norm(col - ζ * col[r_μ_indices])^2
    end

    return sqrt(error2 / normalization2)
end

function isdf_error(u_v, u_c, ζ, r_μ_indices)
    N_unit, N_c, N_k = size(u_c)
    N_unit, N_v, N_k = size(u_v)
    error2 = 0.0
    normalization2 = 0.0
    col = zeros(Complex{Float64}, N_unit)
    for iv in 1:N_v, ic in 1:N_c, ik in 1:N_k
        @views col[:] .= u_c[:, ic, ik] .* conj.(u_v[:, iv, ik])
        normalization2 += norm(col)^2
        error2 += norm(col - ζ * col[r_μ_indices])^2
    end

    return sqrt(error2 / normalization2)
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
    error2 = 0.0
    normalization2 = 0.0
    col = zeros(Complex{Float64}, N_unit)
    for ii in 1:N_i, ik in ik_samples, jj in 1:N_i, jk in ik_samples
        @views col[:] .= u_i[:, jj, jk] .* conj.(u_i[:, ii, ik])
        normalization2 += norm(col)^2
        error2 += norm(col - ζ * col[r_μ_indices])^2
    end

    return sqrt(error2 / normalization2)
end

function isdf_error_estimate(u_v, u_c, ζ, r_μ_indices, N_k_samples)
    N_unit, N_c, N_k = size(u_c)
    N_unit, N_v, N_k = size(u_v)
    ik_samples = rand(1:N_k, N_k_samples)
    error2 = 0.0
    normalization2 = 0.0
    col = zeros(Complex{Float64}, N_unit)
    for iv in 1:N_v, ic in 1:N_c, ik in ik_samples
        @views col[:] .= u_c[:, ic, ik] .* conj.(u_v[:, iv, ik])
        normalization2 += norm(col)^2
        error2 += norm(col - ζ * col[r_μ_indices])^2
    end

    return sqrt(error2 / normalization2)
end

function assemble_M(prob::AbstractBSEProblem)
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
