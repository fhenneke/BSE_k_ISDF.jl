# ISDF
import Random: randperm#, MersenneTwister # TODO: sort out imports

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

# abstract type AbstractInterpolationMethod
# end

# struct UniformInterpolation <: AbstractInterpolationMethod

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
    N_v, N_c, N_k = size(prob)
    N_sub_max = 30

    N_sub_vv = min(N_sub_max, N_v * N_k)
    N_sub_cc = min(N_sub_max, N_c * N_k)
    N_sub_vc = min(N_sub_max^2, N_v * N_c * N_k)

    r_μ_vv_indices = find_r_μ(prob.u_v, N_μ_vv, N_sub_vv)
    r_μ_cc_indices = find_r_μ(prob.u_c, N_μ_cc, N_sub_cc)
    r_μ_vc_indices = find_r_μ(prob.u_v, prob.u_c, N_μ_vc, N_sub_vc)

    return ISDF(prob, r_μ_vv_indices, r_μ_cc_indices, r_μ_vc_indices)
end

"""
    find_r_μ(u_i, N_μ, N_sub)

Return an array of indices corresponding to interpolation
points in the unit cell. The points are computed via the algorithm proposed in
Lu, Ying; 2016; Fast Allgorithm for periodic Fitting for Bloch Waves
"""
function find_r_μ(u_i, N_μ, N_sub)
    return qrcp(u_i, N_sub).p[1:N_μ]
end
function qrcp(u_i, N_sub)
    N_r, N_i, N_k = size(u_i)
    
    random_phase = rand(MersenneTwister(42), N_i * N_k)
    random_unit_complex = cos.(2 .* pi .* random_phase) .+ im * sin.(2 .* pi .* random_phase)
    u_i_r_transformed = zeros(Complex{Float64}, N_i * N_k)
    sub_ind = randperm(MersenneTwister(42), N_i * N_k)[1:N_sub]
    M = zeros(Complex{Float64}, N_sub^2, N_r)
    for ir in 1:N_r
        u_i_r_transformed[:] .= random_unit_complex .* vec(u_i[ir, :, :])
        fft!(u_i_r_transformed)
        M[:, ir] = vec(conj.(reshape(u_i_r_transformed[sub_ind], :, 1)) .* reshape(u_i_r_transformed[sub_ind], 1, :))
    end
    F = qr(M, Val(true))

    return F
end

function find_r_μ(u_v, u_c, N_μ, N_sub)
    return qrcp(u_v, u_c, N_sub).p[1:N_μ]
end
function qrcp(u_v, u_c, N_sub)
    N_r, N_v, N_k = size(u_v)
    N_r, N_c, N_k = size(u_c)
    
    random_phase = rand(MersenneTwister(42), N_v * N_c * N_k)
    random_unit_complex = cos.(2 .* pi .* random_phase) .+ im * sin.(2 .* pi .* random_phase)
    u_vc_r_transformed = zeros(Complex{Float64}, N_v * N_c * N_k)
    sub_ind = randperm(MersenneTwister(42), N_v * N_c * N_k)[1:N_sub]
    M = zeros(Complex{Float64}, N_sub, N_r)
    for ir in 1:N_r
        u_vc_r_transformed[:] .= random_unit_complex .* vec(conj.(reshape(u_v[ir, :, :], N_v, 1, N_k)) .* reshape(u_c[ir, :, :], 1, N_c, N_k))
        fft!(u_vc_r_transformed)
        M[:, ir] = u_vc_r_transformed[sub_ind]
    end
    F = qr(M, Val(true))

    return F
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
    error_M_vc = isdf_error(prob.u_v, prob.u_c, isdf.ζ_vc, isdf.r_μ_vc_indices)
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
    rng = MersenneTwister(0)
    ik_samples = rand(rng, 1:N_k, N_k_samples)
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
    rng = MersenneTwister(0)
    ik_samples = rand(rng, 1:N_k, N_k_samples)
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
