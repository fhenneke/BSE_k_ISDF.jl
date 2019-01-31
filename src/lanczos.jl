# lanczos absorption
using ProgressMeter

function lanczos(A, u_init, M, k)
    α = zeros(k)
    β = zeros(k)
    u = copy(u_init)
    u_old = zero(u)
    v_old = zero(u)
    U = zeros(eltype(u), length(u), k + 1)
    U[:, 1] .= u

    v = M * u
    x = A * v
    α[1] = real(v' * x)
    x .-= α[1] .* u
    y = M * x
    β[1] = sqrt(x' * y)
    u_old .= u
    u .= x ./ β[1]
    U[:, 2] .= u
    v_old .= v
    v .= y ./ β[1]
    @showprogress 1 "Lanczos iteration ..." for j in 2:k
        x .= A * v .- β[j - 1] .* u_old
        α[j] = real(v' * x)
        x .-= α[j] .* u
        y .= M * x
        β[j] = sqrt(real(x' * y))
        u_old .= u
        u .= x ./ β[j]
        U[:, j + 1] .= u
        v_old .= v
        v .= y ./ β[j]
    end
    return α, β, U
end

function lanczos_eig(H, d, N_iter)
    α, β, U = lanczos(H, d, I, N_iter)
    return lanczos_eig(α, β, N_iter)
end

function lanczos_eig(α::AbstractVector, β, N_iter)
    T̂ = SymTridiagonal(vcat(α[1:N_iter], α[(N_iter - 1):-1:1]), vcat(β[1:N_iter], β[(N_iter - 2):-1:1]))
    F = eigen(T̂)
    ev_lanczos, ef_lanczos = F.values, F.vectors
    return ev_lanczos, ef_lanczos
end

function optical_absorption_vector(prob::BSEProblem1D)
    return optical_absorption_vector(prob.u_v, prob.u_c, prob.E_v, prob.E_c, prob.prob.r_unit, prob.prob.k_bz)
end

function optical_absorption_vector(u_v, u_c, E_v, E_c, r_unit, k_bz) # TODO: make general? only works in 1d atm
    N_unit = length(r_unit)
    N_v = size(u_v, 2)
    N_c = size(u_c, 2)
    N_k = length(k_bz)
    l = N_unit * (r_unit[2] - r_unit[1])

    d = vec([im * dot(u_c[:, ic, ik],
                  modified_momentum_operator(r_unit, l, k_bz[ik]) *
                    u_v[:, iv, ik]) /
            (E_c[ic, ik] - E_v[iv, ik])
            for iv in 1:N_v, ic in 1:N_c, ik in 1:N_k])

    return d
end

function lanczos_optical_absorption(prob::BSEProblem, isdf::ISDF, N_iter, g, Erange) #TODO: adapt to 3d
    l = prob.prob.l

    H = setup_H(prob, isdf)
    d = optical_absorption_vector(prob)
    ev_lanczos, ef_lanczos = lanczos_eig(H, normalize(d), N_iter)

    optical_absorption = zeros(length(Erange))
    for j in 1:(2 * N_iter - 1)
        optical_absorption .+= abs2(ef_lanczos[1, j]) * g.(Erange .- ev_lanczos[j])
    end
    optical_absorption .*= norm(d)^2 * 8 * π^2 / l

    return optical_absorption
end

function lanczos_optical_absorption(α::AbstractVector, β, N_iter, g, Erange) # TODO: unify with matrix, vector function below?
    ev_lanczos, ef_lanczos = lanczos_eig(α, β, N_iter)

    optical_absorption = zeros(length(Erange))
    for j in 1:(2 * N_iter - 1)
        optical_absorption .+= abs2(ef_lanczos[1, j]) * g.(Erange .- ev_lanczos[j])
    end

    return optical_absorption
end

function lanczos_optical_absorption(H, d, N_iter, g, Erange)
    α, β, U = lanczos(H, d, I, N_iter)
    return lanczos_optical_absorption(α, β, N_iter, g, Erange)
end

function optical_absorption(prob::BSEProblem1D, g, Erange)
    H = assemble_exact_H(prob)
    F = eigen(Hermitian(H))
    ev, ef = F.values, F.vectors

    absorption = optical_absorption(ev, ef, prob.u_v, prob.u_c, prob.E_v, prob.E_c, prob.prob.r_unit, prob.prob.k_bz, g, Erange)

    return absorption
end

function optical_absorption(ev, ef, u_v, u_c, E_v, E_c, r_unit, k_bz, g, Erange)
    N_v = size(u_v, 2)
    N_c = size(u_c, 2)
    N_k = length(k_bz)
    l = (r_unit[2] - r_unit[1]) * length(r_unit)

    d = optical_absorption_vector(u_v, u_c, E_v, E_c, r_unit, k_bz)
    osc = [abs2(dot(d, ef[:, n])) for n in 1:(N_v * N_c * N_k)]

    absorption = zeros(length(Erange))
    for j in 1:(N_v * N_c * N_k)
        absorption .+= osc[j] * g.(Erange .- ev[j])
    end
    absorption .*= 8 * π^2 / l

    return absorption
end
