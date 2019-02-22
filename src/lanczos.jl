# lanczos absorption
using ProgressMeter

"""
    lanczos(A, u_init, k[, M])

Perform k steps of Lanczos method for the operator `A` and initial
vector `u_init`. The output are the vectors `α`, `β` and `U` with
`A * U ≈ U * SymTridiagonal(α, β[1:(end - 1)])`.
"""
function lanczos(A, u_init, k, M = I)
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
    α, β, U = lanczos(H, d, N_iter)
    return lanczos_eig(α, β, N_iter)
end

function lanczos_eig(α::AbstractVector, β, N_iter)
    T̂ = SymTridiagonal(vcat(α[1:N_iter], α[(N_iter - 1):-1:1]), vcat(β[1:N_iter], β[(N_iter - 2):-1:1]))
    F = eigen(T̂)
    ev_lanczos, ef_lanczos = F.values, F.vectors
    return ev_lanczos, ef_lanczos
end

function lanczos_optical_absorption(prob::AbstractBSEProblem, isdf::ISDF, N_iter, g, Erange) #TODO: adapt to 3d
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
    α, β, U = lanczos(H, d, N_iter)
    return lanczos_optical_absorption(α, β, N_iter, g, Erange)
end
