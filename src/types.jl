# types for problems and solutions, mainly containers

abstract type SPProblem end

struct SPProblem1D <: SPProblem
    V_sp::Function
    l::Float64
    r_unit::Vector{Float64}
    r_super::Vector{Float64}
    k_bz::Vector{Float64}
end

struct SPSolution1D
    prob
    ev
    ef
end

abstract type BSEProblem end

struct BSEProblem1D <: BSEProblem
    prob::SPProblem1D
    N_core::Int
    N_v::Int
    N_c::Int
    N_k::Int
    E_v::Array{Float64, 2}
    E_c::Array{Float64, 2}
    u_v::Array{Complex{Float64}, 3}
    u_c::Array{Complex{Float64}, 3}
    v_hat::Array{Complex{Float64}, 2}
    w_hat::Array{Complex{Float64}, 3}
end

struct ISDF
    N_μ_vv
    N_μ_cc
    N_μ_vc
    r_μ_vv_indices
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
