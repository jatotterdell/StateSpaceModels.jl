abstract type SSM end

struct LGSSM{T <: Real} <: SSM
    A::AbstractArray{T} # State matrix
    B::AbstractArray{T} # Measurement matrix
    R::AbstractArray{T} # State Cholesky
    S::LowerTriangular{T, Array{T,2}} # Measurement Cholesky
    Σᵥ::Matrix{T} # Initial state covariance
    # μᵥ::Vector{T} # Initial state mean
end
