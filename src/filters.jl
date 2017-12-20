"""

```math
\\tilde X = \\hat X_{t+1|t} = A\\hat X_{t|t}
```
"""
function KalmanFilter(M::LGSSM, y)
    RR = M.R * M.R'
    SS = M.S * M.S'
    ydim, qdim = size(M.B)
    xdim, pdim = size(M.R)
    n = size(y, 1)

    ϵ = zeros(ydim, 1, n)
    Γ = zeros(ydim, ydim, n)
    Γ⁻¹ = zeros(ydim, ydim, n)
    K = zeros(xdim, ydim, n) # Kalman filter gain
    H = zeros(xdim, ydim, n) # Kalman prediction gain
    Λ = zeros(xdim, xdim, n)

    Xp = zeros(xdim, 1, n)
    Σp = zeros(xdim, xdim, n)
    Xf = zeros(xdim, 1, n)
    Σf = zeros(xdim, xdim, n)

    ℓ = 0.0

    for t in 1:n
        # Prediction
        if t == 1
            Xp[:,:,t] = 0.0
            Σp[:,:,t] = M.Σᵥ
        else
            Xp[:,:,t] = M.A * Xf[:,:,t-1]
            Σp[:,:,t] = M.A * Σf[:,:,t-1] * M.A' + RR
        end

        # Correction
        ϵ[:,:,t] = y[t] - M.B * Xp[:,:,t]
        Γ[:,:,t] = M.B*Σp[:,:,t]*M.B' + SS
        Γ⁻¹[:,:,t] = inv(Γ[:,:,t])
        K[:,:,t] = Σp[:,:,t]*M.B'*Γ⁻¹[:,:,t]
        H[:,:,t] = M.A*K[:,:,t]
        Λ[:,:,t] = M.A - H[:,:,t]*M.B
        Xf[:,:,t] = Xp[:,:,t] + K[:,:,t]*ϵ[:,:,t]
        Σf[:,:,t] = Σp[:,:,t] - K[:,:,t]*M.B*Σp[:,:,t]
        ℓ += logdet(Γ[:,:,t]) + ϵ[:,:,t]'*Γ⁻¹[:,:,t]*ϵ[:,:,t]
    end
    return Dict(:Xp => Xp,
                :Xf => Xf,
                :Σp => Σp,
                :Σf => Σf,
                :ϵ => ϵ,
                :Γ => Γ,
                :Γ⁻¹ => Γ⁻¹,
                :K => K,
                :H => H,
                :Λ => Λ,
                :ℓ => -ℓ/2)
end


"""

    Disturbanve Smoother

    Algorithm 5.2.15 Cappe 2005
"""
function DisturbanceSmoother(M::LGSSM, y)
    n = size(y, 1)
    pdim = size(M.B, 2)
    qdim = size(y, 2)
    RR = M.R*M.R'

    p = zeros(pdim, 1, n-1)
    C = zeros(pdim, pdim, n-1)
    U = zeros(pdim, pdim, n-1)
    Ξ = zeros(pdim, pdim, n-1)

    Xs = zeros(pdim, 1, n)
    Σs = zeros(pdim, pdim, n)

    KF = KalmanFilter(M, y)

    # Smoothed disturbances
    for t in n-1:-1:1
        if t == n-1
            p[:,:,t] = M.B'*KF[:Γ⁻¹][:,:,n]*KF[:ϵ][:,:,n]
            C[:,:,t] = M.B'*KF[:Γ⁻¹][:,:,n]*M.B
        else
            p[:,:,t] = M.B'*KF[:Γ⁻¹][:,:,t+1]*KF[:ϵ][:,:,t+1] + KF[:Λ][:,:,t+1]'*p[:,:,t+1]
            C[:,:,t] = M.B'*KF[:Γ⁻¹][:,:,t+1]*M.B + KF[:Λ][:,:,t+1]'*C[:,:,t+1]*KF[:Λ][:,:,t+1]
        end
        U[:,:,t] = M.R'*p[:,:,t]
        Ξ[:,:,t] = I - M.R'*C[:,:,t]*M.R
    end

    # Smoothed states
    Xs[:,:,1] = M.Σᵥ*(M.B'*KF[:Γ⁻¹][:,:,1]*KF[:ϵ][:,:,1] + KF[:Λ][:,:,1]*p[:,:,1])
    Σs[:,:,1] = M.Σᵥ - M.Σᵥ*(M.B'*KF[:Γ⁻¹][:,:,1]*M.B + KF[:Λ][:,:,1]'*C[:,:,1]*KF[:Λ][:,:,1])*M.Σᵥ
    for t in 1:n-1
        Xs[:,:,t+1] = M.A*Xs[:,:,t] + M.R*U[:,:,t]
        Σs[:,:,t+1] = M.A*Σs[:,:,t]*M.A' + M.R*Ξ[:,:,t]*M.R' -
                        M.A*KF[:Σp][:,:,t]*KF[:Λ][:,:,t]'*C[:,:,t]*RR -
                        RR*C[:,:,t]*KF[:Λ][:,:,t]*KF[:Σp][:,:,t]*M.A'
    end
    return Dict(:Xp => KF[:Xp],
                :Xf => KF[:Xf],
                :Xs => Xs,
                :Σp => KF[:Σp],
                :Σf => KF[:Σf],
                :Σs => Σs,
                :ϵ => KF[:ϵ],
                :Γ => KF[:Γ],
                :Γ⁻¹ => KF[:Γ⁻¹],
                :K => KF[:K],
                :H => KF[:H],
                :Λ => KF[:Λ],
                :p => p,
                :C => C,
                :U => U,
                :Ξ => Ξ,
                :ℓ => KF[:ℓ])
end

"""
    Backward Information Recursion

    Proposition 5.2.21 Cappe (2005)
"""
function BackwardInformationRecursion(M::LGSSM, y)
    n = size(y, 1)
    pdim = size(M.B, 2)
    qdim = size(y, 2)
    RR = M.R*M.R'
    SS = M.S*M.S'
    SS⁻¹ = inv(SS)

    κ = zeros(pdim, 1, n)
    κ̃ = zeros(pdim, 1, n)
    Π = zeros(pdim, pdim, n)
    Π̃ = zeros(pdim, pdim, n)

    Xs = zeros(pdim, 1, n)
    Σs = zeros(pdim, pdim, n)

    KF = KalmanFilter(M, y)

    for t in n-1:-1:1
        κ̃[:,:,t+1] = M.B'*SS⁻¹*y[t+1] + κ[:,:,t+1]
        Π̃[:,:,t+1] = M.B'*SS⁻¹*M.B + Π[:,:,t+1]
        V = inv(I + Π̃[:,:,t+1]*RR)
        κ[:,:,t] = M.A'V*κ̃[:,:,t+1]
        Π[:,:,t] = M.A'V*Π̃[:,:,t+1]*M.A
    end
    Dict(:Xp => KF[:Xp],
                :Xf => KF[:Xf],
                :Σp => KF[:Σp],
                :Σf => KF[:Σf],
                :κ => κ,
                :Π => Π)
end

"""
    Forward-backward smoother

    Algorithm 5.2.22 Cappe (2005)
"""
function ForwardBackwardSmoother(M::LGSSM, y)
    n = size(y, 1)
    pdim = size(M.B, 2)
    qdim = size(y, 2)
    RR = M.R*M.R'
    SS = M.S*M.S'
    SS⁻¹ = inv(SS)

    κ = zeros(pdim, 1, n)
    κ̃ = zeros(pdim, 1, n)
    Π = zeros(pdim, pdim, n)
    Π̃ = zeros(pdim, pdim, n)

    Xs = zeros(pdim, 1, n)
    Σs = zeros(pdim, pdim, n)

    KF = KalmanFilter(M, y)

    W = inv(I + Π[:,:,n]*KF[:Σf][:,:,n])
    Xs[:,:,n] = KF[:Xf][:,:,n] + KF[:Σf][:,:,n]*W*(κ[:,:,n] - Π[:,:,n]*KF[:Xf][:,:,n])
    Σs[:,:,n] = KF[:Σf][:,:,n] - KF[:Σf][:,:,n]*W*Π[:,:,n]*KF[:Σf][:,:,n]

    for t in n-1:-1:1
        κ̃[:,:,t+1] = M.B'*SS⁻¹*y[t+1] + κ[:,:,t+1]
        Π̃[:,:,t+1] = M.B'*SS⁻¹*M.B + Π[:,:,t+1]
        V = inv(I + Π̃[:,:,t+1]*RR)
        κ[:,:,t] = M.A'V*κ̃[:,:,t+1]
        Π[:,:,t] = M.A'V*Π̃[:,:,t+1]*M.A
        W = inv(I + Π[:,:,t]*KF[:Σf][:,:,t])
        Xs[:,:,t] = KF[:Xf][:,:,t] + KF[:Σf][:,:,t]*W*(κ[:,:,t] - Π[:,:,t]*KF[:Xf][:,:,t])
        Σs[:,:,t] = KF[:Σf][:,:,t] - KF[:Σf][:,:,t]*W*Π[:,:,t]*KF[:Σf][:,:,t]
    end
    Dict(:Xp => KF[:Xp],
                :Xf => KF[:Xf],
                :Xs => Xs,
                :Σp => KF[:Σp],
                :Σf => KF[:Σf],
                :Σs => Σs,
                :κ => κ,
                :Π => Π)
end

"""
    Backward Information Recursion

    Algorithm 6.1.2 Cappe (2005)
"""
function BackwardMarkovianStateSampling(M::LGSSM, y, N::Int)
    n = size(y, 1)
    KF = KalmanFilter(M, y)
    pdim = size(M.B, 2)
    qdim = size(y, 2)

    X = zeros(pdim, n, N)
    X[:,1,:] = rand(MvNormal(reshape(KF[:Xf][:,:,1],1), KF[:Σf][:,:,1]), N)
    for t in 2:n
        X[:,t,:] = rand(MvNormal(reshape(KF[:Xf][:,:,t],1), KF[:Σf][:,:,t]), N)
    end
    return X
end


"""
    Sampling with Dual Smoothing

    Algorithm 6.1.3 Cappe (2005)
"""
function SamplingWithDualSmoothing(M::LGSSM, y)
    n = size(y, 1)
    pdim = size(M.B, 2)
    qdim = size(y, 2)

    XY′ = Generate(M, n)
    KF = DisturbanceSmoother(M, y)
    KF′ = DisturbanceSmoother(M, XY′[:Y])
    return KF[:Xf] + XY′[:X] - KF′[:Xf]
end
