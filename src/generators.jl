function Generate(M::LGSSM, n::Int, N::Int)
    ydim, qdim = size(M.B)
    xdim, pdim = size(M.R)
    L = cholfact(M.Σᵥ)[:L]
    # RR = M.R*M.R'
    # SS = M.S*M.S'
    X = zeros(xdim, n, N)
    Y = zeros(ydim, n, N)
    for t in 1:n
        # Generate States
        if t == 1
            X[:, 1, :] = L*randn(xdim, N)
        else
            X[:,t,:] = M.A*X[:,t-1,:] + M.R*randn(pdim ,N)
        end

        # Generate Measurements
        Y[:,t,:] = M.B*X[:,t,:] + M.S*randn(ydim, N)
    end
    return Dict(:X => X, :Y => Y)
end
