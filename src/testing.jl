using Plots; gr()

# Random Walk model
M = LGSSM(diagm(1.), diagm(1.), LowerTriangular(diagm(1.)), LowerTriangular(diagm(1.)), diagm(1.))
x = cumsum(randn(50))
y = x + randn(50)
KF = DisturbanceSmoother(M, y)

plot(x)
plot!(KF[:Xf][:,1,:]')
plot!(KF[:Xs][:,1,:]')


# Local level linear model
M = LGSSM([1. 1.; 0. 1.], [1. 0.], LowerTriangular(diagm([1.,1.])), LowerTriangular(diagm(1.)) , eye(2))
S = Generate(M, 100, 1)
plot(S[:Y][:,:,1]')
plot(S[:X][:,:,1]')

# Structural Time Series Model
# Local level model
# Seasonal component with s = 4
A = blkdiag(sparse([1],[1],[1.]), sparse([1,1,1,2,3], [1,2,3,1,2],[-1.,-1.,-1.,1.,1.]))
B = [1. 1. 0. 0.]
S = LowerTriangular(diagm(0.1))
R = vcat(diagm([1., 1e-2]), zeros(2,2))
M = LGSSM(A, B, R, S, 0.5*eye(4))
sim = Generate(M, 20*4, 5)
plot(sim[:Y][1,:,:])
KF = KalmanFilter(M, sim[:Y][1,:,1])

# Structural Time Series Model
# Local level linear model
# Seasonal component with s = 4
A = blkdiag(sparse([1,1,2],[1,2,2],[1.,1.,1.]), sparse([1,1,1,2,3], [1,2,3,1,2],[-1.,-1.,-1.,1.,1.]))
B = [1. 0. 1. 0. 0.]
S = LowerTriangular(diagm(0.1))
R = vcat(diagm([1e-2, 1e-1, 1.]), zeros(2,3))
M = LGSSM(A, B, R, S, 0.5*eye(5))
sim = Generate(M, 20*4, 50)
plot(sim[:Y][1,:,:])
