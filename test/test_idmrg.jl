using ITensors

if !isdefined(Main, :NHDMRG)
    include("../src/NHDMRG.jl")
    using .NHDMRG
end

d = 3  # Local Hilbert space dimension for spin-1
Sx = [0 1 0; 1 0 1; 0 1 0] ./ sqrt(2)
Sy = [0 -1im 0; 1im 0 -1im; 0 1im 0] ./ sqrt(2)
Sz = [1 0 0; 0 0 0; 0 0 -1]
Id = [1 0 0; 0 1 0; 0 0 1]

W = ComplexF64.(zeros(d, d, 5, 5))

W[:,:,1,1] = Id
W[:,:,1,2] = Sx
W[:,:,1,3] = Sy
W[:,:,1,4] = Sz
W[:,:,2,5] = (Sx^2 + Sy^2 + Sz^2) / 3
W[:,:,3,5] = Sx
W[:,:,4,5] = Sy
W[:,:,5,5] = Sz

doIDMRG(W, 30; method=LR, maxL=100, dispon=3, Wleftindex=1, Wrightindex=5, sigma=ComplexF64(-5))