using ITensors

if !isdefined(Main, :NHDMRG)
    include("../src/NHDMRG.jl")
    using .NHDMRG
    println("Imported NHDMRG")
end

function ranmat(shape...)
    return randn(shape...) .+ 1im.*randn(shape...)
end

function compare_DMRG_to_ED(mpo::MPO, sites::Sites; chi_max::Int=50, M::MPS=nothing, Mb::MPS=nothing, method::DM_Method=LR)

    L = length(sites)

    if isnothing(M)
        M = randomMPS(Complex, sites; linkdims=min(L, chi_max))
    end
    if isnothing(Mb)
        Mb = conj(M)'
    end

    chi = 10
    while chi < chi_max
        _,_,M,Mb,_,_ = doDMRG_bb(M, Mb, mpo, chi; numsweeps=1, method=method)
        chi = min(chi+5, chi_max)
    end

    # DMRG
    Ek, _, Y, _, _, _ = doDMRG_bb(M, Mb, mpo, chi_max; numsweeps=5, method=method)
    vD = MPS_to_Vector(Y)

    # ED
    op = MPO_to_Matrix(mpo)
    w,v = eigen(op)
    mw = sortperm(abs.(w))[1]

    angle = abs(dot(vD, v[:,mw])) / (norm(vD) * norm(v[:,mw]))

    fprintln("DMRG Energy $(Ek[end]) v.s. ED Energy $(w[mw])")
    fprintln("State overlap: $angle")

end

function do_DMRGX_and_save(mpo::MPO, sites::Sites, k::Int, savename::String; chi_max::Int=50, tol::Float64=0., numsweeps::Int=10, M::MPS=undef, Mb::MPS=undef, method::DM_Method=LR)

    if isnothing(M)
        M = randomMPS(Complex, sites; linkdims=min(L, chi_max))
    end
    if isnothing(Mb)
        Mb = conj(M)'
    end

    doDMRG_excited(M, Mb, mpo, chi_max;
        k=k, expected_gap=1., tol=tol,
        numsweeps=numsweeps, dispon=2, debug=false, method=method,
        cut=1e-8, stop_if_not_converge=true, savename=savename, override=false)

end



"""
M = np.random.randn(sz,sz,sz) + 1j*np.random.randn(sz,sz,sz)
Mb = np.random.randn(sz,sz,sz) + 1j*np.random.randn(sz,sz,sz)

Y, Yb, I, Ib = right_decomp(M, Mb, chi_max=10)

log_write(np.shape(M), np.shape(Mb))
log_write(np.shape(Y), np.shape(Yb), np.shape(I), np.shape(Ib))

log_write(np.linalg.norm(ncon([Y,Yb],[[-1,1,2],[-2,1,2]])-np.eye(np.shape(Y)[0])))
log_write(np.linalg.norm(ncon([Y,I],[[1,-2,-3],[-1,1]])-M))
log_write(np.linalg.norm(ncon([Yb,Ib],[[1,-2,-3],[-1,1]])-Mb))
"""

sz = 2
L = 6
sites = [Index(4, "Site$i") for i = 1:L]
M = randomMPS(ComplexF64, sites; linkdims=2)
### Non-Hermitian
#Mb = cat([ranmat(1,sz,sz)], [ranmat(sz,sz,sz) for _ = 2:L-1], [ranmat(sz,sz,1)])
Mb = conj(M)'
#W = randomMPO(sites)
diss = sqrt(0.1)
W = LindbladMPO_W(getOp(Dict("ZZ"=>1, "Z"=>0.7, "X"=>1.5)), [getOp(Dict("X"=>diss)), getOp(Dict("Y"=>diss)), getOp(Dict("Z"=>diss))]; dagger=false)
mpo = MPOonSites(W, sites; leftindex=1, rightindex=2)
do_DMRGX_and_save(mpo, sites, 5, "Test1111Datas"; M=M, Mb=Mb, method=BB) # This is just to test things work for a trivially small system.

#doDMRG_excited_IncL(W, 40, 5, 2; k=3, debug=true, method=BB, stop_if_not_converge=true)

#compare_DMRG_to_ED(W, sites; M=M, Mb=Mb)

# exit()

# println("M:")
# println(M)
# println()

# println("Mb:")
# println(Mb)
# println()

# println("W:")
# println(W)
# println()


# ### Hermitian
# #Mb = [m.conj() for m in M]
# #W = [ranmatH(1,sz,sz,sz)] + [ranmatH(sz,sz,sz,sz) for _ in range(L-2)] + [ranmatH(sz,1,sz,sz)]

# E, Hdifs, _,_,_,_ = doDMRG_bb(M, Mb, W, 50; debug=true, numsweeps=1, method=LR)

"""
from matplotlib import pyplot as plt
plt.plot(np.arange(np.size(E))[50:], np.real(E)[50:])
plt.plot(np.arange(np.size(E))[50:], np.imag(E)[50:])
plt.show()
plt.close()

plt.plot(np.arange(np.size(Hdifs)), np.abs(Hdifs))
plt.show()
plt.close()
"""
