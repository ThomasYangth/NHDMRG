
# This is a version where I try to implement DMRG for increasing system size, but this is a really clumsy implementation.
function doDMRG_excited_IncL(W0::Array{<:Number, 4}, chi_max::Int, L0::Int, doubles::Int;
    k::Int=1, expected_gap::Float64=1., tol::Float64 = 1e-15,
    numsweeps::Int = 10, dispon::Int = 2, debug::Bool = false, method::DM_Method = LR,
    cut::Float64 = 1e-8, stop_if_not_converge::Bool = true, savename = nothing, override::Bool = false)

    if isnothing(savename)
        savename = Dates.format(now(), "MMddyy-HHMMSS")
    end

    filename = "$savename.jld2"

    # Ms[i,j] will be the j-th excited state for system size L0*2^i.
    Ms = Array{MPS, 2}(undef, k, doubles)
    Mbs = Array{MPS, 2}(undef, k, doubles)
    Es = Array{ComplexF64, 1}()

    # At length L0, do a sparse diagonalization for the full matrix, to get the initial states.
    sites = [Index(size(W0)[1], "Site $i") for i = 1:L0*(2^doubles)]
    initSites = sites[1:L0]
    W_L0 = MPOonSites(W0, initSites; leftindex=1, rightindex=2)
    W_L0_mat, comb = MPO_to_Matrix(W_L0)
    Mci = inds(comb)[1]
    wR, vR = doeig(W_L0_mat, ComplexF64[randn() + im * randn() for _ in 1:L0]; k=k, use_sparse=false)
    wL, vL = doeig(transpose(W_L0_mat), ComplexF64[randn() + im * randn() for _ in 1:L0]; k=k, use_sparse=false)

    fprintln("Found right eigenvalues: ", wR)
    fprintln("Found left eigenvalues: ", wL)

    if !isapprox(wR, wL; atol=1e-6)
        throw(ErrorException("Eigenvalues of W_L0 are not symmetric!"))
    end

    # Double the states vR and vL to serve as initial guesses for the system size 2*L0.
    for i = 1:k

        Y, Yb = decomp(ITensor(vR[:,i], Mci)*comb, ITensor(vL[:,i], Mci')*comb', initSites; chi_max=chi_max, timing=debug, method=method, linknames=["$j-link-$(j+1)" for j=1:L0-1])

        initSites2 = sites[L0+1:2*L0]
        comb2 = combiner(initSites2...)
        Mci2 = inds(comb2)[1]
        Z, Zb = decomp(ITensor(vR[:,i], Mci2)*comb2, ITensor(vL[:,i], Mci2')*comb2', initSites2[end:-1:1]; chi_max=chi_max, timing=debug, method=method, linknames=["$(j-1)-link-$j" for j=2*L0:-1:L0+2])
    
        M = ITensor[]
        Mb = ITensor[]
        for j = 1:L0-1
            push!(M, Y[j])
            push!(Mb, Yb[j])
        end

        # Add a trivial link to connect the two halves
        newlink = Index(1, "$L0-link-$(L0+1)")
        push!(M, Y[end]*ITensor(1, newlink))
        push!(Mb, Yb[end]*ITensor(1, newlink'))
        push!(M, ITensor(1, newlink)*Z[end])
        push!(Mb, ITensor(1, newlink')*Zb[end])

        for j = 1:L0-1
            push!(M, Z[L0-j])
            push!(Mb, Zb[L0-j])
        end

        Ms[i,1] = MPS(M)
        Mbs[i,1] = MPS(Mb)
    
    end

    for j = 1:doubles

        this_L = L0*(2^j)
        this_sites = sites[1:this_L]
        thisW = MPOonSites(W0, this_sites; leftindex=1, rightindex=2)

        for thisk = 1:k
            # Do DMRG for this size
            _, _, Ms[thisk,j], Mbs[thisk,j] = doDMRG(Ms[thisk,j], Mbs[thisk,j], thisW, chi_max;
                    normalize_against = [(Ms[i,j], Mbs[i,j], -expected_gap*(thisk-i)) for i = 1:thisk-1],
                    sigma = shift_eps*im + 0.1, numsweeps=numsweeps, dispon=dispon, debug=debug, method=method,
                    stop_if_not_converge=stop_if_not_converge)
            # Double the MPS for the next initial guess
            if j < doubles
                Ms[thisk,j+1], Mbs[thisk,j+1] = doubleMPSSize(Ms[thisk,j], Mbs[thisk,j]; chi_max=chi_max, method=method, timing=debug, newsites=sites[this_L+1:2*this_L])
            end
        end

    end

end