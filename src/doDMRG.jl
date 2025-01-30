# -*- coding: utf-8 -*-
# doDMRG.jl

"""

    Non-Hermitian Density Matrix Renormalization Group, by Tian-Hua Yang,
    based on www.tensors.net, (v1.1) by Glen Evenbly.

"""

using Dates
using Statistics # For std
using JLD2 # For saving and processing array

using Printf

export doDMRG_excited, doDMRG_excited_IncL, doDMRG_IncChi, doIDMRG, doDMRG

"""
    doDMRG_excited(M::MPS, Mb::MPS, W::MPO, chi_max::Int;
    k::Int=1, expected_gap::Float64=1, tol::Float64 = 1e-15,
    numsweeps::Int = 10, dispon::Int = 2, debug::Bool = false, method::DM_Method = LR,
    cut::Float64 = 1e-8, stop_if_not_converge::Bool = true, savename = nothing, override::Bool = false)

Function does DMRG to find the first k excited states of a Hamiltonian given by the MPO W.

# Arguments
- `M`: MPS, initial guess for right eigenvector.
- `Mb`: MPS, initial guess for left eigenvector.
- `W`: MPO, Hamiltonian, or Liouvillian.
- `chi_max`: Int, maximal bond dimension.
- `k`: Int, number of excited states; default 1, ground state only.
- `expected_gap`: Float, default 1. Expected gap between excited states,
  used to normalize the Hamiltonian when searching for excited states.
- `tol`: Float, default 1e-15. Tolerance when diagonalizing a local block.
- `numsweeps`: Int, default 10. Number of sweeps.
- `savename`: String, default nothing. A temporary file that saves the already-converged states.


# Returns
Description of the return value.

# Examples

"""
function doDMRG_excited(M::MPS, Mb::MPS, W::MPO, chi_max::Int;
    k::Int=1, expected_gap::Float64=1, tol::Float64 = 1e-15,
    numsweeps::Int = 10, dispon::Int = 2, debug::Bool = false, method::DM_Method = LR,
    cut::Float64 = 1e-8, stop_if_not_converge::Bool = true, savename = nothing, override::Bool = false)

    L = length(W)

    Ms = Array{MPS, 1}()
    Mbs = Array{MPS, 1}()
    Es = Array{ComplexF64, 1}()

    chi_start = 50
    vt_amp = 15

    if isnothing(savename)
        savename = Dates.format(now(), "MMddyy-HHMMSS")
    end

    if !isa(W, MPO)
        W = MPO(W)
    end

    filename = "$savename.jld2"

    if !override && isfile(filename)
        fprintln("Loading file $filename")
        jldopen(filename, "r") do file

            # Read the already converged energies
            if haskey(file, "Es")
                Es = file["Es"]
            end

            # Read the already converged eigenstates
            k_loop_1 = false
            for i = 1:k
                thisM = []
                for j = 1:L
                    if !haskey(file, "M$(i)R$(j)")
                        k_loop_1 = true
                        break
                    end
                    push!(thisM, file["M$(i)R$(j)"])
                end
                if k_loop_1
                    break
                end
                push!(Ms, MPS(thisM))
            end

            k_loop_2 = false
            for i = 1:k
                thisM = []
                for j = 1:L
                    if !haskey(file, "M$(i)L$(j)")
                        k_loop_2 = true
                        break
                    end
                    push!(thisM, file["M$(i)L$(j)"])
                end
                if k_loop_2
                    break
                end
                push!(Mbs, MPS(thisM))
            end

        end
    end

    function dosave()
        jldopen(filename, (!override && isfile(filename)) ? "r+" : "w+") do file

            file["Es"] = Es

            for (i,M) in enumerate(Ms)
                for (key,mat) in M.asdict("M$(i)R")
                    if override && !haskey(file, key)
                        file[key] = mat
                    end
                end
            end

            for (i,M) in enumerate(Mbs)
                for (key,mat) in M.asdict("M$(i)L")
                    if override && !haskey(file, key)
                        file[key] = mat
                    end
                end
            end

        end
        fprintln("Saved current data: len(Es)=$(length(Es)), len(Ms)=$(length(Ms)), len(Mbs)=$(length(Mbs)).")
    end
        
    for thisk = 1:k

        fprintln("Finding eigenvalue $(thisk)")

        if thisk == length(Ms) + 1

            sigma = shift_eps*im
            #sigma = length(Es) > 0 ? Es[end] : 0

            if method == BB

                Ekeep, Hdifs, Y, Yb = doDMRG_IncChi(M, Mb, W, chi_max;
                    normalize_against = [(Ms[i],Mbs[i],-expected_gap*(thisk-i)) for i = 1:thisk-1],
                    sigma=sigma, vt_amp=vt_amp, tol_end=tol, chi_start=chi_start,
                    numsweeps=numsweeps, dispon=dispon, debug=debug, method=method)

                if Hdifs[end] < 1e-3
                    fprintln("Found eigenvalue $thisk = $(fmtcpx(Ekeep[-1]))")
                else
                    fprintln("ERROR: Failed to converge for eigenvalue $thisk: <Delta H^2> = $(fmtf(Hdifs[end]))")
                    if stop_if_not_converge
                        throw(ErrorException())
                    end
                end

                push!(Es, Ekeep[-1])
                push!(Ms, Y)
                push!(Mbs, Yb)

            elseif method == LR

                Ekeep, Hdifs, Y, _ = doDMRG_IncChi(M, Mb, W, chi_max;
                    normalize_against = [(Ms[i],Mbs[i],-expected_gap*(thisk-i)) for i = 1:thisk-1],
                    sigma=sigma, vt_amp=vt_amp, tol_end=tol, chi_start=chi_start,
                    numsweeps=numsweeps, dispon=dispon, debug=debug, method=method)

                if Hdifs[end] < 1e-3
                    fprintln("Found eigenvalue $thisk = $(fmtcpx(Ekeep[-1]))")
                else
                    fprintln("ERROR: Failed to converge for eigenvalue $thisk: <Delta H^2> = $(fmtf(Hdifs[end]))")
                    if stop_if_not_converge
                        throw(ErrorException())
                    end
                end

                push!(Es, Ekeep[-1])
                push!(Ms, Y)

            else
                throw(ArgumentError("Unrecognized method: $method"))
            end

            dosave()

        end

        if thisk != length(Ms)
            throw(ErrorException("Unexpected: thisk = $(thisk), length(Ms) = $(length(Ms))."))
        end

        if thisk == length(Mbs) + 1 && method == LR

            # Right-normalize the M solution
            _,_, Z, Zb = doDMRG(Ms[thisk], conj.(Ms[thisk]), W, chi_max; numsweeps=0, updateon=false)

            Ekeep, Hdifs, Y, _ = doDMRG_IncChi(Z, Zb, W, chi_max;
                    normalize_against = [(Ms[i],Mbs[i],-expected_gap*(thisk-i)) for i = 1:thisk-1],
                    sigma = shift_eps*im, vt_amp=vt_amp, tol_end=tol, chi_start=chi_start,
                    numsweeps=numsweeps, dispon=dispon, debug=debug, method=method)

            if Hdifs[end] < 1e-3
                fprintln("Found eigenvalue $thisk = $(fmtcpx(Ekeep[end]))")
            else
                fprintln("ERROR: Failed to converge for eigenvalue $thisk: <Delta H^2> = $(fmtf(Hdifs[end]))")
                if stop_if_not_converge
                    throw(ErrorException())
                end
            end

            push!(Mbs, Y)

        end

        if thisk != length(Mbs)
            throw(ErrorException("Unexpected: thisk = $(thisk), length(Mbs) = $(length(Mbs))."))
        end

        dosave()

    end

    return Ms, Mbs, Es

end


"""
    function doDMRG_IncChi(M::MPS, Mb::MPS, W::MPO, chi_max::Int;
        chi_inc::Int = 10, chi_start::Int = 20, init_sweeps::Int = 5, inc_sweeps::Int = 2,
        tol_start::Float64 = 1e-3, tol_end::Float64 = 1e-6, vt_amp::Int = 3, vt_sweeps::Int = 3,
        numsweeps::Int = 10, dispon::Int = 2, debug = false, method::DM_Method = LR,
        sigma::ComplexF64 = shift_eps*im, normalize_against = [])

Do DMRG with increasing bond dimensions and decreasing tolerance.
The precedure is running the following DMRGs in sequence:
- chi = chi_start, tol = tol_start, numsweeps = init_sweeps
- chi = chi + chi_inc, tol = tol_start, numsweeps = inc_sweeps
- chi = chi + 2*chi_inc, tol = tol_start, numsweeps = inc_sweeps
- ...
- chi = chi_max, tol = tol_start, numsweeps = vt_sweeps
- chi = chi_max, tol = tol_start*10^(-vt_amp), numsweeps = vt_sweeps
- chi = chi_max, tol = tol_start*10^(-2*vt_amp), numsweeps = vt_sweeps
- ...
- chi = chi_max, tol = tol_end, numsweeps = numsweeps

More more specifications, see `?doDMRG`.
"""
function doDMRG_IncChi(M::MPS, Mb::MPS, W::MPO, chi_max::Int;
    chi_inc::Int = 10, chi_start::Int = 20, init_sweeps::Int = 5, inc_sweeps::Int = 2,
    tol_start::Float64 = 1e-3, tol_end::Float64 = 1e-6, vt_amp::Int = 3, vt_sweeps::Int = 3,
    numsweeps::Int = 10, dispon::Int = 2, debug = false, method::DM_Method = LR,
    sigma::ComplexF64 = shift_eps*im, normalize_against = [])

    _,_,M,Mb = doDMRG(M, Mb, W, chi_start;
        tol=tol_start, numsweeps=init_sweeps, dispon=dispon, updateon=true,
        debug=debug, method=method, normalize_against=normalize_against, sigma=sigma)
    
    chi = chi_start + chi_inc
    while chi < chi_max
        _,_,M,Mb = doDMRG(M, Mb, W, chi;
            tol=tol_start, numsweeps=inc_sweeps, dispon=dispon, updateon=true,
            debug=debug, method=method, normalize_against=normalize_against, sigma=sigma)
        chi += chi_inc
    end

    chi = chi_max
    tol = tol_start
    while tol > tol_end
        _,_,M,Mb = doDMRG(M, Mb, W, chi;
            tol=tol, numsweeps=vt_sweeps, dispon=dispon, updateon=true,
            debug=debug, method=method, normalize_against=normalize_against, sigma=sigma)
        tol *= (10.)^(-vt_amp)
    end

    return doDMRG(M, Mb, W, chi_max; tol=tol_end, numsweeps=numsweeps,
        dispon=dispon, updateon=true, debug=debug, method=method, normalize_against=normalize_against, sigma=sigma)

end

"""
    function doDMRG(M::MPS, Mb::MPS, W::MPO, chi_max::Int;
        numsweeps::Int = 10, sigma::ComplexF64 = shift_eps*im, dispon = 2, updateon = true, debug = false,
        method::DM_Method = LR, tol::Float64=0., normalize_against = [], stop_if_not_converge::Bool=false)

Function does DMRG to find the eigenstate of a Hamiltonian given by the MPO W, with eigenvalue closest to sigma.

# Arguments
- `M`: MPS, initial guess for right eigenvector.
- `Mb`: MPS, initial guess for left eigenvector.
- `W`: MPO, Liouvillian.
- `chi_max`: Int, maximal bond dimension.
- `numsweeps`: Int, default 10. Number of sweeps.
- `sigma`: ComplexF64, default shift_eps*im. Target eigenvalue.
- `dispon`: Int, default 2. Level of display. 2 prints full information, 1 prints partial information, 0 mutes the output.
- `updateon`: Bool, default true. Whether to update the MPS during the DMRG process. If false, simply do normalization to the MPS.
- `debug`: Bool, default false. Whether to print debug information.
- `method`: DM_Method, default LR. Method to truncate the density matrix. Options: LR, BB. See `?decomp()`.
- `tol`: Float64, default 0. Tolerance when diagonalizing a local block.
- `normalize_against`: Array{Tuple{MPS, MPS, ComplexF64}, 1}, default []. Normalize the MPS against the given states.
    If given, do DMRG with respect to the Hamiltonian H + sum_i amp[i] * Mi * Mib, where amp[i], Mi and Mib are the i-th element in the array.
- `stop_if_not_converge`: Bool, default false. Whether to throw an error if the DMRG does not converge.

# Returns
- `Ekeep`: Array{ComplexF64, 1}, the energy at each sweep.
- `Hdifs`: Array{Float64, 1}, the variance in energy at each sweep.
- `M`: MPS, the right eigenvector.
- `Mb`: MPS, the left eigenvector.
"""
function doDMRG(M::MPS, Mb::MPS, W::MPO, chi_max::Int;
    numsweeps::Int = 10, sigma::ComplexF64 = shift_eps*im, dispon = 2, updateon = true, debug = false,
    method::DM_Method = LR, tol::Float64=0., normalize_against = [], stop_if_not_converge::Bool=false)
 
    converged = false

    ##### left-to-right 'warmup', put MPS in right orthogonal form
    # Index of W is: left'' - right'' - physical' - physical
    # Index notation: no prime = ket, one prime = bra, two primes = operator link
    Nsites = length(M)
    if length(Mb) != Nsites
        throw(ArgumentError("Length of M and Mb must match!"))
    end

    sites, links = find_sites_and_links(M)

    physdim = dim(sites[1])
    trivSites = Int(floor(log(physdim, chi_max))) # The first and last trivSites need not be sweeped.

    # Each element in normalize_against should be a tuple (Mi, Mib, amp)
    # Corresponding to adding a term amp * Mi*Mib to the Hamiltonian
    # For this we record LNA, RNA, LNAb, RNAb
    # LNA[i] corresponds to the product of Mib with the current M at site i
    # Simialr for the other three
    NumNA = length(normalize_against)
    LNA = Vector{ITensor}[]
    RNA = Vector{ITensor}[]
    LNAb = Vector{ITensor}[]
    RNAb = Vector{ITensor}[]
    Namp = ComplexF64[]
    MN = MPS[]
    MNb = MPS[]

    unit_itensor = ITensor(ComplexF64(1))

    for (i,item) in enumerate(normalize_against)
        push!(LNA, fill(unit_itensor, Nsites))
        push!(RNA, fill(unit_itensor, Nsites))
        push!(LNAb, fill(unit_itensor, Nsites))
        push!(RNAb, fill(unit_itensor, Nsites))
        push!(MN, item[1])
        push!(MNb, item[2])
        push!(Namp, item[3])
    end
    
    # The L[i] operator is the MPO contracted with the MPS and its dagger for sites <= i-1
    # R[i] is contracted for sites >= i+1
    L = fill(unit_itensor, Nsites)
    R = fill(unit_itensor, Nsites)

    pos = trivSites+1

    for p = Nsites:-1:pos+1 # Do right normalization, from site Nsites to trivSites+2

        # Shape of M is: left bond - physical bond - right bond
        if size(M[p]) != size(Mb[p])
            throw(ArgumentError("Shapes of M[p] and Mb[p] must match!"))
        end
        
        # Set the p-th matrix to right normal form, and multiply the transform matrix to p-1
        M[p], Mb[p], I, Ib, links[p-1] = decomp(M[p], Mb[p], links[p-1]; chi_max=chi_max, timing=debug, method=method, return_newlink=true) # linkind=0 means last

        M[p-1] = M[p-1]*I
        Mb[p-1] = Mb[p-1]*Ib

        # Construct R[p-1]. The indices of R is: left'' - left' - left
        R[p-1] = Mb[p]*R[p]*W[p]*M[p]

        if debug
            fprintln("M[$p] = ", inds(M[p]))
            fprintln("Mb[$p] = ", inds(Mb[p]))
            fprintln("I[$p] = ", inds(I))
            fprintln("Ib[$p] = ", inds(Ib))
            fprintln("M[$(p-1)] = ",inds(M[p-1]))
            fprintln("Mb[$(p-1)] = ", inds(Mb[p-1]))
            fprintln("R[$(p-1)] = ", inds(R[p-1]))
        end

        for i = 1:NumNA

            if debug
                fprintln("MN[$i][$p] = ", inds(MN[i][p]))
                fprintln("MNb[$i][$p] = ", inds(MNb[i][p]))
                fprintln("RNA[$i][$p] = ", inds(RNA[i][p]))
                fprintln("RNAb[$i][$p] = ", inds(RNAb[i][p]))
            end

            RNA[i][p-1] = RNA[i][p]*MNb[i][p]*M[p]*delta(sites[p],sites[p]')
            RNAb[i][p-1] = RNAb[i][p]*MN[i][p]*Mb[p]*delta(sites[p],sites[p]')
        end

    end

    for p = 1:pos-1 # Do left normalization, from site 1 to trivSites

        if size(M[p]) != size(Mb[p])
            throw(ArgumentError("Shapes of M[p] and Mb[p] must match!"))
        end
        
        # Set the p-th matrix to left normal form, and multiply the transform matrix to p+1
        M[p], Mb[p], I, Ib, links[p] = decomp(M[p], Mb[p], links[p]; chi_max=chi_max, timing=debug, method=method, return_newlink=true) # linkind=0 means last

        M[p+1] = M[p+1]*I
        Mb[p+1] = Mb[p+1]*Ib

        # Construct L[p+1]. The indices of R is: left'' - left' - left
        L[p+1] = Mb[p]*L[p]*W[p]*M[p]

        if debug
            fprintln("links = ", links)
            fprintln("M[$p] = ", inds(M[p]))
            fprintln("Mb[$p] = ", inds(Mb[p]))
            fprintln("I[$p] = ", inds(I))
            fprintln("Ib[$p] = ", inds(Ib))
            fprintln("M[$(p+1)] = ",inds(M[p+1]))
            fprintln("Mb[$(p+1)] = ", inds(Mb[p+1]))
            fprintln("L[$(p+1)] = ", inds(L[p+1]))
        end

        for i = 1:NumNA
            LNA[i][p+1] = LNA[i][p]*MNb[i][p]*M[p]*delta(sites[p],sites[p]')
            LNAb[i][p+1] = LNAb[i][p]*MN[i][p]*Mb[p]*delta(sites[p],sites[p]')
        end

    end

    # Normalize M[pos] and Mb[pos] so that the trial wave functions are bi-normalized
    ratio = 1/sqrt((Mb[pos]*(M[pos]'))[])
    M[pos] *= ratio
    Mb[pos] *= ratio

    # At this point we have turned M[2:end] to right normal form, and constructed R[2:end]
    # We start the sweep at site 0
    # The effective Hamiltonian at site [i] is the contraction of L[i], R[i], and W[i]
    
    Ekeep = ComplexF64[]
    Hdifs = Float64[]

    k = 1

    while k < numsweeps+1
        
        ##### final sweep is only for orthogonalization (disable updates)
        if k == numsweeps+1
            updateon = false
            dispon = 0
        end
        
        ###### Optimization sweep: left-to-right
        for p = pos:Nsites-pos

            # Optimize at this step
            if updateon
                E, M[p], Mb[p] = eigLR(L[p], R[p], W[p], M[p], Mb[p],
                    sigma = sigma, use_sparse = true, tol = tol, timing = debug,
                    normalize_against = [(LNA[i][p],RNA[i][p],MNb[i][p],LNAb[i][p],RNAb[i][p],MN[i][p],Namp[i]) for i = 1:NumNA])    
                push!(Ekeep, E)
            end

            # Move the pointer one site to the right, and left-normalize the matrices at the currenter pointer
            M[p], Mb[p], I, Ib, links[p] = decomp(M[p], Mb[p], links[p]; chi_max=chi_max, timing=debug, method=method, return_newlink=true)

            M[p+1] = I*M[p+1]
            Mb[p+1] = Ib*Mb[p+1]

            # Construct L[p+1]
            L[p+1] = Mb[p]*L[p]*W[p]*M[p]

            for i = 1:NumNA
                LNA[i][p+1] = LNA[i][p]*MNb[i][p]*M[p]*delta(sites[p],sites[p]')
                LNAb[i][p+1] = LNAb[i][p]*MN[i][p]*Mb[p]*delta(sites[p],sites[p]')
            end

            if debug
                fprintln("links = ", links)
                fprintln("M[$p] = ", inds(M[p]))
                fprintln("Mb[$p] = ", inds(Mb[p]))
                fprintln("I[$p] = ", inds(I))
                fprintln("Ib[$p] = ", inds(Ib))
                fprintln("M[$(p+1)] = ",inds(M[p+1]))
                fprintln("Mb[$(p+1)] = ", inds(Mb[p+1]))
                fprintln("L[$(p+1)] = ", inds(L[p+1]))
            end
        
            ##### display energy
            if dispon == 2
                fprintln("Sweep: $k of $numsweeps, Loc: $p, chi: $chi_max, Energy: $(fmtcpx(Ekeep[end]))")
            end
            
        end
        
        ###### Optimization sweep: right-to-left
        for p = Nsites-pos+1:-1:pos+1

            # Optimize at this step
            if updateon
                E, M[p], Mb[p] = eigLR(L[p], R[p], W[p], M[p], Mb[p],
                    sigma = sigma, use_sparse = true, tol = tol, timing = debug,
                    normalize_against = [(LNA[i][p],RNA[i][p],MNb[i][p],LNAb[i][p],RNAb[i][p],MN[i][p],Namp[i]) for i = 1:NumNA])    
                push!(Ekeep, E)
            end

            # Move the pointer one site to the left, and right-normalize the matrices at the currenter pointer
            M[p], Mb[p], I, Ib, links[p-1] = decomp(M[p], Mb[p], links[p-1]; chi_max=chi_max, timing=debug, method=method, return_newlink=true)
            M[p-1] = M[p-1]*I
            Mb[p-1] = Mb[p-1]*Ib

            # Construct R[p-1]. The indices of R is: left'' - left - left'
            R[p-1] = Mb[p]*R[p]*W[p]*M[p]

            for i = 1:NumNA
                RNA[i][p-1] = RNA[i][p]*MNb[i][p]*M[p]*delta(sites[p],sites[p]')
                RNAb[i][p-1] = RNAb[i][p]*Mb[p]*MN[i][p]*delta(sites[p],sites[p]')
            end

            if debug
                fprintln("M[$p] = ", inds(M[p]))
                fprintln("Mb[$p] = ", inds(Mb[p]))
                fprintln("I[$p] = ", inds(I))
                fprintln("Ib[$p] = ", inds(Ib))
                fprintln("M[$(p-1)] = ",inds(M[p-1]))
                fprintln("Mb[$(p-1)] = ", inds(Mb[p-1]))
                fprintln("R[$(p-1)] = ", inds(R[p-1]))
            end
        
            ##### display energy
            if dispon == 2
                fprintln("Sweep: $k of $numsweeps, Loc: $p, chi: $chi_max, Energy: $(fmtcpx(Ekeep[end]))")
            end

        end
        
        # Calculate <H^2>-<H>^2
        RR = ITensor(1. +0im)
        _,Wlinks = find_sites_and_links(W)
        newlinks = Vector{Index}(undef, Nsites-1)
        for p = Nsites:-1:1
            physind = sites[p]
            newphys = Index(dim(physind), tags(physind))
            Wlow = W[p]*delta(physind',newphys) # Lower W has indices (physind, newphys, links), contracts with M
            Wupp = W[p]*delta(physind,newphys) # Upper W has indices (newphys, physind', links), contracts with Mb
            # Renew the left link
            if p > 1
                leftlink = Wlinks[p-1]
                newlinks[p-1] = Index(dim(leftlink), join(tags(leftlink), ",")*",upp")
                Wupp *= delta(leftlink, newlinks[p-1])
            end
            # Renew the right link
            if p < Nsites
                Wupp *= delta(Wlinks[p], newlinks[p])
            end
            if debug
                fprintln("At step $p, inds(Wlow) = $(inds(Wlow))")
                fprintln("At step $p, inds(Wupp) = $(inds(Wupp))")
                fprintln("At step $p, inds(M) = $(inds(M[p]))")
                fprintln("At step $p, inds(Mb) = $(inds(Mb[p]))")
            end

            RR = (Mb[p] * RR * M[p]) * (Wupp * Wlow)
            if debug
                fprintln("At step $p, inds(RR) = $(inds(RR))")
            end
        end

        Hdif = abs(RR[] - Ekeep[end]^2)
        push!(Hdifs, Hdif)

        if dispon >= 1
            fprintln("Sweep: $k of $numsweeps, Energy: $(fmtcpx(Ekeep[end])), Hdif: $(fmtf(Hdif)), Bonddim: $chi_max, tol: $tol")
        end

        cut = max(tol, eps(Float64)) * 10
        # Early termination if converged
        if abs(std(Ekeep[end-(2*Nsites-4*pos+1):end])) < cut && Hdif < cut
            fprintln("Converged")
            converged = true
            k = numsweeps+1
        end

        k += 1

    end

    # Clean up memory
    foreach(finalize, [LNA, RNA, LNAb, RNAb, Namp, MN, MNb, L, R])
    GC.gc()

    if !converged && stop_if_not_converge
        throw(ErrorException("Failed to converge after $numsweeps sweeps."))
    end
            
    return Ekeep, Hdifs, M, Mb

end

"""
    function doIDMRG(M::MPS, Mb::MPS, W::MPO, chi_max::Int;
        numsweeps::Int = 10, sigma::ComplexF64 = shift_eps*im, dispon = 2, updateon = true, debug = false,
        method::DM_Method = LR, tol::Float64=0., normalize_against = [], stop_if_not_converge::Bool=false)

Function does DMRG to find the eigenstate of a Hamiltonian given by the MPO W, with eigenvalue closest to sigma.

# Arguments
- `M`: MPS, initial guess for right eigenvector.
- `Mb`: MPS, initial guess for left eigenvector.
- `W`: MPO, Liouvillian.
- `chi_max`: Int, maximal bond dimension.
- `numsweeps`: Int, default 10. Number of sweeps.
- `sigma`: ComplexF64, default shift_eps*im. Target eigenvalue.
- `dispon`: Int, default 2. Level of display. 2 prints full information, 1 prints partial information, 0 mutes the output.
- `updateon`: Bool, default true. Whether to update the MPS during the DMRG process. If false, simply do normalization to the MPS.
- `debug`: Bool, default false. Whether to print debug information.
- `method`: DM_Method, default LR. Method to truncate the density matrix. Options: LR, BB. See `?decomp()`.
- `tol`: Float64, default 0. Tolerance when diagonalizing a local block.
- `normalize_against`: Array{Tuple{MPS, MPS, ComplexF64}, 1}, default []. Normalize the MPS against the given states.
    If given, do DMRG with respect to the Hamiltonian H + sum_i amp[i] * Mi * Mib, where amp[i], Mi and Mib are the i-th element in the array.
- `stop_if_not_converge`: Bool, default false. Whether to throw an error if the DMRG does not converge.

# Returns
- `Ekeep`: Array{ComplexF64, 1}, the energy at each sweep.
- `Hdifs`: Array{Float64, 1}, the variance in energy at each sweep.
- `M`: MPS, the right eigenvector.
- `Mb`: MPS, the left eigenvector.
"""
function doIDMRG(W::Array{<:Number, 4}, chi_max::Int; Wleftindex::Int = 1, Wrightindex::Int = 2,
    sigma::ComplexF64 = shift_eps*im, dispon::Int = 2, maxL::Int = 0, converge_tol::Float64 = 1e-8,
    method::DM_Method = LR, tol::Float64=0., normalize_against = [], return_M::Bool = false)

    debug = (dispon > 2)
    physdim = size(W)[1]
    Wlinkdim = size(W)[3]

    L0 = max(5, Int(ceil(log(physdim, chi_max)))+1) # Initial diagonalization

    # Solve a L0-sized system first
    sitesL = [Index(physdim, "Site::$i") for i = 1:L0]
    mpo1 = MPOonSites(W, sitesL; leftindex=Wleftindex, rightindex=Wrightindex)
    W_L0_mat, comb = MPO_to_Matrix(mpo1)
    Mci = inds(comb)[1]
    fprintln("Solving initial system with L0 = $L0")
    # For now only implement k=1
    wR, vR = doeig(W_L0_mat, ComplexF64[randn() + im * randn() for _ in 1:L0]; sigma=sigma, k=chi_max, use_sparse=false)
    wL, vL = doeig(transpose(W_L0_mat), ComplexF64[randn() + im * randn() for _ in 1:L0]; sigma=sigma, k=chi_max, use_sparse=false)
    fprintln("Found right eigenvalues: ", wR)
    fprintln("Found left eigenvalues: ", wL)
    if !isapprox(wR, wL; atol=1e-6)
        throw(ErrorException("Eigenvalues of W_L0 are not symmetric!"))
    end

    link1 = Index(chi_max, "Link-dummy") # The link that corresponds to eigenvalue index in wR and wL
    vR_tens = ITensor(vR, Mci, link1)*comb
    vL_tens = ITensor(vL, Mci', link1')*comb'
    # Construct the initial MPS on the left
    Y, Yb, MlinkL = decomp2MPS(vR_tens, vL_tens, sitesL;
        chi_max=chi_max, timing=debug, method=method, linknames=["$j~Mlink~$(j+1)" for j=1:L0])

    sitesR = [Index(physdim, "Site::$i") for i = -1:-1:-L0]
    vR_tens = replaceindsP(vR_tens, sitesL, sitesR)
    vL_tens = replaceindsP(vL_tens, sitesL, sitesR; plvl=1)
    # Construct the initial MPS on the right
    Z, Zb, MlinkR = decomp2MPS(vR_tens, vL_tens, sitesR;
        chi_max=chi_max, timing=debug, method=method, linknames=["$j~Mlink~$(j+1)" for j=-2:-1:-(L0+1)])

    # If we need to keep track of M, create arrays that store the M's
    # ML keeps track on the tensors on the left of the current site
    # MR keeps track on the tensors on the right of the current site, from the rightmost site to the current site
    # sitesL and sitesR keeps track of the physical indices of the MPS
    if return_M
        ML = [i for i in Y]
        MR = [i for i in Z]
        MbL = [i for i in Yb]
        MbR = [i for i in Zb]
    end

    # Array to save energy
    Es = ComplexF64[]

    # A is the tensor at the middle, the contraction of two MPS blocks
    # It should have indices: MlinkL, MlinkR, siteL[end], siteR[end]
    # To this end, we replaced the dangling links of A with MlinkL and MlinkR
    oMlinkL = setdiffS(inds(Y[end]), [MlinkL, sitesL[end]])
    oMlinkR = setdiffS(inds(Z[end]), [MlinkR, sitesR[end]])
    A = replaceinds(Y[end]*Z[end]*delta(MlinkL, MlinkR),
        oMlinkL=>MlinkL, oMlinkR=>MlinkR)
    Ab = replaceinds(Yb[end]*Zb[end]*delta(MlinkL', MlinkR'),
        oMlinkL'=>MlinkL', oMlinkR'=>MlinkR')

    # Construct tensor L, which is the contraction of the MPO with the MPS and its dagger for the left sights
    mpoL0 = MPOonSites(W, vcat(sitesL, sitesR[end:-1:1]); leftindex=Wleftindex, rightindex=Wrightindex)
    L = ITensor(ComplexF64(1))
    R = ITensor(ComplexF64(1))

    for i = 1:L0
        L = L * Y[i] * mpoL0[i] * Yb[i]
        R = R * Z[i] * mpoL0[2*L0+1-i] * Zb[i]
    end
    
    # Replace the link in the middle
    L_midlink = [i for i in inds(L) if (i != MlinkL && i != MlinkL')]
    if length(L_midlink) != 1
        throw(ErrorException("Unexpected links in L: $(inds(L))"))
    end
    R_midlink = [i for i in inds(R) if (i != MlinkR && i != MlinkR')]
    if length(R_midlink) != 1
        throw(ErrorException("Unexpected links in R: $(inds(R))"))
    end
    WlinkL = Index(Wlinkdim, "$L0~Wlink~$(L0+1)")
    WlinkR = Index(Wlinkdim, "$(-L0-1)~Wlink~$(-L0)")
    L *= delta(L_midlink[1], WlinkL)
    R *= delta(R_midlink[1], WlinkR)

    pos = L0+1

    # From now on do iteration, add two sites and optimize
    while true

        # Collect garbage
        GC.gc()

        # We should have L with indices: MlinkL, MinkL', WlinkL; and R similar.
        # We construct two new sites, siteL and siteR, and two new links, newMlink and newWlink
        # Contract two W's in the middle, and solve an eigenproblem with (MlinkL, siteL, siteR, MlinkR)

        siteL = Index(physdim, "Site::$pos")
        siteR = Index(physdim, "Site::$(-pos)")
        newWlink = Index(Wlinkdim, "Dummy W link $pos")
        WL = ITensor(W, siteL', siteL, WlinkL, newWlink)
        WR = ITensor(W, siteR', siteR, newWlink, WlinkR)

        comb = combiner(MlinkL, siteL, siteR, MlinkR)
        Mci = inds(comb)[1]

        # Initialize the Hamiltonian and the two 
        Ham = L*R*WL*WR
        # Right now A has indices of MlinkL, MlinkR, sitesL[end], sitesR[end]
        A = replaceinds(A, sitesL[end]=>siteL, sitesR[end]=>siteR)
        Ab = replaceinds(Ab, sitesL[end]'=>siteL', sitesR[end]'=>siteR')

        w, A, Ab = eigLR(Ham, A, Ab; sigma=sigma, tol=tol, timing=debug)
        fprintln("Position $pos, got energy $w")

        # Save the old links
        oMlinkL = MlinkL
        oMlinkR = MlinkR
        
        # Decompose A into two MPS blocks
        tML, tMbL, I, Ib, MlinkL = decomp(A, Ab, [siteR, MlinkR];
            chi_max=chi_max, timing=debug, method=method, return_newlink=true, newlink_name="$pos~Mlink~$(pos+1)")
        tMR, tMbR, tI, tIb, MlinkR = decomp(I, Ib, MlinkL;
            chi_max=chi_max, timing=debug, method=method, return_newlink=true, newlink_name="$(-pos-1)~Mlink~$(-pos)")

        # Update L and R
        WlinkL = Index(Wlinkdim, "$pos~Wlink~$(pos+1)")
        WlinkR = Index(Wlinkdim, "$(-pos-1)~Wlink~$(-pos)")
        L *= tML*tMbL*WL*delta(newWlink, WlinkL)
        R *= tMR*tMbR*WR*delta(newWlink, WlinkR)

        # Replace the old links with the new links in A as the guess for the next iteration
        A = replaceinds(A, oMlinkL=>MlinkL, oMlinkR=>MlinkR)
        Ab = replaceinds(Ab, oMlinkL'=>MlinkL', oMlinkR'=>MlinkR')

        if return_M
            push!(ML, tML)
            push!(MR, tMR)
            push!(MbL, tMbL)
            push!(MbR, tMbR)
        end
        push!(Es, w)

        stop = false

        # Convergence test
        if length(Es) > 1 && abs(Es[end]-Es[end-1]) < converge_tol
            fprintln("Converged at position $pos, energy $w")
            stop = true
        end

        # Max iteration test
        if pos >= maxL
            fprintln("Maximal iteration reached at position $pos, energy $w")
            fprintln("No convergence.")
            stop = true
        end

        if stop
            if return_M
                return (;Es=Es, ML=ML, MR=MR, MbL=MbL, MbR=MbR, I=tI, Ib=tIb)
            else
                return (;Es=Es, ML=tML, MR=tMR, MbL=tMbL, MbR=tMbR, I=tI, Ib=tIb)
            end
        end

        # Move to next site
        push!(sitesL, siteL)
        push!(sitesR, siteR)
        pos += 1

    end

end