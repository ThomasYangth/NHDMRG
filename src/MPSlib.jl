using LinearAlgebra: I

export MPSonSites, MPOonSites, LindbladMPO, LindbladMPO_W

"""
    function MPSonSites(M0::Array{<:Number, 3}, sites::Sites; leftindex::Int=1, rightindex::Int=2)

Construct a translation-invariant MPS from a 3-tensor. The 3-tensor should have indices
in the order (left-link, site, right-link). At the first site, the left-link is contracted to leftindex,
and similar at the last site.

# Arguments
- `M0::Array{<:Number, 3}`: The 3-tensor representing the MPS.
- `sites::Sites`: The site indices.
- `leftindex::Int=1`: The index to contract the left-link at the first site.
- `rightindex::Int=2`: The index to contract the right-link at the last site.

# Returns
- `MPS`: The MPS constructed from the 3-tensor.
"""
function MPSonSites(M0::Array{<:Number, 3}, sites::Sites; leftindex::Int=1, rightindex::Int=2)
    L = length(sites)
    if L == 1
        return MPS([ITensor(M0[leftindex,:,rightindex], sites[1])])
    else
        links = [Index(thisax, "$i-link-$(i+1)" for i=1:L-1)]
        ML = ITensor(M0[leftindex,:,:], sites[1], links[1])
        MR = ITensor(M0[:,:,rightindex], links[end], sites[end])
        return MPS(vcat([ML], [ITensor(M0, links[i-1], sites[i], links[i]) for i = 2:L-1], [MR]))
    end
end

"""
    function MPOonSites(M0::Array{<:Number, 4}, sites::Sites; leftindex::Int=1, rightindex::Int=2)

Similar to MPSonSites. The 4-tensor M0 should have indices in the order (site', site, left-link, right-link).

# Arguments
- `M0::Array{<:Number, 4}`: The 4-tensor representing the MPO.
- `sites::Sites`: The site indices.
- `leftindex::Int=1`: The index to contract the left-link at the first site.
- `rightindex::Int=2`: The index to contract the right-link at the last site.

# Returns
- `MPO`: The MPO constructed from the 4-tensor.
"""
function MPOonSites(M0::Array{<:Number, 4}, sites::Sites; leftindex::Int=1, rightindex::Int=2)
    L = length(sites)
    if L == 1
        return MPO([ITensor(M0[:,:,leftindex,rightindex], sites[1]', sites[1])])
    else
        links = [Index(size(M0)[3], "$i-link-$(i+1)") for i=1:L-1]
        ML = ITensor(M0[:,:,leftindex,:], sites[1]', sites[1], links[1])
        MR = ITensor(M0[:,:,:,rightindex], sites[end]', sites[end], links[end])
        return MPO(vcat([ML], [ITensor(M0, sites[i]', sites[i], links[i-1], links[i]) for i = 2:L-1], [MR]))
    end
end

"""
    function OpSumMPS(op::Operator, sites::Sites)

Create a MPS correspond to a translationally invariant operator given by local terms.

# Arguments
- `op::Operator`: The operator to be represented. Each term is translated on an OBC lattice to all possible positions.
- `sites::Sites`: The site indices.

# Returns
- `MPS`: The MPS representation of the operator.
"""
function OpSumMPS(op::Operator, sites::Sites)

    # Creates a MPS with the following bond dimension:
    # Axis 0 - indicates initial state
    # Axis 1 - indicates final state
    # All operators are added in a transition process 0 -> some axes -> 1
    # If the operator is one-site or two site, "some axes" is one axes
    # Otherwise, the dimensionality of "some axes" equals to (size of operator - 1)

    L = length(sites)

    thisax = 3 # Current axes
    dims = [] # dims[i]:dims[i+1] are the axes for operator i
    spans = [] # spans[i] records the span of operator i
    types = [] # types[i] records the Pauli string of operator i
    vals = [] # vals[i] records the coefficient of operator i
    signs = [] # signs[i] records the sign of operator i

    single_site = [0,0,0,0] # Record the amplitude of single-site operators

    for term in op
        span = len(term)
        if span > 1
            push!(dims, thisax)
            push!(spans, span)
            push!(types, term.inds)
            push!(vals, abs(term.coef)^(1/span))
            push!(signs, sign(term.coef))
            thisax += (span - 1)
        else
            single_site[term.inds[1]] += val
        end
    end

    # The bulk MPS tensor
    M0 = ComplexF64.(zeros(thisax, 4, thisax))
    M0[1, 1, 1] = 1 # Initial state to itself
    M0[2, 1, 2] = 1 # Final state to itself

    for i = 1:4
        M0[1, i, 2] = single_site[i] # Initial state can directly hop to final state, yielding a single-site operator
    end

    for (i,dim) in enumerate(dims)
        span = spans[i]
        val = vals[i]
        type_indices = types[i]
        M0[1, type_indices[1], dim] = val * signs[i] # Initial state to transition axes
        M0[dim + span - 2, type_indices[end], 1] = val # Transition axes to final state
        for j in 1:span-2
            M0[dim + j - 1, type_indices[j], dim + j] = val # Within transition axes
        end
    end

    return MPSonSites(M0, sites; leftindex=1, rightindex=2)

end

"""
    sizeMPO(sites::Array{Index})

Returns a MPO that measures the size of an operator.
"""
function sizeMPO(sites::Sites)

    M0 = ComplexF64.(zeros(4, 4, 2, 2))
    M0[:,:, 1,1] = Matrix(I, 4, 4) # Initial state to itself
    M0[:,:, 2,2] = Matrix(I, 4, 4) # Final state to itself
    M0[:,:, 1,2] = Diagonal([0,1,1,1]) # Operator size

    return MPOonSites(M0, sites; leftindex=1, rightindex=2)

end

"""
    function LindbladMPO_W(H, Lis; dagger = false)

Realized in a similar way as OpSumMPS. Returns the 4-tensor W, instead of a MPO.

# Arguments
- `H::Operator`: The Hamiltonian.
- `Lis::Array{Operator}`: The jump operators.
- `dagger::Bool=false`: if false, returns the Schrodinger-picture Lindbladian which acts on density matrices.
    If true, returns the Heisenberg-picture Lindbladian which acts on operators.

# Returns
- `W::Array{ComplexF64, 4}`: The 4-tensor representing the MPO of the Lindbladian.
"""
function LindbladMPO_W(H, Lis; dagger = false)
    thisax = Ref(3) # Current axes
    dims = Ref([]) # dims[i]:dims[i+1] are the axes for operator i
    mats = Ref([])

    single_site = Ref(ComplexF64.(zeros(4,4))) # Record the amplitude of single-site operators

    function add_mats(mat)
        span = length(mat)
        if span > 1
            push!(dims[], thisax[])
            push!(mats[], mat)
            thisax[] += (span-1)
        elseif span == 1
            single_site[] += mat[1]
        end
    end

    # Hamiltonian
    for term in H
        if dagger
            add_mats(getMats(term; type="L", add_coef=1im))
            add_mats(getMats(term; type="R", add_coef=-1im))
        else
            add_mats(getMats(term; type="L", add_coef=-1im))
            add_mats(getMats(term; type="R", add_coef=1im))
        end
    end
        
    function mul_mats(mat1, mat2)
        return [mat1[i]*mat2[i] for i in eachindex(mat1)]
    end

    # Jump operators
    for Li in Lis
        for tL0 in Li
            for tR in Li
                tL = conj(tL0)
                if dagger
                    add_mats(mul_mats(getMats(tL; type="L"), getMats(tR; type="R")))
                else
                    add_mats(mul_mats(getMats(tL; type="R"), getMats(tR; type="L")))
                end
                add_mats(mul_mats(getMats(tL; type="L", add_coef=-1/2), getMats(tR; type="L")))
                add_mats(mul_mats(getMats(tR; type="R"), getMats(tL; type="R", add_coef=-1/2)))
            end
        end
    end

    # The bulk MPO tensor
    M0 = ComplexF64.(zeros(4, 4, thisax[], thisax[]))
    M0[:,:, 1,1] = Matrix(I, 4, 4) # Initial state to itself
    M0[:,:, 2,2] = Matrix(I, 4, 4) # Final state to itself
    M0[:,:, 1,2] = single_site[] # Initial state can directly hop to final state, yielding a single-site operator
    for (i,dim) in enumerate(dims[])
        mat = mats[][i]
        span = length(mat)
        M0[:,:, 1,dim] = mat[1] # Initial state to transition axes
        M0[:,:, dim+span-2,2] = mat[end] # Transition axes to final state
        for j = 1:span-2
            M0[:,:, dim+j-1,dim+j] = mat[j] # Within transition axes   
        end
    end

    return M0

end

"""
    function LindbladMPO(H, Lis, sites; dagger = false)

Realized in a similar way as OpSumMPS.
"""
function LindbladMPO(H, Lis, sites; dagger = false)

    return MPOonSites(LindbladMPO_W(H, Lis; dagger=dagger), sites; leftindex=1, rightindex=2)

end

"""
    function printMPSO(W)

Print the indices and the array value of a MPS or MPO.
"""
function printMPSO(W)
    H = reduce(*, W)
    Winds = inds(H)
    fprintln("Indices: $inds")
    fprintln(Array(H, Winds...))
end

function MPS_to_Vector(W)
    H = reduce(*, W)
    Winds = inds(H)
    comb = combiner(Winds...)
    mati = inds(comb)[1]
    return Array(H*comb, mati)
end

function MPO_to_Matrix(W)
    H = reduce(*, W)
    Winds = [i for i in inds(H) if plev(i)==0]
    comb = combiner(Winds...)
    mati = inds(comb)[1]
    return Array(H*comb*comb', mati', mati), comb
end

function printInds(W)
    for (p,A) in enumerate(W)
        fprintln("Pos $p, indices $(inds(A))")
    end
end

"""
    find_sites_and_links(M::Union{MPS,MPO})

Given a MPS or MPO, returns the site indices and link indices.

# Arguments
- `M::Union{MPS,MPO}`: The MPS or MPO to be analyzed.

# Returns
- `sites::Vector{Index{Int}}`: The site indices.
- `links::Vector{Index{Int}}`: The link indices.
"""
function find_sites_and_links(M::Union{MPS,MPO})

    Nsites = length(M)

    links = Vector{Index{Int}}(undef, Nsites-1)
    sites = Vector{Index{Int}}(undef, Nsites)

    Minds = [[id for id in inds(M[i]) if plev(id)==0] for i in 1:Nsites]

    # Obtain the link indices
    for i = 1:Nsites
        if i < Nsites
            common_inds = intersect(Minds[i], Minds[i+1])
            if length(common_inds) != 1
                throw(ArgumentError("MPS tensors must have exactly one common index!\nInstead, inds(M[$i])=$(inds(M[i])), inds(M[$(i+1)])=$(inds(M[i+1]))."))
            end
            links[i] = common_inds[1]
        end
        i_site = setdiff(Minds[i], links)
        if length(i_site) != 1
            throw(ArgumentError("MPS tensors must have exactly one site index!\nInstead, inds(M[$i])=$(inds(M[i]))."))
        end
        sites[i] = i_site[1]
    end

    return sites, links

end

function replaceindsP(A::ITensor, oldinds::Sites, newinds::Sites; plvl::Int=0)

    if length(oldinds) != length(newinds)
        throw(ArgumentError("The number of old indices must be the same as the number of new indices!"))
    end

    A = deepcopy(A)

    for i in eachindex(oldinds)
        for j = 0:plvl
            replaceind!(A, prime(oldinds[i], j), prime(newinds[i], j))
        end
    end

    return A
end

function renameinds(A::ITensor, inds::Sites, newnames::Vector{String}; plvl::Int=0)

    if length(inds) != length(newnames)
        throw(ArgumentError("The number of old indices must be the same as the number of new names!"))
    end

    newinds = [Index(dim(inds[i]), newnames[i]) for i in eachindex(inds)]
    return replaceinds(A, inds, newinds; plvl=plvl), newinds
    
end