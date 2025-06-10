# This file realizes the iDMRG algorithm for the 1D quantum spin chain.
# The algorithm is based on the paper:
# https://arxiv.org/pdf/0804.2509
# With modifications to handle non-Hermitian Hamiltonians
# For now only two-site update is used.

include("BiorthoLib.jl")

function iDMRG(H::ITensor)

    # We keep track of several quantities during the algorithm:
    # LamP(b), LamC(b): The Previous and Current Lambda matrices
    # L: The left environment tensor
    # R: The right environment tensor

    
    # Get the vectors spanned by LamP and LamPb
    LPsvd = svd(LamP)
    LPbsvd = svd(LamPb)
    # Bi-orthogonalize the left basis
    A0, A0b, Ll, Llb = SVDbiortho(Matrix(LPsvd.U), Matrix(LPbsvd.U); return_lambda=true)
    # Now A0 and A0b are biorthogonal, and LPsvd.U = A0*Ll
    # So we have LamP = A0 * LamR, where LamR = Ll * LPsvd.S * Lpsvd.Vd
    LamR = Ll * Diagonal(LPsvd.S) * Matrix(LPsvd.V)'
    LamRb = Llb * Diagonal(LPbsvd.S) * Matrix(LPbsvd.V)'
    # Do it similarly on the right
    B0, B0b, Lr, Lrb = SVDbiortho(Matrix(LPsvd.V), Matrix(LPbsvd.V); return_lambda=true)
    LamL = Matrix(LPsvd.U) * Diagonal(LPsvd.S) * Lr' # Here LamP = LamL * B0'
    LamLb = Matrix(LPbsvd.U) * Diagonal(LPbsvd.S) * Lrb'

    # Construct a two-site problem
    H_twosite = 

end