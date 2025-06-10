"""
    two_sided_krylov_schur(A, v1, w1, m, ell; max_restarts=50, tol=1e-6)

Compute a two‐sided Krylov–Schur decomposition for a nonnormal matrix A ∈ ℂⁿˣⁿ.
v1 and w1 are starting (normalized) vectors for the right and left Krylov spaces.
m is the desired (minimum) dimension and ell (≥ m) is the maximum expansion dimension.
The function returns (V, W, H, K) such that A*V ≈ V*H and A'*W ≈ W*K.
"""
function two_sided_krylov_schur(A, v1, w1, m, l; max_restarts=50, tol=1e-6, eig_selector::Function=abs)
    n = size(A,1)
    # Normalize starting vectors (if not already)
    v1 = v1 / norm(v1)
    w1 = w1 / norm(w1)
    
    # Initialize the right and left basis matrices with first vectors.
    V = zeros(ComplexF64, l, l)
    W = zeros(ComplexF64, l, l)
    vlp = zeros(ComplexF64, l)
    wlp = zeros(ComplexF64, l)
    
    # Initialize empty Hessenberg matrices (will be built column by column)
    H = zeros(ComplexF64, l, l)
    K = zeros(ComplexF64, l, l)
    h = zeros(ComplexF64, l)
    k = zeros(ComplexF64, l)

    # Set the first column of V and W
    V[:,1] = v1
    W[:,1] = w1

    Ad = A' # Dagger of A
    
    # Restart loop
    for restart in 1:max_restarts

        # === Step 2: Expand the Krylov decompositions to dimension l ===
        # Here we perform simultaneous two-sided Arnoldi expansions.

        this_l = l
        
        # For indices starting at 2 up to l+1,
        # perform Arnoldi iterations for both right and left spaces.
        for j = 2:l+1

            # --- Right Arnoldi: compute new vector from A*v_j ---
            v = A * V[:,j-1]
            # Gram-Schmidt Orthogonalize against previous vector
            overlaps = V[:,1:j-1]' * v
            v -= V[:,1:j-1] * overlaps
            # Normalize the vector
            norm_v = norm(v)
            if norm_v < tol
                # early termination of Arnoldi expansion
                # this_l = j - 1
                # break
            end
            H[1:j-1,j-1] = overlaps
            if j <= l
                V[:,j] = v ./ norm_v
                # Write in the Hessenberg matrix
                H[j,j-1] = norm_v
            else
                # Write to the (l+1)-th column / row
                vlp = v ./ norm_v
                h[j] = norm_v
                h[1:j-1] = 0
            end

            # Verify validity of the right Arnoldi expansion
            if j <= l
                loss = norm(A*V[:,1:j-1] - V[:,1:j]*H[1:j,1:j-1])
            else
                loss = norm(A*V - V*H - vlp*h')
            end
            fprintln("Right Arnoldi loss: $loss")
            
            # --- Left Arnoldi: compute new vector from A'*w_j ---
            w = Ad * W[:,j-1]
            # Gram-Schmidt Orthogonalize against previous vector
            overlaps = W[:,1:j-1]' * w
            w -= W[:,1:j-1] * overlaps
            # Normalize the vector
            norm_w = norm(w)
            if norm_w < tol
                # early termination of Arnoldi expansion
                # this_l = j - 1
                # break
            end
            if j <= l 
                W[:,j] = w ./ norm_w
                # Write in the Hessenberg matrix
                K[j,j-1] = norm_w
                K[1:j-1,j-1] = overlaps
            else
                # Write to the (l+1)-th column / row
                wlp = w ./ norm_w
                k[j] = norm_w
                k[1:j-1] = overlaps
            end

        end

        M = W' * V
        dv = M \ (W' * vlp) # dv = M^(-1) * W' * vlp
        H += dv * h'
        vlp -= V * dv
        dw = M' \ (V' * wlp) # dw = M^(-1)^dagger * V' * wlp
        K += dw * k'
        wlp -= W * dw
        
        # === Step 5: Schur decompositions of H_ext and K_ext ===
        schurH = schur(H)
        args = sortperm(eig_selector.(diag(schurH.T))) # Sort the eigenvalues of T in increasing order
        select_pos = fill(false, l)
        select_pos[args[1:m]] .= true # select_pos is true for the best chi eigenvalues
        S,Q = ordschur(schurH, select_pos)

        schurK = schur(K)
        args = sortperm(eig_selector.(diag(schurK.T))) # Sort the eigenvalues of T in increasing order
        select_pos = fill(false, l)
        select_pos[args[1:m]] .= true # select_pos is true for the best chi eigenvalues
        T,Z = ordschur(schurK, select_pos)
        
        # === Step 6: Partition Q, S, Z, T ===
        # We select the m leading components.
        Q1 = Q[:,1:m]
        Z1 = Z[:,1:m]
        
        # === Step 7: Set V_m, H_m, and compute h_m ===
        V_m = V * Q1
        # The coefficient vector for the residual is:
        h = Q1' * h  # note: h* in pseudocode
        # === Step 8: Set W_m, K_m, and compute k_m ===
        W_m = W * Z1
        k = Z1' * k
        
        # === Step 9: Set M_m ===
        M_m = Z1' * M * Q1

        H_m = S[1:m,1:m] + (V_m'*vlp)*h'
        K_m = T[1:m,1:m] + (W_m'*wlp)*k'
        vlp -= 

        # === Step 10: 
        Update H_m and v_{m+1} ===
        # Compute the projection of the next right vector onto V_m:
        proj_right = V_m' * v_next
        # Update H_m with the correction term:
        H_m += proj_right * h_vec'
        # Orthogonalize v_next against V_m:
        v_res = v_next - V_m * proj_right
        norm_v_res = norm(v_res)
        if norm_v_res > tol
            v_res = v_res / norm_v_res
        end
        
        # === Step 11: Update K_m and w_{m+1} ===
        proj_left = W_m' * w_next
        K_m += proj_left * k_vec'
        w_res = w_next - W_m * proj_left
        norm_w_res = norm(w_res)
        if norm_w_res > tol
            w_res = w_res / norm_w_res
        end
        
        # === Step 12: Check convergence ===
        # Here one might check the residual norms or changes in eigenvalues.
        if norm_v_res < tol && norm_w_res < tol
            println("Converged after $restart restarts.")
            return (V_m, W_m, H_m, K_m)
        end

        # Restart: Use the new approximate vectors as starting vectors
        V = V_m
        W = W_m
    end
    println("Maximum restarts reached without full convergence.")
    return (V_m, W_m, H_m, K_m)
end