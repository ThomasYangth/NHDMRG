using ITensors
using Printf

USEGPU = false

function fprintln(x...)
    println(x...)
    flush(stdout)
end

function fprint(x...)
    print(x...)
    flush(stdout)
end

function fmtf(x)
    return @sprintf("%.3f", x)
end

function fmtcpx(z)
    return @sprintf("%.3f %+.3fim", real(z), imag(z))
end

function plaindot(v1, v2)
    return transpose(v1)*v2
end

function setdiffS(a, b; errorMsg = "")
    diff = setdiff(a, b)
    if length(diff) != 1
        if errorMsg != ""
            throw(ArgumentError(errorMsg))
        else
            throw(ArgumentError("setdiffS: length(diff) != 1\na = $a, b = $b"))
        end
    end
    return diff[1]
end

export DM_Method, Sites, BB, LR

@enum DM_Method BB LR
const Sites = Vector{Index{Int64}}