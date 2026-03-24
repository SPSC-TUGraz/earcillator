function stlOsciCP(x, param)
    return complex.(param["u"], param["w"] ).*x .- complex.(1, param["b"]).*(abs.(x).^2).*x .+ param["sig"] .+  DisReaCoupling(x, param);
end

function directCoupling(x, param)
    x = reshape(x, param["U"], param["B"], param["F"])
    v_new = similar(x)  # create a new vector with the same size and type as v
    
    @views v_new[:, :, 2:2:end] = x[:, :, 1:2:end]
    @views v_new[:, :, 1:2:end] = x[:, :, 2:2:end]
    return vec(v_new)
end

function DisReaCoupling(x, param)
    x = reshape(x, param["U"], param["B"], param["F"])
    v_new = similar(x)  # create a new vector with the same size and type as v

    @views v_new[:, :, 2:end-1] .= complex.(reshape(param["k"][1:end-2], 1, 1, :) .* real(x[:, :, 1:end-2]), reshape(param["d"][1:end-2], 1, 1, :) .* imag(x[:, :, 1:end-2]))
                            .+ complex.(reshape(param["k"][3:end], 1, 1, :) .* real(x[:, :, 3:end]), reshape(param["d"][3:end], 1, 1, :) .* imag(x[:, :, 3:end]))

    v_new[:, :, 1] = complex.(param["k"][2] * real(x[:, :, 2]), param["d"][2]* imag(x[:, :, 2]))      # keep the first value unchanged
    v_new[:, :, end] = complex.(param["k"][end-1]* real(x[:, :, end-1]), param["d"][end-1]* imag(x[:, :, end-1]))   # keep the last value unchanged

    return vec(v_new)
end