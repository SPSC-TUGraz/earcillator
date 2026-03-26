module SDE

include("STLOscillator_coupled.jl")

using DifferentialEquations
using Random, Distributions
using Interpolations
using IterTools

export SLOT

"""
SLOT(sig, fs, u, b, fc, noise_u, noise_v)

sig :: Matrix{ComplexF64}   (A × N)
fs  :: Float64
noise_u :: Float64
noise_v :: Float64

Other parameters are vectors (u, b, fc).

Returns:
    t   :: Vector{Float64}
    eta :: Array{ComplexF64} (E × F*A × T)
"""
function SLOT(
    sig::AbstractArray{<:Complex},
    fs::Real,
    u::AbstractVector,
    b::AbstractVector,
    fc::AbstractVector,
    noise_u::Real,
    noise_v::Real,
    k :: AbstractVector,
    d :: AbstractVector;
    E::Int=1
)

    # -----------------------------------------
    # Derived simulation parameters
    # -----------------------------------------
    dt = 1/fs
    t_start = 0.0
    t_end = dt * (size(sig, 2) - 1)
    t_span = (t_start, t_end)
    t_data = reshape(t_span[1]:dt:t_span[2], size(sig, 2))
    A = size(sig, 1)
    F = length(fc)

    all_intp = [LinearInterpolation(t_data, sig[idx, :], extrapolation_bc=Flat()) for idx in 1:size(sig, 1)]


    wc = 2*pi.*fc
    w = wc .+ b.*u 
    w[wc .< 0] = abs.(wc[wc .< 0])

    params = Dict()

    params["w"] = vec(repeat(reshape(w, :, 1), 1, A))
    params["b"] = vec(repeat(reshape(b, :, 1), 1, A))
    params["u"] = vec(repeat(reshape(u, :, 1), 1, A))
    params["sig"] = 0
    params["F"] = F
    params["A"] = A
    params["d"] = d 
    params["k"] = k

    osciFunc = SLOsciNNCP

    # setup function and starting values for stochastic differential problem

    u0_base = vec(repeat(reshape(u, :, 1), 1, A))
    eta0 = sqrt.(abs.(u0_base)) .+ 1e-12 .* randn(length(u0_base)) .+ 0im
    init_val = zeros(ComplexF64, E, length(eta0))


    function driftInit!(dx, x, p, t) 
        params["sig"] = 0
        dx .= osciFunc(x, params)
        

    end

    function diffusionInit!(dx, x, p, t)

        dx .= p[2]
    end


    p = [noise_u, noise_v]
    sim_time = (300+E)*dt
    sde_init = SDEProblem(driftInit!, diffusionInit!, eta0, (0, sim_time), p)

    # get init values for esemble of sde problem
    sol = solve(sde_init, dt=8e-15, saveat= 0:dt:sim_time)

    offset = floor(Int, sim_time/dt - E)


    for i in (offset+1):offset+E
        init_val[i-offset, :] .= vec(sol.u[i])
    end


    function drift!(dx, x, p, t) 
        sig_t = [input_intp(t) for input_intp in all_intp]
        params["sig"] = vec(repeat(reshape(sig_t, 1, :), F, 1))
        dx .= osciFunc(x, params)

    end

    function diffusion!(dx, x, p, t)

        dx .= p[2]

    end


    sde = SDEProblem(drift!, diffusion!, init_val[1, :], t_span, p)


    # setup esemble sde problem
    # prob_func to vary starting point for every trajectory
    prob_func = function (prob, i, repeat)
        eta0_rand = init_val[i, :] # perturb initial condition
        remake(prob, u0=eta0_rand)
    end
    
   

    # Wrap in ensemble problem
    ensemble_prob = EnsembleProblem(sde, prob_func=prob_func)

    # Solve ensemble with N trajectories
    sol = solve(ensemble_prob, trajectories = E, SRIW1(), EnsembleThreads(), dt=1e-14, saveat=t_data)

    num_steps = length(sol[1].t)

    data = zeros(ComplexF64, E, F*size(sig, 1), num_steps)

    # Fill array with trajectory values (assume scalar u)
    for i in 1:E
        
        data[i, :, :] .= sol.u[i]
         
    end

    return sol[1].t, data
end

end # module
