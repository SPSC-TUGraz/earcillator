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

Other parameters are vectors (u, b, fc, noise_v).

Returns:
    t   :: Vector{Float64}
    eta :: Array{ComplexF64} (E × U*B*V*F*A × T)
"""
function SLOT(
    sig::AbstractArray{<:Complex},
    fs::Real,
    u::AbstractVector,
    b::AbstractVector,
    fc::AbstractVector,
    noise_u::Real,
    noise_v::AbstractVector,
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
    A = size(sig, 1)

    U = length(u)
    B = length(b)
    F = length(fc)
    V = length(noise_v)

    # -----------------------------------------
    # Time vector & interpolation of input
    # -----------------------------------------
    t_span = (t_start, t_end)
t_data = reshape(t_span[1]:dt:t_span[2], size(sig, 2))

input_intpors = [LinearInterpolation(t_data, sig[idx, :], extrapolation_bc=Flat()) for idx in 1:size(sig, 1)]


param_combos = collect(product(u, b, noise_v, fc, input_intpors))
all_u = vec([p[1] for p in param_combos])
all_b = vec([p[2] for p in param_combos])
all_v = vec([p[3] for p in param_combos])
all_f = vec([p[4] for p in param_combos])
all_intp = vec([p[5] for p in param_combos])

wc = 2*pi.*all_f
w = wc .+ all_b.*all_u 
w[wc .< 0] = abs.(wc[wc .< 0])

params = Dict()

params["w"] = w
params["b"] = all_b
params["u"] = all_u
params["sig"] = 0
params["U"] = U
params["B"] = B
params["F"] = F
params["d"] = d 
params["k"] = k

# setup function and starting values for stochastic differential problem
eta0 = sqrt.(abs.(all_u)) .+ 1e-12 .* randn(size(all_u)) .+ 0im

function drift!(dx, x, p, t) 
    params["sig"] = [input_intp.(t) for input_intp in all_intp]
    dx[:] .= stlOsciCP(x, params)
end

function diffusion!(dx, x, p, t)
    dx[:] .= p[2] .* ones(size(eta0))
end


p = [noise_u, all_v]
sim_time = 2*E*dt
sde_init = SDEProblem(drift!, diffusion!, eta0, (0, sim_time), p)

# get init values for esemble of sde problem
sol = solve(sde_init, dt=8e-15, saveat= 0:dt:sim_time)

init_val = zeros(ComplexF64, E, U*B*V*F*size(sig, 1))

offset = floor(Int, sim_time/dt/2)
for i in (offset+1):offset+E
    init_val[i-offset, :] .= sol.u[i][:]
end

# -----------------------------------------
# Split simulation into 10s segments
# -----------------------------------------
seg_len = 10.0  # seconds
segments = ceil(Int, t_end / seg_len)

current_eta = init_val
all_data = ComplexF64[]
all_t = Float64[]

for seg in 1:segments
    seg_start = (seg - 1) * seg_len
    seg_end = min(seg * seg_len, t_end)
    t_span_seg = (seg_start, seg_end)
    t_save_seg = collect(seg_start:dt:seg_end)

    # Single SDE problem
    sde_seg = SDEProblem(drift!, diffusion!, current_eta, t_span_seg, p)

    # Ensemble problem to generate E trajectories
    prob_func = function (prob, i, repeat)
        remake(prob, u0=current_eta[i, :])
    end
    ensemble_prob = EnsembleProblem(sde_seg, prob_func=prob_func)

    sol_seg = solve(ensemble_prob, trajectories=E, SRIW1(), EnsembleThreads();
                    dt=1e-14, saveat=t_save_seg, verbose=false, progress=false)

    num_steps = length(sol_seg[1].t)
    traj_data = zeros(ComplexF64, E, length(current_eta), num_steps)

    for i in 1:E
        traj_data[i, :, :] = hcat(sol_seg[i].u...)
    end

    if seg == 1
        all_data = traj_data
        all_t = sol_seg[1].t
    else
        # concatenate along time axis, skipping duplicate point at boundary
        all_data = cat(all_data, traj_data[:, :, 2:end]; dims=3)
        all_t = vcat(all_t, sol_seg[1].t[2:end])
    end

    # Update initial condition for next segment
    current_eta = traj_data[:, :, end]
end

    return all_t, all_data
end

end # module
