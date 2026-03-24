import numpy as np
import subprocess
import os
import julia


def slo_transformation(
    sig,
    fs,
    *,
    all_u=np.array([1.0]),
    all_b=np.array([2.65]),
    fc=np.linspace(50, 800, int(5e1)),
    noise_u=0,
    noise_std=np.array([np.sqrt(0.1)]),
    coupling=False,
    kk = np.zeros(int(5e1)),
    dd = np.zeros(int(5e1)),
    julia_path = "/home/handel-tug/julia/bin/julia",
    script_sde="SLOT.jl",
    script_coupled="STLO_ODE_coupled.jl",
):
    """
    Wrapper around STLO simulation.
    
    Parameters
    ----------
    sig : np.array
        Input signal, shape (A, N)
    fs : float
        Sampling frequency
    
    Optional Parameters
    -------------------
    all_u, all_b : arrays
        Model parameters
    fc : array
        Characteristic frequency vector
    noise_u : float
    noise_std : array
    coupling : bool
        If True → use coupled Julia script
    julia_path : str
        path to julia binary
    script_sde, script_coupled : str
        Julia script paths
        
    Returns
    -------
    t : np.array
        Time vector
    eta : np.array
        Model output
    """

    # Derived parameters
    dt = np.array(1 / fs)
    t_start = np.array(0)
    t_end = dt * (sig.shape[1] - 1)

    E = 1
    F = len(fc)
    U = len(all_u)
    B = len(all_b)
    V = len(noise_std)
    A = sig.shape[0]

    if len(fc) != len(kk): 
        kk = np.zeros(F)
        dd = np.zeros(F)


    jl = julia.Julia(runtime=julia_path, compiled_modules=False)
    jl.include(script_sde)
   
    SLOT = jl.eval("SDE.SLOT")
  
    # ------------------------
    # Run Julia script
    # ------------------------
    t, eta_j = SLOT(sig.astype(np.complex128), fs, all_u, all_b, fc, noise_u, noise_std, kk, dd, E=E)

    
    t = np.array(t)
    eta = np.array(eta_j)

    if coupling:
        eta = eta.reshape((E, U, B, F, A, -1))
    else:
        eta = eta.reshape((E, U, B, V, F, A, -1))

    return t, eta
