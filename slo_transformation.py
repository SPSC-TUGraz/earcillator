import numpy as np
import subprocess
import os
import julia


def slo_transformation(
    sig,
    fs,
    *,
    E=1,
    all_u=np.zeros(int(5e1)),
    all_b=np.ones(int(5e1)),
    fc=np.linspace(50, 800, int(5e1)),
    noise_u=0,
    noise_std=np.sqrt(0.1),
    kk = 5*np.ones(int(5e1)),
    dd = 5*np.ones(int(5e1)),
    julia_path = "/home/example/julia/bin/julia",
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
    noise_std : float

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

    F = len(fc)
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
    eta = np.array(eta_j, order="F")
    eta = eta.reshape((E, F, A, -1), order="F")

    return t, eta
