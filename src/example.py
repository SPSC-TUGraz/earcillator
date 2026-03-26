#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 10:34:10 2025

@author: handel-tug
"""
import slo_transformation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def mag2db(x):
    return 20 * np.log10(np.maximum(x, 1e-300))  # avoid log(0)

def f2Dis_Greenwood(f):
    A_G = 20682
    B_G = 140.6
    alpha_G = 61.765
    return np.log10((f + B_G)/A_G)/(-alpha_G)

def dis2f_Greenwood(d):
    A_G = 20682
    B_G = 140.6
    alpha_G = 61.765
    return A_G*pow(10, (d*-alpha_G)) - B_G

A = 9
F = 60

d = 5* np.ones(F)
k = 5* np.ones(F)
u = np.zeros(F)
b = np.ones(F)
fc = dis2f_Greenwood(np.linspace(f2Dis_Greenwood(200), f2Dis_Greenwood(16e3), F))

fs = 5e5  # sampling rate
duration = 0.5  # seconds
f0 = 1000       # cosine frequency

SPL = np.linspace(30, 110, A)
A0 = pow(10, (SPL/10)) * 2e-5
noise_std = 0

t_vec = np.arange(0, duration, 1/fs)
sig = np.reshape(A0, [A, 1])*np.cos(2 * np.pi * f0 * t_vec)


"""!!!! Be aware that you have to define your julia path !!!!"""
t, eta = slo_transformation.slo_transformation(sig, fs, all_u = u, all_b = b, fc = fc, kk = k, dd = d, noise_std = noise_std, julia_path = "/home/handel-tug/julia/bin/julia")

"""================== Displacement Compression ==================="""
plt.figure()

colors = plt.cm.tab20(np.linspace(0, 1, A))  # similar to distinguishable_colors
legend_entries = []

for n in range(A):
    # MATLAB:
    # rms(..., 2) over time dimension
    data = np.real(eta[0, :, n, int(eta.shape[3]/2):])  # match indexing
    
    rms_vals = np.sqrt(np.mean(data**2, axis=1))  # RMS over time
    
    y = mag2db(rms_vals * 1e-9)

    plt.plot(fc, y, color=colors[n])
    legend_entries.append(f"{SPL[n]-20} dB")

# formatting
legend_entries.append("ref")

plt.xscale("log")
plt.xticks(2**np.arange(14), labels=[str(v) for v in 2**np.arange(14)])

plt.ylim([-350, -100])
plt.gca().invert_xaxis()

plt.grid(True)
plt.xlabel("Characteristic Frequency [Hz]")
plt.ylabel("Displacement [nm]")

plt.legend(legend_entries)


"""================== Traveling Wave ==================="""
t_sub = t[0:len(t)//200:100]
fc_sub = fc
aa = 2

y_vals = f2Dis_Greenwood(fc_sub) * 100
Z = np.real(eta[0, :, aa, 0:len(t)//200:100])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

norm = plt.Normalize(np.min(Z), np.max(Z))
cmap = plt.cm.viridis

for i in range(len(y_vals)):
    x = np.asarray(t_sub).ravel()
    y = np.full(x.shape, y_vals[i])
    z = np.asarray(Z[i, :]).ravel()

    points = np.column_stack((x, y, z)).reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = Line3DCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(z[:-1])
    lc.set_linewidth(1.5)
    ax.add_collection3d(lc)

ax.set_xlim(np.min(t_sub), np.max(t_sub))
ax.set_ylim(np.min(y_vals), np.max(y_vals))
ax.set_zlim(np.min(Z), np.max(Z))

ax.set_xlabel("Time [s]")
ax.set_ylabel("Distance from Base [cm]")
ax.set_zlabel("Displacement [nm]")
ax.set_title("Traveling Wave")

ax.invert_xaxis()
ax.view_init(elev=25, azim=55)

mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
mappable.set_array(Z)
cb = fig.colorbar(mappable, ax=ax)
cb.set_label("Displacement [nm]")
# match MATLAB orientation

ax.view_init(elev=25, azim=55)


plt.tight_layout()
plt.show()
