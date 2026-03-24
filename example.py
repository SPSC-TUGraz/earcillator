#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 10:34:10 2025

@author: handel-tug
"""
import slo_transformation
import numpy as np



fs = 44100  # sampling rate
duration = 1  # seconds
f0 = 1000       # cosine frequency

t_vec = np.arange(0, duration, 1/fs)
sig = 200*np.cos(2 * np.pi * f0 * t_vec)[None, :]

t, complex_output = slo_transformation.slo_transformation(sig, fs, kk=np.ones(int(5e1)), dd=np.ones(int(5e1)))