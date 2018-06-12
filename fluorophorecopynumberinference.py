#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 15:58:12 2018

@author: kuhlmanlab

Method based upon the work by Nayak and Rutenberg (2011)
"""

from collections import OrderedDict
import numpy as np
import scipy.optimize as spopt


def exp_plus_constant_ceilinged(x, A, tau, C, ceiling):
    return np.minimum(A*np.exp(-x/tau) + C, ceiling)


def fit_photobleach_rate(arg1, arg2=None):
    if arg2 is None:
        y = arg1
        x = np.arange(len(y))
    else:
        y = arg2
        x = arg1
    C_initial_guess = np.mean(y[int(3*len(y)/4):len(y)])
    A_initial_guess = np.mean(y[0:2])-C_initial_guess
    tau_approx_index = np.argmin(np.abs(y - (A_initial_guess*np.exp(-1) +
                                             C_initial_guess)))
    tau_initial_guess = x[tau_approx_index]
    ceiling_initial_guess = 65000.0
    try:
        popt, pcov = spopt.curve_fit(exp_plus_constant_ceilinged, x, y,
                                     p0=(A_initial_guess, tau_initial_guess,
                                         C_initial_guess,
                                         ceiling_initial_guess))
    except RuntimeError:
        popt = np.array([np.nan]*4)
        pcov = np.ones((4, 4))*np.nan
    params = OrderedDict()
    stds = OrderedDict()
    params['A'] = popt[0]
    stds['A'] = pcov[0, 0]**.5
    params['tau'] = popt[1]
    stds['tau'] = pcov[1, 1]**.5
    params['C'] = popt[2]
    stds['C'] = pcov[2, 2]**.5
    params['ceiling'] = popt[3]
    stds['ceiling'] = pcov[3, 3]**.5
    return params, stds


def single_cell_inference(I, time):
    params, stds = fit_photobleach_rate(time, I)
    p = np.exp(-time/params['tau'])
    delta_p = -np.diff(p)
    I_fit = exp_plus_constant_ceilinged(time, *params.values())
    I_0 = I_fit[0]
    I_var = (I - I_fit)**2
    return 6 * np.sum(I_var[:-1]/I_0*delta_p)
