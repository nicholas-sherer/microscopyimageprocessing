# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 15:47:18 2016

@author: kuhlmanlab
"""

import numpy as np
from functools import partial


def distance(p, points):
    """
    The distance between the point p and a collection of other points pairwise.
    """
    return np.sum((points-p)**2, 1)**.5


def weightedAverage(weights, values):
    """
    the weights must be positive
    """
    normed_weights = weights / np.sum(weights)
    return np.sum(normed_weights*values)


def distanceWeightedAverage(p, points, values, dist_func):
    """
    Computes a weighted average at a point p by averaging values evaluated at
    points weighted by the distance of those points from p.
    """
    d = distance(p, points)
    weights = dist_func(d)
    return weightedAverage(weights, values)


def expWeight(x, r):
    """
    An exponential function with coefficient one and exponent -1/r
    """
    return np.exp(-x/r)


def gaussianWeight(x, sigma):
    return np.exp(-x**2/sigma**2)


def createDistanceFunction(points, values, dist_func):
    """
    Returns a particular distance weighted average function for a set of points
    and a distance function, so you can evaluate it for various p.
    """
    return partial(distanceWeightedAverage, points=points, values=values,
                   dist_func=dist_func)


def createExpDistFunc(points, values, r):
    """
    Returns a distance weighted average function with the choice of an
    exponential distance function.
    """
    return createDistanceFunction(points, values, partial(expWeight, r=r))


def createGaussianDistFunc(points, values, sigma):
    """
    Returns a distance weighted average function with the choice of a gauusian
    distance function.
    """
    return createDistanceFunction(points, values,
                                  partial(gaussianWeight, sigma=sigma))


def evaluateDistsBox(xmax, ymax, dist_func):
    answer = np.zeros((xmax, ymax))
    for i in range(xmax):
        for j in range(ymax):
            answer[i, j] = dist_func(np.array([i, j]))
    return answer
