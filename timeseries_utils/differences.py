#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 19:45:58 2018

@author: hfwittmann
"""
import numpy as np

def _get_xis(series, order, axis=0):
    
    multiseries = len(series.shape) == 2
    
    if multiseries:
        xi_shape = [order, series.shape[1]]
    
    if not multiseries:
        xi_shape =[order]
        
    xi = np.zeros(xi_shape) + np.NaN
    
    for i in range(order):
        xi[i] = np.diff(series,n=i,axis=axis)[0]
    
    return xi


def diffinv(D_series, differences=1, axis=0, getxiFrom=None):
    """
    differences	
    an integer representing the order of the difference.
    
    xi	
    a numeric vector, matrix, or time series containing the initial values for the integrals. If missing, zeros are used.
    """
    if getxiFrom is not None:
        xi = _get_xis(getxiFrom, order=differences)
    else:
        xi = None

    
    assert (xi is None) or (isinstance(xi, np.ndarray) and (len(xi) == differences)), \
        "if x1 is specified at all then it should be a list and its length should equal the differences"
    

    # If missing, zeros are used.
    if  xi is None:
        xi = [0 for D in range(differences)]
        

    # go backwards to zero
    for D in range(differences-1,-1,-1):                   
        
        D_series = np.insert(D_series, 0, xi[D], axis=axis)
        D_series = np.cumsum(D_series, axis=axis)

    
    return D_series


def diffinv_multi(multi_DSeries, differences=1, axis=0, getxiFrom=None):
    """
    D_series is expected to be a multiseries : timesteps * numberofSeries
    
    """
    nOfSeries = multi_DSeries.shape[1]

    assert (getxiFrom is None) or (getxiFrom.shape[1] == nOfSeries), \
        "if getxiFrom is specified at all then it should be a matrix and its number of columns should equal the number of Series"
        
    
    
    multi_list = list()
    
    for S in range(nOfSeries):
        series_reconstructed = diffinv(multi_DSeries[:,S], differences=differences, axis=axis, getxiFrom = getxiFrom[:,S])
        multi_list.append(series_reconstructed)
        
    multiseries_reconstructed = np.vstack(multi_list).T
    
    return multiseries_reconstructed
    
    
    
    




