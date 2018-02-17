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


def diffinv(D_series, differences=1, axis=0, originalSeries=None):
    """
    differences	
    an integer representing the order of the difference.
    
    xi	
    a numeric vector, matrix, or time series containing the initial values for the integrals. If missing, zeros are used.
    """
    if originalSeries is not None:
        xi = _get_xis(originalSeries, order=differences)
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

def diffinv_rolling(prediction_diff:np.array, differences:int=1, axis:int=0, originalSeries=None):
    
    
    # start: function parameter assertions
    #    assert isinstance(C,Configuration), 'C should be an instance of Configuration'
    #    assert isinstance(D,Data), 'D should be an instance of Data'
    
    assert type(prediction_diff)==np.ndarray, 'VARIABLES_train is expected to be an numpy array'
    assert (originalSeries is not None), 'If you really want originalSeries to be None, then use diffinv'
    
    # end: function parameter assertions
    
    
    nOfPoints_diff, nOfSeries_diff = prediction_diff.shape
    nOfPoints, nOfSeries = originalSeries.shape
    
    assert nOfPoints_diff + differences == nOfPoints, \
        "The shapes of D_series and originalSeries should be appriximately the same, expect for the differences"
    
    rolling_prediction = np.zeros(shape=[nOfPoints, nOfSeries]) + np.NaN # intialise
    # rolling prediction
    
    rolling_prediction[:differences] = originalSeries[:differences]
    # the valueas at the beginning of the time series
    
    
    realised_diffs = np.diff(originalSeries, n=differences, axis=axis)
    
    for p in range(nOfPoints_diff):
            
        past_diffs = realised_diffs[:p] # already realised
        future_diffs = prediction_diff[p:] # yet to come
        
        snapshot_diffs = np.concatenate((past_diffs, future_diffs),axis = 0)
        
        snapshot = diffinv(D_series=snapshot_diffs, differences=differences, originalSeries=originalSeries)
        
        # 
        now = p+differences # borderline between past and future
        rolling_prediction[now] = snapshot[now]


    return rolling_prediction

def diffinv_multi(multi_DSeries, differences=1, axis=0, originalSeries=None):
    """
    D_series is expected to be a multiseries : timesteps * numberofSeries
    
    """
    nOfSeries = multi_DSeries.shape[1]

    assert (originalSeries is None) or (originalSeries.shape[1] == nOfSeries), \
        "if originalSeries is specified at all then it should be a matrix and its number of columns should equal the number of Series"
        
    
    
    multi_list = list()
    
    for S in range(nOfSeries):
        series_reconstructed = diffinv(multi_DSeries[:,S], differences=differences, axis=axis, originalSeries = originalSeries[:,S])
        multi_list.append(series_reconstructed)
        
    multiseries_reconstructed = np.vstack(multi_list).T
    
    return multiseries_reconstructed
    
    
    
    




