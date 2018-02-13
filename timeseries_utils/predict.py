#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:08:00 2018

@author: hfwittmann
"""

import numpy as np

from timeseries_utils.prepare import prepareData
from timeseries_utils.differences import diffinv_multi


def fit_prepare(MVS, inSampleRatio = 0.7,\
        p_order=7, differencingOrder=1, differencingAxis = 0,\
        steps_prediction=1,
        scaler=None):
    
    if scaler is None:
    
        MVS_for_fit = MVS
                   
    
    if scaler is not None:
        
        MVS_for_fit = scaler.fit_transform(MVS)
        
    X, Y, test_X, test_Y = \
        _handle(MVS,inSampleRatio, p_order=p_order,steps_prediction=steps_prediction)  
    
    X_diff = np.diff(X, n=differencingOrder, axis=differencingAxis)
    Y_diff = np.diff(Y, n=differencingOrder, axis=differencingAxis)
    test_X_diff = np.diff(test_X, n=differencingOrder, axis=differencingAxis)
    test_Y_diff = np.diff(test_Y, n=differencingOrder, axis=differencingAxis)
    
    X_fit, Y_fit, test_X_fit, test_Y_fit = \
        _handle(MVS_for_fit,inSampleRatio, p_order=p_order,steps_prediction=steps_prediction)
    
    X_diff_fit = np.diff(X, n=differencingOrder, axis=differencingAxis)
    Y_diff_fit = np.diff(Y, n=differencingOrder, axis=differencingAxis)
    test_X_diff_fit = np.diff(test_X, n=differencingOrder, axis=differencingAxis)
    test_Y_diff_fit = np.diff(test_Y, n=differencingOrder, axis=differencingAxis)

    X_diff_fit = X_diff
    Y_diff_fit = Y_diff
    test_X_diff_fit = test_X_diff
    test_Y_diff_fit = test_Y_diff

    return  X, Y, test_X, test_Y, X_diff, Y_diff, test_X_diff, test_Y_diff, \
            X_fit, Y_fit, test_X_fit, test_Y_fit, X_diff_fit, Y_diff_fit, test_X_diff_fit, test_Y_diff_fit

def predict(X_fit, Y_fit , model, differencingOrder=1, differencingAxis = 0, scaler=None):
       
    nOfPoints_X, nOfLags_X, nOfSeries_X = X_fit.shape
    nOfPoints_Y, nOfSeries_Y = Y_fit.shape
    
    assert nOfPoints_X==nOfPoints_Y, "nOfPoints should match for X and Y"
    assert nOfSeries_X==nOfSeries_Y, "nOfSeries should match for X and Y"
    
    X_diff = np.diff(X_fit, n=differencingOrder, axis=differencingAxis)
    Y_diff = np.diff(Y_fit, n=differencingOrder, axis=differencingAxis)

    prediction_diff= model.predict(X_diff)
    
    
    Y_rolling_prediction = Y_fit + np.NaN # intialise
    # rolling prediction
    Y_rolling_prediction[:differencingOrder] = Y_fit[:differencingOrder]
    # the valueas at the biginning of the time series

    
    for p in range(nOfPoints_X-differencingOrder):
                
        past_diffs = Y_diff[:p] # already realised
        future_diffs = prediction_diff[p:] # yet to come
    
        snapshot_diffs = np.concatenate((past_diffs, future_diffs),axis = 0)
    
        snapshot = diffinv_multi(multi_DSeries=snapshot_diffs, differences=differencingOrder, getxiFrom=Y_fit)
        
        # 
        now = p+differencingOrder # borderline between past and future
        Y_rolling_prediction[now] = snapshot[now]
        
    
    
    
    if scaler:
        Y_rolling_prediction = scaler.inverse_transform(Y_rolling_prediction)
    
    return Y_rolling_prediction
    




def _handle(MVS, inSampleRatio = 0.7, p_order= 7, steps_prediction=1):
        
    preparedD = prepareData(MVS = MVS, inSampleRatio = inSampleRatio, p_order= p_order, steps_prediction=steps_prediction)
    inSample = preparedD['inSample']
    outOfSample = preparedD['outOfSample']   
    

    X = inSample['X']
    Y = inSample['Y']
    
    test_X = outOfSample['X']
    test_Y = outOfSample['Y']
        
    
    return X, Y, test_X, test_Y





