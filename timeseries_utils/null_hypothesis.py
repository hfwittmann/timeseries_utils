#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:24:21 2018

@author: hfwittmann
"""

import numpy as np
from sklearn.metrics import mean_squared_error


def calculatePredictionAccuracy(actual:np.array, predicted:np.array):
    
    # start: function parameter assertions
    assert type(actual)==np.ndarray, 'actual is expected to be an numpy array'
    assert type(predicted)==np.ndarray, 'predicted is expected to be an numpy array'
    # end: function parameter assertions
    
    assert actual.shape == predicted.shape, "Shapes of actual and predtited should match!"

    nofPointsOutofSample, nOfSeries = predicted.shape
    
    # Let's use 0 as the null hypothesis
    nullHypothesis = np.zeros(shape=predicted.shape)
    deviation_nullHypothesis = mean_squared_error(actual, nullHypothesis,multioutput='raw_values')
    
    
    deviation = mean_squared_error(actual, predicted, multioutput='raw_values')
    
    deviationRatio =  deviation/deviation_nullHypothesis
    # in this case deviationRatio = 1. arima finds nothing better than the null hypothsis
    performanceVSnullhypothesis = 1 - deviationRatio
    
    performanceVSnullhypothesis = np.array(performanceVSnullhypothesis * 100, dtype=int)
    
    return performanceVSnullhypothesis