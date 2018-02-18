#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 12:24:21 2018

@author: hfwittmann
"""

import numpy as np
from sklearn.metrics import mean_squared_error


def calculateForecastSkillScore(actual:np.array, predicted:np.array, movingAverage=None):
    
    # start: function parameter assertions
    assert type(actual)==np.ndarray, 'actual is expected to be an numpy array'
    assert type(predicted)==np.ndarray, 'predicted is expected to be an numpy array'
    # end: function parameter assertions
    
    assert actual.shape == predicted.shape, "Shapes of actual and predtited should match!"

    nofPointsOutofSample, nOfSeries = predicted.shape
    
    # Let's use 0 as the null hypothesis
    referenceHypothesis = np.zeros(shape=predicted.shape)
    if movingAverage:
        referenceHypothesis = calcMovingAverage(actual, movingAverage)
    
    deviation_nullHypothesis = mean_squared_error(actual, referenceHypothesis, multioutput='raw_values')
    
    
    deviation = mean_squared_error(actual, predicted, multioutput='raw_values')
    
    deviationRatio =  deviation/deviation_nullHypothesis
    # in this case deviationRatio = 1. arima finds nothing better than the null hypothsis
    performanceVSnullhypothesis = 1 - deviationRatio
    
    performanceVSnullhypothesis = np.array(performanceVSnullhypothesis * 100, dtype=int)
    
    return performanceVSnullhypothesis



def calcMovingAverage(actual: np.array, movingAverage:int=1):
    
    # start: function parameter assertions
    assert type(actual)==np.ndarray, 'actual is expected to be an numpy array'
    assert type(movingAverage)==int, 'movingAverage is expected to be an integer'
    # end: function parameter assertions
    
    assert movingAverage>0, 'movingAverage must at least be one'
    
    
    nOfPoints, nOfSeries = actual.shape
    
    reference = np.zeros(shape = [nOfPoints, nOfSeries]) # start with zeros where no data is avaiable + np.NaN
    
    for s in range(nOfSeries):
        
        reference[movingAverage:,s] = np.convolve(actual[:-1,s], np.ones((movingAverage,))/movingAverage, mode='valid')
    
    
    return reference

