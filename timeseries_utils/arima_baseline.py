#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:41:49 2018

@author: hfwittmann
"""

# =============================================================================
# Load R packages
import rpy2.rinterface
 
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
 
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr
stats = importr('stats')
graphics = importr('graphics')
base = importr('base')
nnet = importr('nnet')
forecast = importr('forecast', '~/anaconda3/envs/py35_2/lib/R/library') 
# for some reason this cant load the forcast from the main R-install 
# fracdiff.so':
#  /usr/lib/libblas.so.3: undefined symbol: gotoblas

#library(fpp)

# =============================================================================


# =============================================================================
# load python packages
import numpy as np
import matplotlib.pyplot as pyplot
    

# =============================================================================

def fitMultipleUnivariateSeries(SERIES_train:np.array , SERIES_test:np.array):
    
    assert type(SERIES_train)==np.ndarray, 'SERIES_train is expected to be an numpy array'
    assert type(SERIES_test)==np.ndarray, 'SERIES_test is expected to be an numpy array'
       
    nOfPoints_train, nOfSeries_train = SERIES_train.shape
    nOfPoints_test, nOfSeries_test = SERIES_test.shape
    
    assert nOfSeries_train == nOfSeries_test, 'The number of series of SERIES_train and SERIES_test should match'
    
    FC = np.zeros([nOfPoints_test, nOfSeries_test ])
    FITS =list()
    
    for Series in range (nOfSeries_test):
 
        # fit training data
        y_train = SERIES_train[:,Series]        
        y_train_ts = stats.as_ts(y_train)
        train = y_train_ts # ?? stats.window        
        fit = forecast.auto_arima(train)
        
        # refit testing data
        y_test = SERIES_test[:,Series]        
        y_test_ts = stats.as_ts(y_test)
        
        refit = forecast.Arima(y_test_ts, model=fit)
        fc = stats.fitted(refit) # ?? stats.window
        
        FC[:,Series] = fc
        FITS.append(fit)
        
        
    return FC, FITS




