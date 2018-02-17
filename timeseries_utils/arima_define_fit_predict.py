#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:29:59 2018

@author: hfwittmann
"""

import numpy as np
from timeseries_utils import fitMultipleUnivariateSeries

from timeseries_utils.differences import diffinv_rolling
from timeseries_utils import Configuration, Data

def defineFitPredict(C: Configuration, D:Data, VARIABLES_train:np.array, SERIES_train:np.array, VARIABLES_test:np.array, SERIES_test:np.array):
    
    # start: function parameter assertions
    assert isinstance(C,Configuration), 'C should be an instance of Configuration'
    assert isinstance(D,Data), 'D should be an instance of Data'
    
    assert type(VARIABLES_train)==np.ndarray, 'VARIABLES_train is expected to be an numpy array'
    assert type(VARIABLES_test)==np.ndarray, 'VARIABLES_test is expected to be an numpy array'
    assert type(SERIES_train)==np.ndarray, 'SERIES_train is expected to be an numpy array'
    assert type(SERIES_test)==np.ndarray, 'SERIES_test is expected to be an numpy array'
    # end: function parameter assertions
    
    
    differencingOrder = C.dictionary['differencingOrder']    
    
    # differencing on the way in
    SERIES_train_diff = np.diff(SERIES_train, n=differencingOrder, axis=0)
    SERIES_test_diff = np.diff(SERIES_test, n=differencingOrder, axis=0)
    
    PREDICTION_diff, MODELS = fitMultipleUnivariateSeries(SERIES_train=SERIES_train_diff,
                                                                      SERIES_test=SERIES_test_diff)

    # inverse of differencing on the way out
    PREDICTION = diffinv_rolling(prediction_diff = PREDICTION_diff, differences=differencingOrder, axis=0, originalSeries= SERIES_test)
    

    return PREDICTION, MODELS
