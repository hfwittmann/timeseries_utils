#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:29:19 2018

@author: hfwittmann
"""

import numpy as np
from keras import models, layers, regularizers

from timeseries_utils.differences import diffinv_rolling
from timeseries_utils import Configuration, Data

def __defineModel(C, D):
    
    # define custom
    model = models.Sequential()
    
    units = 64
    model.add(layers.LSTM(units=units,recurrent_regularizer=regularizers.l1_l2(l1=2e-4, l2=2e-4), dropout=0.01, recurrent_dropout=0.01, return_sequences=True, input_shape=(C.dictionary['p_order'],  D.numberOfSeries()), stateful=False))
    model.add(layers.LSTM(units=units,recurrent_regularizer=regularizers.l1_l2(l1=2e-4, l2=2e-4), dropout=0.01, recurrent_dropout=0.01, return_sequences=False))
    
    
    model.add(layers.Dense( D.numberOfSeries(), activation='linear'))
    model.compile(loss='mse', optimizer='rmsprop')  

    return model
    
    
def __fitModel(model, VARIABLES_train, SERIES_train):    
    model.fit(VARIABLES_train, SERIES_train, epochs=200, verbose=0)    
    return None
    
def __predictWithModel(model, VARIABLES_test):    
    return model.predict(VARIABLES_test)
    

def defineFitPredict(C: Configuration, D:Data, VARIABLES_train:np.array, SERIES_train:np.array, VARIABLES_test:np.array, SERIES_test:np.array):
    
    # start: function parameter assertions
    assert isinstance(C,Configuration), 'C should be an instance of Configuration'
    assert isinstance(D,Data), 'D should be an instance of Data'
    
    assert type(VARIABLES_train)==np.ndarray, 'VARIABLES_train is expected to be an numpy array'
    assert type(VARIABLES_test)==np.ndarray, 'VARIABLES_test is expected to be an numpy array'
    assert type(SERIES_train)==np.ndarray, 'SERIES_train is expected to be an numpy array'
    assert type(SERIES_test)==np.ndarray, 'SERIES_test is expected to be an numpy array'
    # end: function parameter assertions
    
    # SERIES_train only serves for the re-contruction of SERIES from the differences 
    # for a rolling forecast
    
    # differencing on the way in ....
    differencingOrder = C.dictionary['differencingOrder']
    VARIABLES_train_diff = np.diff(VARIABLES_train,n=differencingOrder, axis=0)
    VARIABLES_test_diff = np.diff(VARIABLES_test,n=differencingOrder, axis=0)
    SERIES_train_diff = np.diff(SERIES_train,n=differencingOrder, axis=0)
    
    
    model = __defineModel(C=C, D=D)
    __fitModel(model=model, VARIABLES_train=VARIABLES_train_diff, SERIES_train=SERIES_train_diff)
    
    PREDICTION_diff = __predictWithModel(model=model, VARIABLES_test=VARIABLES_test_diff)
    
               

     # inverse of differencing
    PREDICTION = \
                diffinv_rolling(prediction_diff = PREDICTION_diff, differences=differencingOrder, axis=0, originalSeries = SERIES_test)
    
    
    return PREDICTION, model
    