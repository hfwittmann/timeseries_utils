#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:29:19 2018

@author: hfwittmann
"""

import warnings
import numpy as np
from keras import models, layers, regularizers

from timeseries_utils.keras_model import model_save, model_load
from timeseries_utils.differences import diffinv_rolling
from timeseries_utils import Configuration, Data

def __defineModel(C, D):
    
    try: # to load from cache
        
        model = model_load(directory = 'cache', name = C.dictionary['modelnamePlus'] )

    except:
    
        warnings.warn(message='No cache found')
        # define custom
        model = models.Sequential()
        
        units = 16
        model.add(layers.LSTM(units=units,
                                recurrent_regularizer=regularizers.l1_l2(l1=2e-4, l2=2e-4),
                                dropout=0.01, 
                                recurrent_dropout=0.01, 
                                return_sequences=False, 
                                input_shape=(C.dictionary['p_order'],  
                                          D.numberOfSeries()), 
                                          stateful=False)
                                )
        #        model.add(layers.LSTM(units=units,
        #                                recurrent_regularizer=regularizers.l1_l2(l1=2e-4, l2=2e-4),
        #                                dropout=0.01,
        #                                recurrent_dropout=0.01,
        #                                return_sequences=False))
        
        
        model.add(layers.Dense( D.numberOfSeries(), activation='linear'))
        model.compile(loss='mse', optimizer='rmsprop')  
        
        return model

    finally:
        
        return model
    
    
def __fitModel(model, C: Configuration, VARIABLES_train:np.array, SERIES_train:np.array, VARIABLES_test:np.array, SERIES_test:np.array):
    
    for epoch in range(C.dictionary['epochs']):
        model.fit(VARIABLES_train, SERIES_train, validation_data=(VARIABLES_test, SERIES_test), epochs=1, verbose=C.dictionary['verbose'])  
        model_save(model, directory = 'cache', name = C.dictionary['modelnamePlus'] )
        
        if C.dictionary['verbose'] >0:
            print('epoch: ', epoch)
        
        
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
    
    C.dictionary['modelnamePlus'] = C.dictionary['modelname'] + '_lstm_define_fit_predict.h5'
    
    # differencing on the way in ....
    differencingOrder = C.dictionary['differencingOrder']
    VARIABLES_train_diff = np.diff(VARIABLES_train,n=differencingOrder, axis=0)
    VARIABLES_test_diff = np.diff(VARIABLES_test,n=differencingOrder, axis=0)
    SERIES_train_diff = np.diff(SERIES_train,n=differencingOrder, axis=0)
    SERIES_test_diff = np.diff(SERIES_test,n=differencingOrder, axis=0)
    
    model = __defineModel(C=C, D=D)
    __fitModel(model=model,
               C=C,
               VARIABLES_train=VARIABLES_train_diff, 
               SERIES_train=SERIES_train_diff, 
               VARIABLES_test=VARIABLES_test_diff,
               SERIES_test=SERIES_test_diff)
    
    PREDICTION_diff = __predictWithModel(model=model, VARIABLES_test=VARIABLES_test_diff)
    
               

     # inverse of differencing
    PREDICTION = \
                diffinv_rolling(prediction_diff = PREDICTION_diff, differences=differencingOrder, axis=0, originalSeries = SERIES_test)
    
    
    return PREDICTION, model
    