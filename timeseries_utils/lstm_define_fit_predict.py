#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:29:19 2018

@author: hfwittmann
"""


from keras import models, layers, regularizers

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
    model.fit(VARIABLES_train, SERIES_train, epochs=100, verbose=0)    
    return None
    
def __predictWithModel(model, VARIABLES_test):    
    return model.predict(VARIABLES_test)
    

def defineFitPredict(C, D, VARIABLES_train, SERIES_train, VARIABLES_test):
    
    model = __defineModel(C=C, D=D)
    __fitModel(model=model, VARIABLES_train=VARIABLES_train, SERIES_train=SERIES_train)
    
    PREDICTION = __predictWithModel(model=model, VARIABLES_test=VARIABLES_test)
    
    return PREDICTION, model
    