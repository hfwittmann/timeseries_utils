#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:29:49 2018

@author: hfwittmann
"""

from keras import models, layers, regularizers

def __defineModel(C, D):
    
    # define custom
    model = models.Sequential()
    
    # model.add(layers.Dense(units=64,input_shape=(C.dictionary['p_order'], D.numberOfSeries(), )))
    # # swap dimensions
    # model.add(layers.Permute((2,1),input_shape=(C.dictionary['p_order'], D.numberOfSeries(), )))
    # model.add(layers.Flatten())
    
    units = 64
    
    model.add(layers.Dense(units=units, kernel_regularizer=regularizers.l1_l2(l1=5e-4, l2=5e-4), input_shape=( C.dictionary['p_order'], D.numberOfSeries(), )))
    model.add(layers.Flatten())
    
    # model.add(layers.Dense(units=64,kernel_regularizer=regularizers.l1_l2(l1=5e-4, l2=5e-4)))
    model.add(layers.Dropout(rate=0.02))
    model.add(layers.Dense(units=D.numberOfSeries(), activation='linear'))
    # model.add(layers.Dense(units=1))
    # model.add(layers.Reshape((2,)))
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
    