#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:41:44 2018

@author: hfwittmann
"""

import infix
# import numpy as np
import pandas as pd

from keras import models, layers, regularizers


@infix.or_infix
def implies(x, y):
    #https://stackoverflow.com/questions/16405892/is-there-an-implication-logical-operator-in-python
    return (not x) or y

    
    
    return None

def prepareModel(units=64, nOfSeries=11, p_order=3):

    model = models.Sequential()
    
    model.add(layers.LSTM(units=units,recurrent_regularizer = regularizers.l1_l2(l1=2e-4, l2=2e-4), dropout=0.01, recurrent_dropout=0.01, return_sequences=True, input_shape=(p_order, nOfSeries)))
    model.add(layers.LSTM(units=units,recurrent_regularizer = regularizers.l1_l2(l1=2e-4, l2=2e-4), dropout=0.01, recurrent_dropout=0.01, return_sequences=True))
    model.add(layers.LSTM(units=units,recurrent_regularizer = regularizers.l1_l2(l1=2e-4, l2=2e-4), dropout=0.01, recurrent_dropout=0.01, return_sequences=False))

    model.add(layers.Dense(nOfSeries, activation='linear', kernel_regularizer= regularizers.l1_l2(l1=2e-4, l2=2e-4)  ))
    model.compile(loss='mse', optimizer='adam')  
    
    return model


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):

        #    https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg



def prepareData(MVS, inSampleRatio, p_order=None, steps_prediction=1):
    
    '''
    MVS stands for multivariate time series
    inSampleRatio is the proportion of data used for training
    p_order is the the p in the ARIMA(p,d,q) 
        where parameters p, d, and q are 
        non-negative integers, p is the order (number of time lags) of the autoregressive model, d is the degree of differencing (the number of times the data have had past values subtracted), and q is the order of the moving-average model.
    
    steps_prediction is how many steps into the future should be predicted, typically this is 1
    '''
    
    assert 0 <= inSampleRatio <= 1, "inSampleRatio should be a number between 0 and 1\
                                indicating the percentage of training data wrt the total"
    
    
    assert MVS.ndim == 2, "Y should have two axes nofPoints (x) nOfSeries"
    
    
    nOfPoints, nofSeries = MVS.shape
    
    assert isinstance(p_order, int) , "p_order must be an integer"
    
    assert isinstance(steps_prediction, int) , "Steps_prediction must be an integer"
    
    
    # assert model is keras model    
    # assert type(model) == type
    
    
    splittingBoundary = int(nOfPoints * inSampleRatio)
    
    inSample_pre = MVS[:splittingBoundary]
    outOfSample_pre = MVS[splittingBoundary-p_order:]
    
    inSample_pre2 = series_to_supervised(inSample_pre,n_in=p_order, n_out=steps_prediction)
    outOfSample_pre2 = series_to_supervised(outOfSample_pre,n_in=p_order, n_out=steps_prediction)
    
    inSample_pre3 = inSample_pre2.values.reshape([-1, p_order + steps_prediction, nofSeries])
    outOfSample_pre3 = outOfSample_pre2.values.reshape([-1, p_order + steps_prediction, nofSeries])
    
    inSample_X = inSample_pre3[:,:p_order,:]
    inSample_Y = inSample_pre3[:,p_order,:] # p_order because we are interested only in one step ahead
    
    outOfSample_X = outOfSample_pre3[:,:p_order,:]
    outOfSample_Y = outOfSample_pre3[:,p_order,:] # p_order because we are interested only in one step ahead
    
    
    inSample = {'X': inSample_X,
                   'Y': inSample_Y}   
    outOfSample = { 'X': outOfSample_X,
                        'Y': outOfSample_Y
                        }
        
        
    return { 
            'inSample': inSample,
            'outOfSample': outOfSample
            }
