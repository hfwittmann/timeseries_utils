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


def fitMultipleUnivariateSeries(MUS, percentageTraining = 0.5):
    
    
    nOfPoints = MUS.shape[0]
    nOfSeries = MUS.shape[1]
    
    trainingLength = int(nOfPoints*percentageTraining)

    FC = np.zeros([nOfPoints - trainingLength, nOfSeries])
    FITS = list()
    

    for Series in range(nOfSeries):
        
        y = MUS[:,Series]
        
        y_ts = stats.as_ts(y)
        train = stats.window (y_ts, trainingLength)
        fit = forecast.auto_arima(train)
        refit = forecast.Arima(y_ts, model=fit)
        fc = stats.window(stats.fitted(refit), start = trainingLength+1)
    
        FC[:,Series] =  fc
        FITS.append(fit)
    
    return FC, FITS



#from artificial_data import artificial_data
#
#nOfPoints = 1000
#nOfSeries = 15
#percentageTraining = 0.5
#trainingLength = nOfPoints*percentageTraining
#
#x, MUS = artificial_data(nOfPoints=nOfPoints, nOfSeries=nOfSeries, f_rauschFactor=0.9)
#
#FC = fitMultipleUnivariateSeries(MUS, percentageTraining)
#
## plot each column
#pyplot.figure()
#
#totalNumberOfPlots = 3
#
#plotnumber = 1
#pyplot.subplot(totalNumberOfPlots, 1, plotnumber)
#pyplot.plot(x, MUS)
#pyplot.plot(x[trainingLength:], FC, color='black')
#pyplot.ylabel('Signal and prediction')
#
#
#plotnumber = 2
#pyplot.subplot(totalNumberOfPlots, 1, plotnumber)
#pyplot.plot(x, MUS)
#pyplot.ylabel('Signal')
#
#plotnumber = 3
#pyplot.subplot(totalNumberOfPlots, 1, plotnumber)
#pyplot.plot(x, MUS, color='white')
#pyplot.plot(x[trainingLength:], FC, color='black')
#pyplot.ylabel('Prediction')
#
#pyplot.xlabel('time')


