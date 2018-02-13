#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:29:59 2018

@author: hfwittmann
"""

from timeseries_utils import fitMultipleUnivariateSeries

def defineFitPredict(C, D, VARIABLES_train, SERIES_train, VARIABLES_test):
    

    Prediction, MODELS = fitMultipleUnivariateSeries(D.SERIES(),percentageTraining=C.dictionary['inSampleRatio'])
    
    return Prediction, MODELS
