#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:43:36 2018

@author: hfwittmann
"""

from unittest import TestCase
import numpy as np

from timeseries_utils import Configuration, Data
from timeseries_utils import calculateForecastSkillScore
from timeseries_utils import artificial_data
from timeseries_utils import defineFitPredict_ARIMA, defineFitPredict_DENSE#, defineFitPredict_LSTM



class Test_Predict(TestCase):


    # configuration
    C = Configuration()
    
    C.dictionary['nOfPoints'] = 600
    # below 300 things dont work so nicely, so we use a value considerably above that
    C.dictionary['inSampleRatio'] = 0.5
    
    C.dictionary['differencingOrder'] = 0
    
    C.dictionary['p_order'] = 7
    C.dictionary['steps_prediction'] = 1
    
    
    # create data
    # # set random seed for reporducibility
    np.random.seed(178)
    D = Data()
    nOfSeries = 7
    
    artificial_x, artificial_SERIES = artificial_data(nOfPoints = C.dictionary['nOfPoints'], 
                                       nOfSeries = nOfSeries, 
                                       f_rauschFactor = 0.5)
    
    
    
    for S in range(1, nOfSeries):
        
        signal_S = 'signal' + str(S)
        D.setFollower(signal_S, otherSeries=artificial_SERIES[:,S], otherSeriesName='innovator plus' + signal_S)
    
    
    SERIES_train = D.train(D.SERIES(), Configuration=C)
    SERIES_test = D.test(D.SERIES(), Configuration=C)
    
    VARIABLES_train = D.train(D.VARIABLES(), Configuration=C)
    VARIABLES_test = D.test(D.VARIABLES(), Configuration=C)
    
    
    def test_arima(self):
        
        defineFitPredict = defineFitPredict_ARIMA
        #define, fit model; predict with with model
        Prediction, Model = defineFitPredict(C=self.C,
                                             D=self.D, 
                                             VARIABLES_train=self.VARIABLES_train, 
                                             SERIES_train=self.SERIES_train, 
                                             VARIABLES_test=self.VARIABLES_test,
                                             SERIES_test=self.SERIES_test)
        
        # calculate ForecastSkillScore : 0% as good as NULL-Hypothesis, 100% is perfect prediction
        ForecastSkillScore = calculateForecastSkillScore(actual=self.SERIES_test, predicted=Prediction)
        
        self.assertTrue((ForecastSkillScore > 80).all(), 'ForecastSkillScore should be larger than 90%')


    def test_dense(self):
        
        defineFitPredict = defineFitPredict_DENSE
        #define, fit model; predict with with model
        Prediction, Model = defineFitPredict(C=self.C,
                                             D=self.D, 
                                             VARIABLES_train=self.VARIABLES_train, 
                                             SERIES_train=self.SERIES_train, 
                                             VARIABLES_test=self.VARIABLES_test,
                                             SERIES_test=self.SERIES_test)
        
        # calculate ForecastSkillScore : 0% as good as NULL-Hypothesis, 100% is perfect prediction
        ForecastSkillScore = calculateForecastSkillScore(actual=self.SERIES_test, predicted=Prediction)
        
        self.assertTrue((ForecastSkillScore > 80).all(), 'ForecastSkillScore should be larger than 90%')
        

        #    takes long time to run, therefore, we dont use the test
        #    def test_lstm(self):
        #        
        #        defineFitPredict = defineFitPredict_LSTM
        #        #define, fit model; predict with with model
        #        Prediction, Model = defineFitPredict(C=self.C,
        #                                             D=self.D, 
        #                                             VARIABLES_train=self.VARIABLES_train, 
        #                                             SERIES_train=self.SERIES_train, 
        #                                             VARIABLES_test=self.VARIABLES_test,
        #                                             SERIES_test=self.SERIES_test)
        #        
        #        # calculate ForecastSkillScore : 0% as good as NULL-Hypothesis, 100% is perfect prediction
        #        ForecastSkillScore = calculatePredictionForecastSkillScore(actual=self.SERIES_test, predicted=Prediction)
        #        
        #        self.assertTrue((ForecastSkillScore > 80).all(), 'ForecastSkillScore should be larger than 90%')

        
    
# for easy debugging self=Test_Predict()
        
        