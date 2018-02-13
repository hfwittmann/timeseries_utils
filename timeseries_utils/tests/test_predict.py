#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:53:25 2018

@author: hfwittmann
"""

# from matplotlib import pyplot

import numpy as np
from unittest import TestCase

from timeseries_utils import artificial_data, prepareModel, predict, fit_prepare, fitMultipleUnivariateSeries, save, load
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler

from math import sqrt, log

class TestPredict(TestCase):
    
    
    nOfPoints = 1200
    nOfSeries = 11
    f_rauschFactor = 0.5
    
    inSampleRatio = 0.7
    inSample = int(nOfPoints * inSampleRatio)
    
    p_order= 7
    steps_prediction=1
    
    # generate artifical data
    data_x, data_y = artificial_data(nOfPoints=nOfPoints,
            nOfSeries=nOfSeries,
            f_rauschFactor=f_rauschFactor
            )
    
    # normalize features
    scaler = RobustScaler()
    # scaler = None
    
    def test_basics(self):
        
        self.assertTrue(True)
        
    
    def test_prediction_multivariate_no_diff(self):
        
        filename = 'test_predict___test_prediction_multivariate_no_diff'
        
        try:
            
            myobject = load(directory='cache', name=filename)
            
            X, Y, test_X, test_Y, X_diff, Y_diff, test_X_diff, test_Y_diff, \
                X_fit, Y_fit, test_X_fit, test_Y_fit, X_diff_fit, Y_diff_fit, test_X_diff_fit, test_Y_diff_fit, \
                test_Y_prediction, \
                test_Y_prediction_arima \
            \
            = myobject
            

        except:
            
            M = prepareModel(nOfSeries=self.nOfSeries,p_order=self.p_order)
            
            X, Y, test_X, test_Y, X_diff, Y_diff, test_X_diff, test_Y_diff, \
            X_fit, Y_fit, test_X_fit, test_Y_fit, X_diff_fit, Y_diff_fit, test_X_diff_fit, test_Y_diff_fit \
            = fit_prepare(self.data_y, inSampleRatio = 0.7, differencingOrder=0, differencingAxis = 0, \
                          p_order=self.p_order, scaler=None)
            
            M.fit(X_diff_fit,
                            Y_diff_fit,
                            epochs=10, 
                            batch_size=72, 
                            validation_data=(test_X_diff_fit, test_Y_diff_fit), 
                            verbose=0, 
                            shuffle=True)
            
            test_Y_prediction = predict(test_X_fit, test_Y_fit, model=M,
                            differencingOrder=1, differencingAxis = 0,
                            scaler=None)
            
            test_Y_prediction_arima, model = fitMultipleUnivariateSeries(self.data_y, percentageTraining = self.inSampleRatio)
            
            # .. now save the data
            myobject = \
            \
            X, Y, test_X, test_Y, X_diff, Y_diff, test_X_diff, test_Y_diff, \
                X_fit, Y_fit, test_X_fit, test_Y_fit, X_diff_fit, Y_diff_fit, test_X_diff_fit, test_Y_diff_fit, \
                test_Y_prediction, \
                test_Y_prediction_arima
            
            save(myobject=myobject, directory='cache', name=filename)
        

        #        X.shape
        #        Out[132]: (833, 7, 11)

        #        M.compute_output_shape(input_shape=X.shape)
        #        Out[131]: (833, 11)
        
        finally:
                    
            self.assertEqual(X.shape, X_fit.shape, 'Shapes of X and X_fit should match')
            self.assertEqual(X.shape, X_diff.shape, 'Shapes of X and X_diff should match')
            self.assertEqual(X.shape, X_diff_fit.shape, 'Shapes of X and X_diff_fit should match')
            
            self.assertEqual(test_X.shape, test_X_fit.shape, 'Shapes of test_X and test_X_fit should match')
            self.assertEqual(test_X.shape, test_X_diff.shape, 'Shapes of X and X_diff should match')
            self.assertEqual(test_X.shape, test_X_diff_fit.shape, 'Shapes of X and X_diff_fit should match')
            
            self.assertEqual(X.shape[1:], test_X.shape[1:], 'Shapes of X and test_X should match partially')
            self.assertEqual(X.shape[1:], test_X.shape[1:], 'Shapes of X and test_X should match partially')
    
    
            assert test_X.shape[0] == int(self.nOfPoints * (1 - self.inSampleRatio))- self.steps_prediction + 1, \
                'Prediction is shortened by Sample and steps_prediction, but not by p_order!'
    
            self.assertEqual(X.shape[1], test_X.shape[1], 'Shapes of X and test_X should match partially')
            self.assertEqual(X.shape[2], test_X.shape[2], 'Shapes of X and test_X should match partially')
    
            
            self.assertEqual(Y.shape, Y_fit.shape, 'Shapes of Y and Y_fit should match')
            self.assertEqual(Y.shape, Y_diff.shape, 'Shapes of Y and Y_diff should match')
    
                    
    
            
            #        pyplot.plot(test_Y_prediction_arima)
            #        pyplot.plot(self.data_y[self.inSample + self.p_order :] )
            #        
            #        pyplot.plot(test_Y_prediction)
            #        pyplot.plot(test_Y)
            #        pyplot.plot(test_Y_prediction_arima)
            #        
            #        pyplot.plot(test_Y_prediction-test_Y)
            #        pyplot.plot(test_Y_prediction_arima-test_Y)
            
            
            rmse = sqrt(mean_squared_error(test_Y, test_Y_prediction))
            rmse_arima = sqrt(mean_squared_error(test_Y, test_Y_prediction_arima))
        
            roughlyEqual = np.abs(log(rmse_arima/rmse)) < 0.5
            self.assertTrue(not roughlyEqual, "LSTM and arima should not deliver similar results")
                
            self.assertTrue(rmse_arima*5<rmse, "LSTM with no differencing is much worse, in this series of mutiple univeriate series")


    
    def test_prediction_multivariate(self):        
    
        filename = 'test_predict___test_prediction_multivariate'
        
        try:
            
            myobject = load(directory='cache', name=filename)
            
            X, Y, test_X, test_Y, X_diff, Y_diff, test_X_diff, test_Y_diff, \
                X_fit, Y_fit, test_X_fit, test_Y_fit, X_diff_fit, Y_diff_fit, test_X_diff_fit, test_Y_diff_fit, \
                test_Y_prediction, \
                test_Y_prediction_arima \
            \
            = myobject
            

        except:
            
            M = prepareModel(nOfSeries=self.nOfSeries,p_order=self.p_order)
            
            X, Y, test_X, test_Y, X_diff, Y_diff, test_X_diff, test_Y_diff, \
            X_fit, Y_fit, test_X_fit, test_Y_fit, X_diff_fit, Y_diff_fit, test_X_diff_fit, test_Y_diff_fit \
            = fit_prepare(self.data_y, inSampleRatio = 0.7, differencingOrder=1, differencingAxis = 0, \
                          p_order=self.p_order, scaler=None)
            
            M.fit(X_diff_fit,
                            Y_diff_fit,
                            epochs=10, 
                            batch_size=72, 
                            validation_data=(test_X_diff_fit, test_Y_diff_fit), 
                            verbose=0, 
                            shuffle=True)
            
            test_Y_prediction = predict(test_X_fit, test_Y_fit, model=M,
                            differencingOrder=1, differencingAxis = 0,
                            scaler=None)
            
            test_Y_prediction_arima, model = fitMultipleUnivariateSeries(self.data_y, percentageTraining = self.inSampleRatio)
            
            # .. now save the data
            myobject = \
            \
            X, Y, test_X, test_Y, X_diff, Y_diff, test_X_diff, test_Y_diff, \
                X_fit, Y_fit, test_X_fit, test_Y_fit, X_diff_fit, Y_diff_fit, test_X_diff_fit, test_Y_diff_fit, \
                test_Y_prediction, \
                test_Y_prediction_arima \
            
            save(myobject=myobject, directory='cache', name=filename)
        
        
        finally:
    
    
            #        X.shape
            #        Out[132]: (833, 7, 11)
    
            #        M.compute_output_shape(input_shape=X.shape)
            #        Out[131]: (833, 11)
        
                    
            
            #        pyplot.plot(test_Y_prediction_arima)
            #        pyplot.plot(self.data_y[self.inSample + self.p_order :] )
            #        
            #        pyplot.plot(test_Y_prediction)
            #        pyplot.plot(test_Y)
            #        pyplot.plot(test_Y_prediction_arima)
            #        
            #        pyplot.plot(test_Y_prediction-test_Y)
            #        pyplot.plot(test_Y_prediction_arima-test_Y)
            
            
            rmse = sqrt(mean_squared_error(test_Y, test_Y_prediction))
            rmse_arima = sqrt(mean_squared_error(test_Y, test_Y_prediction_arima))
        
            roughlyEqual = np.abs(log(rmse_arima/rmse)) < 0.5
            self.assertTrue(roughlyEqual, "LSTM and arima should deliver similar results")



    def test_prediction_univariate(self):       
        
        filename = 'test_predict___test_prediction_univariate'
        
        try:
            
            myobject = load(directory='cache', name=filename)
            
            X, Y, test_X, test_Y, X_diff, Y_diff, test_X_diff, test_Y_diff, \
                X_fit, Y_fit, test_X_fit, test_Y_fit, X_diff_fit, Y_diff_fit, test_X_diff_fit, test_Y_diff_fit, \
                test_Y_prediction, \
                test_Y_prediction_arima \
            \
            = myobject
            

        except:
            
            M = prepareModel(nOfSeries=1,p_order=self.p_order)
            
            series = 5 # the series to be fitted            
            
            X, Y, test_X, test_Y, X_diff, Y_diff, test_X_diff, test_Y_diff, \
            X_fit, Y_fit, test_X_fit, test_Y_fit, X_diff_fit, Y_diff_fit, test_X_diff_fit, test_Y_diff_fit \
            = fit_prepare(self.data_y[:,[series]], inSampleRatio = 0.7, differencingOrder=1, differencingAxis = 0, \
                          p_order=self.p_order, scaler=None)
            
            M.fit(X_diff_fit,
                            Y_diff_fit,
                            epochs=10, 
                            batch_size=72, 
                            validation_data=(test_X_diff_fit, test_Y_diff_fit), 
                            verbose=0, 
                            shuffle=True)
            
            
            test_Y_prediction = predict(test_X_fit, test_Y_fit, model=M,
                            differencingOrder=1, differencingAxis = 0,
                            scaler=None)
            
            test_Y_prediction_arima, model = fitMultipleUnivariateSeries(self.data_y[:,[series]], percentageTraining = self.inSampleRatio)
            
            # .. now save the data
            myobject = \
            \
            X, Y, test_X, test_Y, X_diff, Y_diff, test_X_diff, test_Y_diff, \
                X_fit, Y_fit, test_X_fit, test_Y_fit, X_diff_fit, Y_diff_fit, test_X_diff_fit, test_Y_diff_fit, \
                test_Y_prediction, \
                test_Y_prediction_arima \
            
            save(myobject=myobject, directory='cache', name=filename)
            
        finally:
        
            # nOfSeries=1 because it's univariate
            
            
            #        X.shape
            #        Out[132]: (833, 7, 11)
            
            #        M.compute_output_shape(input_shape=X.shape)
            #        Out[131]: (833, 11)
            
    
            #        pyplot.plot(test_Y_prediction_arima)
            #        pyplot.plot(self.data_y[self.inSample + self.p_order :,[series]] )
            #        
            #        pyplot.plot(test_Y_prediction)
            #        pyplot.plot(test_Y)
            #        pyplot.plot(test_Y_prediction_arima)
            #        
            #        pyplot.plot(test_Y_prediction-test_Y)
            #        pyplot.plot(test_Y_prediction_arima-test_Y)
                            
                
                    
            rmse = sqrt(mean_squared_error(test_Y, test_Y_prediction))
            rmse_arima = sqrt(mean_squared_error(test_Y, test_Y_prediction_arima))
        
            roughlyEqual = np.abs(log(rmse_arima/rmse)) < 0.5
            self.assertTrue(roughlyEqual, "LSTM and arima should deliver similar results")
        
        
        
    def test_prediction_univariate_with_scaler(self):    
        
        filename = 'test_predict___test_prediction_univariate_with_scaler'
        
        try:
            
            myobject = load(directory='cache', name=filename)
            
            X, Y, test_X, test_Y, X_diff, Y_diff, test_X_diff, test_Y_diff, \
                X_fit, Y_fit, test_X_fit, test_Y_fit, X_diff_fit, Y_diff_fit, test_X_diff_fit, test_Y_diff_fit, \
                test_Y_prediction, \
                test_Y_prediction_arima \
            \
            = myobject
            

        except:
            
            M = prepareModel(nOfSeries=1,p_order=self.p_order)
            
            series = 5 # the series to be fitted
            
            
            X, Y, test_X, test_Y, X_diff, Y_diff, test_X_diff, test_Y_diff, \
            X_fit, Y_fit, test_X_fit, test_Y_fit, X_diff_fit, Y_diff_fit, test_X_diff_fit, test_Y_diff_fit \
            = fit_prepare(self.data_y[:,[series]], inSampleRatio = 0.7, differencingOrder=1, differencingAxis = 0, \
                  scaler=self.scaler)
            
            M.fit(X_diff_fit,
                            Y_diff_fit,
                            epochs=10, 
                            batch_size=72, 
                            validation_data=(test_X_diff_fit, test_Y_diff_fit), 
                            verbose=0, 
                            shuffle=True)
            
            
            test_Y_prediction = predict(test_X_fit, test_Y_fit, model=M,
                            differencingOrder=1, differencingAxis = 0,
                            scaler=self.scaler)
            
            test_Y_prediction_arima, model = fitMultipleUnivariateSeries(self.data_y[:,[series]], percentageTraining = self.inSampleRatio)
            
            # .. now save the data
            myobject = \
            \
            X, Y, test_X, test_Y, X_diff, Y_diff, test_X_diff, test_Y_diff, \
                X_fit, Y_fit, test_X_fit, test_Y_fit, X_diff_fit, Y_diff_fit, test_X_diff_fit, test_Y_diff_fit, \
                test_Y_prediction, \
                test_Y_prediction_arima \
            
            save(myobject=myobject, directory='cache', name=filename)
            
        finally:
        
            # nOfSeries=1 because it's univariate
            
            

        
            # nOfSeries=1 because it's univariate
            
            
            #        X.shape
            #        Out[132]: (833, 7, 11)
            
            #        M.compute_output_shape(input_shape=X.shape)
            #        Out[131]: (833, 11)
            
                    
            #        pyplot.plot(test_Y_prediction_arima)
            #        pyplot.plot(self.data_y[self.inSample + self.p_order :,[series]] )
            #        
            #        pyplot.plot(test_Y_prediction)
            #        pyplot.plot(test_Y)
            #        pyplot.plot(test_Y_prediction_arima)
            #        
            #        pyplot.plot(test_Y_prediction-test_Y)
            #        pyplot.plot(test_Y_prediction_arima-test_Y)
                    
                    
            rmse = sqrt(mean_squared_error(test_Y, test_Y_prediction))
            rmse_arima = sqrt(mean_squared_error(test_Y, test_Y_prediction_arima))
        
            roughlyEqual = np.abs(log(rmse_arima/rmse)) < 0.5
            self.assertTrue(roughlyEqual, "LSTM and arima should deliver similar results")
        
# for convenient debugging use self = TestPredict()
