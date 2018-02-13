#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:53:25 2018

@author: hfwittmann
"""

# from matplotlib import pyplot
import numpy as np

from unittest import TestCase

from timeseries_utils import artificial_data, prepareData, prepareModel, save, load


class TestPrepare(TestCase):
    
    
    nOfPoints = 1200
    nOfSeries = 11
    f_rauschFactor = 0.2
    
    inSampleRatio = 0.7
    splitPoint = int(nOfPoints * inSampleRatio)
    
    p_order= 7
    steps_prediction=1
    
    differencingOrder = 1
    differencingAxis = 0
    
    filename = 'test_prepare___'
    
    try:
        myobject = load('cache', name=filename)

        data_x, data_y,\
        MVS,\
        preparedD, inSample, outOfSample \
        \
        =myobject

        
    
    except:
    
        # generate artifical data
        data_x, data_y = artificial_data(nOfPoints=nOfPoints,
                nOfSeries=nOfSeries,
                f_rauschFactor=f_rauschFactor
                )
        

        
        MVS = np.diff(data_y, n=differencingOrder, axis=differencingAxis)
            
        preparedD = prepareData(MVS = MVS, inSampleRatio = inSampleRatio, p_order= p_order, steps_prediction=steps_prediction)
        inSample = preparedD['inSample']
        outOfSample = preparedD['outOfSample']  
        
        myobject = \
        \
            data_x, data_y,\
            MVS,\
            preparedD, inSample, outOfSample
            
        save(myobject=myobject, directory='cache', name=filename)
    
    
    def test_basics(self):
        
        self.assertTrue(True)
        
        
    def test_shapes(self):
        

        self.assertEqual(self.splitPoint - self.p_order - self.differencingOrder, self.inSample['X'].shape[0], \
                         'Should be the proportion of points specified by the inSampleRatio minus self.p_order, because of the removal of NAs')
            
        
        return None
        
    
    def test_shapes_prediction_multivariate(self):        
        
        X = self.inSample['X']
        Y = self.inSample['Y']
                
        M = prepareModel(nOfSeries=self.nOfSeries,p_order=self.p_order)
        self.assertEqual(Y.shape, M.compute_output_shape(input_shape=X.shape), 'Shapes should match')
        

    def test_shapes_prediction_univariate(self):        
        
        series = 5 # the series to be fitted
         
        X = self.inSample['X'][:,:,[series]]
        Y = self.inSample['Y'][:,[series]]
                
                
        M = prepareModel(nOfSeries=1,p_order=self.p_order)

        
        self.assertEqual(Y.shape, M.compute_output_shape(input_shape=X.shape), 'Shapes should match')
        


        
# for convenient debugging use self = TestPrepare()
