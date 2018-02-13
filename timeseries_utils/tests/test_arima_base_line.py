#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 21:47:26 2018

@author: hfwittmann
"""

from timeseries_utils.artificial_data import artificial_data       
from timeseries_utils.arima_baseline import fitMultipleUnivariateSeries

from unittest import TestCase

#from matplotlib import pyplot

class TestArimaBaseLine(TestCase):
    
    def test_basics(self):

        nOfPoints = 1000
        nOfSeries = 15
        percentageTraining = 0.3
        trainingLength = int(nOfPoints*percentageTraining)
        
        
        x, MUS = artificial_data(nOfPoints=nOfPoints, nOfSeries=nOfSeries, f_rauschFactor=0.9)
        
        FC, FITS = fitMultipleUnivariateSeries(MUS, percentageTraining)

        self.assertEqual(nOfPoints-trainingLength, FC.shape[0])
        self.assertEqual(nOfSeries, FC.shape[1])
        
        return None


# for easy debugging self = TestArimaBaseLine()
