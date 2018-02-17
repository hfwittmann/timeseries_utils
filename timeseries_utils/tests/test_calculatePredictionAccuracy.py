#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:02:43 2018

@author: hfwittmann
"""

from unittest import TestCase

import numpy as np
from timeseries_utils import calculatePredictionAccuracy

class TestCalculateAccuracy(TestCase):
    
    def test_basic(self):
        
        self.assertTrue(True,'Should be true')
        
        
        series = np.random.normal(size=[100,1])

        series_actual = series.copy()
        
        series_predicted = series.copy()
        accuracy = calculatePredictionAccuracy(actual=series_actual, predicted=series_predicted)
        
        self.assertEqual(int(accuracy), 100, 'Should be 100% accurate')

        series_predicted = 2 * series.copy()
        accuracy = calculatePredictionAccuracy(actual=series_actual, predicted=series_predicted)
        
        self.assertEqual(int(accuracy), 0, 'Should be 0% accurate')

        
        
        return None
        
# for easy debugging self=TestCalculateAccuracy()
        