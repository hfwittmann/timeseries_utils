#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:02:43 2018

@author: hfwittmann
"""

from unittest import TestCase

import numpy as np
from timeseries_utils import calculateForecastSkillScore, calcMovingAverage

class TestCalculateAccuracy(TestCase):
    
    def test_basic(self):
        
        self.assertTrue(True,'Should be true')
        
        
        series = np.random.normal(size=[100,1])

        series_actual = series.copy()
        
        series_predicted = series.copy()
        accuracy = calculateForecastSkillScore(actual=series_actual, predicted=series_predicted)
        
        self.assertEqual(int(accuracy), 100, 'Should be 100% accurate')

        series_predicted = 2 * series.copy()
        accuracy = calculateForecastSkillScore(actual=series_actual, predicted=series_predicted)
        
        self.assertEqual(int(accuracy), 0, 'Should be 0% accurate')

        
        
        return None
    
    
    def test_skill(self):
        
        self.assertTrue(True,'Should be true')
        
        
        series = np.random.normal(size=[100,1])

        series_actual = series.copy()
        
        series_predicted = series.copy()
        accuracy = calculateForecastSkillScore(actual=series_actual, predicted=series_predicted)
        
        self.assertEqual(int(accuracy), 100, 'Should be 100% accurate')

        series_predicted = 2 * series.copy()
        accuracy = calculateForecastSkillScore(actual=series_actual, predicted=series_predicted)
        
        self.assertEqual(int(accuracy), 0, 'Should be 0% accurate')
        
        
        return None
    
    
    def test_skill_movingAverage(self):
        
        self.assertTrue(True,'Should be true')
        
        
        series = np.random.normal(size=[100,1])

        series_actual = series.copy()
        
        series_predicted = series.copy()
        accuracy = calculateForecastSkillScore(actual=series_actual, predicted=series_predicted, movingAverage=1)
        
        self.assertEqual(int(accuracy), 100, 'Should be 100% accurate')

        series_actual_cum = np.cumsum(series_actual, axis=0)
        series_predicted_cum = series_actual_cum.copy() + 0.5 * np.random.normal(size=[100,1])
        # series_actual_cum_ma = calcMovingAverage(series_actual_cum, movingAverage=1)
        
        accuracy = calculateForecastSkillScore(actual=series_actual_cum, predicted=series_predicted_cum , movingAverage=1)
        
        self.assertTrue(int(accuracy)> 50, 'Performs better than chance')
        
        
        return None    
    
    
        
# for easy debugging self=TestCalculateAccuracy()
        