#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 13:00:07 2018

@author: hfwittmann
"""

from unittest import TestCase
from timeseries_utils.artificial_data import artificial_data

import numpy as np

class TestArtificalData(TestCase):
    
    def test_basic(self):
        
        self.assertEqual(1,1, 'Should be equal')


    def  test_artifical_data(self):
        
        Points = 1031
        Series = 7
        Factor = 0.05
        
        output_x, output_y = artificial_data(Points, Series, Factor)

        self.assertEqual((Points, Series), \
                         output_y.shape, \
                         'Output shape should be determined by function input')

        self.assertEqual((Points, 1), \
                         output_x.shape, \
                         "Output shape of x-values should be Points x 1"
                         )        
        
        # omit optional input
        output_x_2, output_y_2 = artificial_data(Points, Series)

        self.assertEqual((Points, Series), \
                         output_y_2.shape, \
                         'Output shape should be determined by function input')
        
        
        self.assertEqual((Points, 1), \
                         output_x_2.shape, \
                         "Output shape of x-values should be Points x 1"
                         )        


    def test_random_ness(self):
        
        
        Points = 51
        Series = 27
        factor_0 = 0
        factor_1 = 0.1
        factor_2 = 0.2
        
        
        np.random.seed(167)        
        output_x_0, output_y_0 = artificial_data(Points, Series, factor_0)
        
        np.random.seed(167)        
        output_x_1, output_y_1 = artificial_data(Points, Series, factor_1)

        np.random.seed(167)        
        output_x_2, output_y_2 = artificial_data(Points, Series, factor_2)

        
        self.assertTrue ( (output_x_0 == output_x_1).all(), 'X Values should be the same')
        self.assertTrue ( (output_x_0 == output_x_2).all(), 'X Values should be the same')
        
        
        Randomness_1 = output_y_1 - output_y_0
        Randomness_2 = output_y_2 - output_y_0
        
        Randomness_1_normalised = Randomness_1/factor_1
        Randomness_2_normalised = Randomness_2/factor_2
        
        DIFF_of_Randomnesses = np.abs(Randomness_1_normalised - Randomness_2_normalised)

        ROUGHLY_ZERO = ( DIFF_of_Randomnesses < 1e-10).all()
    
        self.assertTrue (ROUGHLY_ZERO, 'output_y_0 has no randomness, \
                         output_y_2 should have twice the randomness as output_y_1')
        
        
        
        
        
        
        


# for easy debugging self = TestArtificalData()
