#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 21:33:16 2018

@author: hfwittmann
"""

import numpy as np
from unittest import TestCase
from timeseries_utils.differences import diffinv, diffinv_multi


class TestDiffs(TestCase):


    series1 = np.sin(1 * np.pi * np.arange(1000)/1000)
    series2 = np.sin(2 * np.pi * np.arange(1000)/1000)
    series3 = np.sin(3 * np.pi * np.arange(1000)/1000)
     
    multiSeries = np.vstack((series1,series2,series3)).T
    
    def test_univariate(self):
    
        for n in range(10):
                                
            D_series = np.diff(self.series1,n=n)        
            series_reconstructed = diffinv(D_series=D_series, differences=n, getxiFrom = self.series1)
            # first diffed then cumsummed 
            
            self.assertTrue( np.all(1e-10 > series_reconstructed - self.series1), 
                            'Should be the same after n-differencing and n-integrating')
        

    def test_multivariate(self):
    
        for n in range(10):
                                
            D_series = np.diff(self.multiSeries.T,n=n).T
            nOfSeries = self.multiSeries.shape[1]
            
            multi_list = list()
            
            for S in range(nOfSeries):
                series_reconstructed = diffinv(D_series = D_series[:,S], differences=n, getxiFrom = self.multiSeries[:,S])
                multi_list.append(series_reconstructed)
                
            multiseries_reconstructed = np.vstack(multi_list).T
            
            self.assertTrue( np.all(1e-10 > multiseries_reconstructed - self.multiSeries), 
                            'Should be the same after n-differencing and n-integrating')



    def test_multivariate_2(self):
    
        for n in range(10):
                                
            multi_DSeries = np.diff(self.multiSeries.T,n=n).T

                
            multiseries_reconstructed = \
                diffinv_multi(multi_DSeries = multi_DSeries, differences=n, getxiFrom = self.multiSeries)
            
            self.assertTrue( np.all(1e-10 > multiseries_reconstructed - self.multiSeries), 
                            'Should be the same after n-differencing and n-integrating')
            
            
    def test_multivariate_3(self):
    
        for n in range(10):
                                
            multi_DSeries = np.diff(self.multiSeries,n=n, axis=0)

                
            multiseries_reconstructed = \
                diffinv(D_series = multi_DSeries, differences=n, axis=0, getxiFrom = self.multiSeries)
            
            self.assertTrue( np.all(1e-10 > multiseries_reconstructed - self.multiSeries), 
                            'Should be the same after n-differencing and n-integrating')


# for easy debugging self = TestDiffs()


