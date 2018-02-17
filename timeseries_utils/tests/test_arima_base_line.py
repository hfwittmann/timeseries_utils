#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 21:47:26 2018

@author: hfwittmann
"""

from timeseries_utils.artificial_data import artificial_data       
from timeseries_utils.arima_baseline import fitMultipleUnivariateSeries
from timeseries_utils import Configuration, Data

from unittest import TestCase

#from matplotlib import pyplot

class TestArimaBaseLine(TestCase):
    
    def test_basics(self):


        
        C = Configuration()
        C.dictionary['nOfPoints'] = 1000
        C.dictionary['inSampleRatio'] = 0.5
        
        D = Data()        
        
        nOfSeries = 15
        x, MUS = artificial_data(nOfPoints=C.dictionary['nOfPoints'], nOfSeries=nOfSeries, f_rauschFactor=0.9)
        
        for S in range(nOfSeries):
            
            D.setFollower(name = 'signal'+str(S),
                          otherSeriesName = 'innovator plus signal',
                          otherSeries=MUS[:,-S],
                          shiftBy=None)


        SERIES_train = D.train(D.SERIES(), Configuration=C)
        SERIES_test = D.test(D.SERIES(), Configuration=C)
        
        FC, FITS = fitMultipleUnivariateSeries(SERIES_train, SERIES_test)

        self.assertEqual(C.dictionary['nOfPoints']*(1-C.dictionary['inSampleRatio']), FC.shape[0])
        self.assertEqual(nOfSeries, FC.shape[1])
        
        return None


# for easy debugging self = TestArimaBaseLine()
