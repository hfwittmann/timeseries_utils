#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:18:33 2018

@author: hfwittmann
"""

from unittest import TestCase
import numpy as np

from timeseries_utils import Configuration, Data

class Test_Data_Configuation(TestCase):


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
    nOfSeries = 5
    splitPoint = int(C.dictionary['inSampleRatio'] * C.dictionary['nOfPoints'])
    
    for S in range(nOfSeries):
        D.setInnovator('innovator' + str(S), C=C)
        
    
    SERIES = D.SERIES()
    SERIES_train = D.train(SERIES, Configuration=C)
    SERIES_test = D.test(SERIES, Configuration=C)
    
    VARIABLES = D.VARIABLES()
    VARIABLES_train = D.train(VARIABLES, Configuration=C)
    VARIABLES_test = D.test(VARIABLES, Configuration=C)
    
    def test_basic(self):
        
        self.assertEqual(self.SERIES.shape, (self.C.dictionary['nOfPoints'] , self.nOfSeries) )
        self.assertEqual(self.SERIES_train.shape, (self.splitPoint, self.nOfSeries) )


        self.assertEqual(self.VARIABLES.shape, (self.C.dictionary['nOfPoints'], self.C.dictionary['p_order'], self.nOfSeries) )
        self.assertEqual(self.VARIABLES_train.shape, (self.splitPoint, self.C.dictionary['p_order'], self.nOfSeries) )
       
        return None

# for easy debugging self=Test_Data_Configuation()