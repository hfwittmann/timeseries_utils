#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 12:33:37 2018

@author: hfwittmann
"""

import math
import numpy as np
import scipy.stats as stats


def artificial_data(nOfPoints, nOfSeries, f_rauschFactor=0.5):
    
    # --nOfPoints : This expects a number to specify the number of points in the time series'

    # --nOfSeries: This expects a number to specify the number of time series'

    # --f_rauschFactor', type=float, help='help for --rauschFactor: This expects a number to specify the noise-2-signal ratio'  
    
    
    x = np.linspace(-1, 1, num=nOfPoints)
    
    # each series has a different period
    periods = 10 ** np.linspace(start=0, stop=-1, num=nOfSeries)
    
    # each series has a different offset
    offsets = 4 * np.linspace(start=0, stop=nOfSeries, num=nOfSeries)
    
    # each series has a different mutiplier. The multiplier is used to control the signalSize
    multiplier = np.linspace(start=0, stop=nOfSeries, num=nOfSeries)
    
    y_Signal = np.zeros([nOfPoints, nOfSeries]) * np.NAN
    
    for s in range(nOfSeries):
        
        y_Signal[:, s] = offsets[s] + \
            multiplier[s] * \
            np.sin(2 * math.pi + x/periods[s]) * \
            np.exp(- 2 * x)
            
        
    y_Rauschen = np.random.uniform(low=-1, high=1, size=[nOfPoints, nOfSeries])


    y = (1-f_rauschFactor) * y_Signal + f_rauschFactor * y_Rauschen


    x = x.reshape(-1, 1)
    
    

    return np.array(x), np.array(y)
                