#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:40:29 2018

@author: hfwittmann
"""

import numpy as np

import sortedcontainers
import infix

from timeseries_utils import series_to_supervised

@infix.or_infix
def implies(x, y):
    #https://stackoverflow.com/questions/16405892/is-there-an-implication-logical-operator-in-python
    return (not x) or y


class Data:
    def __init__(self):

        self.dictionary = sortedcontainers.SortedDict()
        self.dictionaryTimeseries = sortedcontainers.SortedDict()

    def __repr__(self):
        return "Data()"

    def __str__(self):
        return self.renderDictionary()

    def setInnovator(self, name, C):

        self.dictionaryTimeseries[name] = np.random.normal(
            size=C.dictionary['nOfPoints'])

        self.dictionary[name] = {'nOfPoints': C.dictionary['nOfPoints']}

        return None

    def setFollower(self,
                    name,
                    otherSeriesName,
                    otherSeries=None,
                    shiftBy=None):
        '''
        There are two cases:
        Simple)
            the follower follows another series directly. 
            In this case it is sufficent to specify the otherSeriesName
        Complicated)
            the follower follows some more complicated expression such as 
            sin(D.dictionaryTimeseries['innovator1']) + cos(D.dictionaryTimeseries['follower2'])
            
        In the complicated case the calculation of the follower series must be done outside the class, 
        and then handed over to otherSeries.
        
        '''

        if (otherSeries is None):
            otherSeries = self.dictionaryTimeseries[otherSeriesName]
            
        if (shiftBy is not None):
            assert shiftBy < 0, "The series is supposed to be a follower series"
        
            # shiftBy=None means it is not shifted
            self.dictionaryTimeseries[name] = np.zeros(
                shape=otherSeries.shape)  # initialize
            self.dictionaryTimeseries[name][-shiftBy:] = otherSeries[:shiftBy]

        elif (shiftBy is None):
            # shiftBy=None means it is not shifted
            self.dictionaryTimeseries[name] = otherSeries
            # shiftBy=None means it is not shifted

            
            
        
        self.dictionary[name] = {
            'follows': otherSeriesName,
            'shiftedBy': shiftBy
        }

        return None

    def SERIES(self):

        return np.vstack(self.dictionaryTimeseries.values()).T

    def VARIABLES(self, p_order=7):

        SERIES = self.SERIES()

        # add dimension of lagged values
        out_pre = np.array(
            series_to_supervised(SERIES, n_in=p_order, n_out=0, dropnan=False))

        # remove nans
        out_pre_2 = np.nan_to_num(out_pre)

        out = out_pre_2.reshape([SERIES.shape[0], p_order, SERIES.shape[1]])

        return out

    def train(self, x, Configuration):
        # in sample
        return x[:Configuration.splitPoint()]

    def test(self, x, Configuration):
        # out of sample
        return x[Configuration.splitPoint():]

    def calculateShape(self):

        self.dictionary['SERIES-shape'] = self.SERIES().shape

    def numberOfSeries(self):

        return self.SERIES().shape[1]

    def renderDictionary(self):

        self.calculateShape()

        out = ''

        for c in self.dictionary:
            out += '\n - ' + c + ':' + str(self.dictionary[c])

        return out
