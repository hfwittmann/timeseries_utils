#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 09:38:52 2018

@author: hfwittmann
"""

import sortedcontainers

class Configuration:
    def __init__(self):

        self.dictionary = sortedcontainers.SortedDict()
        
        self.dictionary['epochs'] = 20
        self.dictionary['verbose'] = 0
        self.dictionary['differencingOrder'] = 0
        self.dictionary['steps_prediction'] = 1
        self.dictionary['modelname'] = 'modelname'

    def __repr__(self):
        return "Configuration()"

    def __str__(self):
        return self.renderDictionary()

    # split point between training and testing
    def splitPoint(self):

        if ('nOfPoints' in self.dictionary) and (
                'inSampleRatio' in self.dictionary):
            self.dictionary['splitPoint'] = int(
                self.dictionary['nOfPoints'] *
                self.dictionary['inSampleRatio'])
            return self.dictionary['splitPoint']

    def renderDictionary(self):

        self.splitPoint()

        out = ''
        for c in self.dictionary:
            out += '\n - ' + c + ':' + str(self.dictionary[c])

        return out