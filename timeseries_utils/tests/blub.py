#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 10:56:51 2018

@author: hfwittmann
"""

defineFitPredict = defineFitPredict_DENSE
#define, fit model; predict with with model
Prediction, Model = defineFitPredict(C=self.C,
                                     D=self.D, 
                                     VARIABLES_train=self.VARIABLES_train, 
                                     SERIES_train=self.SERIES_train, 
                                     VARIABLES_test=self.VARIABLES_test,
                                     SERIES_test=self.SERIES_test)

# calculate ForecastSkillScore : 0% as good as NULL-Hypothesis, 100% is perfect prediction
ForecastSkillScore = calculateForecastSkillScore(actual=self.SERIES_test, predicted=Prediction)