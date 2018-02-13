# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 18:16:43 2016

@author: hfwittmann
"""
    
import sys
    
from unittest import TestCase
from timeseries_utils.command_line import main, myparse_args

class TestConsole(TestCase):
    
    def test_parser(self):
        
        Points = 31
        
        parsed = myparse_args(['--nOfPoints', str(Points)])
        self.assertEqual(parsed.nOfPoints, 31)


    def test_commandline(self):
        
        Points = 31
        Series = 23
        Factor = 0.5
        
        sys.argv[1:] = ['--nOfPoints', str(Points)]
        sys.argv[3:] = ['--nOfSeries', str(Series)]
        sys.argv[5:] = ['--f_rauschFactor', str(Factor)]
        
        output_x, output_y = main()
        
        self.assertEqual((Points, Series),
                             output_y.shape,
                            'Output shape should be determined by function input')
        
# for simple debugging : self = TestConsole()
