# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:29:35 2016

@author: hfwittmann
"""

import argparse
import sys

import timeseries_utils


def main():
    

    parsed_args = myparse_args(sys.argv[1:])
    
    
    
    out = timeseries_utils.artificial_data(**vars(parsed_args))
    
    return out


# https://stackoverflow.com/questions/18160078/how-do-you-write-tests-for-the-argparse-portion-of-a-python-module
def myparse_args(args):
    
    ''' Example of taking inputs'''
    
    parser = argparse.ArgumentParser(prog = 'My artifical data program')
    
    # https://stackoverflow.com/questions/2086556/specifying-a-list-as-a-command-line-argument-in-python
    # parser.add_option("-t", "--tracks", action="append", type="int")
    
    
    parser.add_argument('-P', '--nOfPoints', nargs = '?', type=int, \
                        help='help for --nOfPoints : This expects a number to specify the number of points in the time series')

    parser.add_argument('-S', '--nOfSeries', nargs = '?', type=int, \
                        help='help for --nOfSeries: This expects a number to specify the number of time series')


    parser.add_argument('-R', '--f_rauschFactor', nargs = '?', type=float, \
                        help='help for --rauschFactor: This expects a number to specify the noise-2-signal ratio')
    
    args_parsed = parser.parse_args(args)
    
    return args_parsed
