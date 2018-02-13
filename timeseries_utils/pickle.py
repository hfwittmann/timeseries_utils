#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:18:40 2018

@author: hfwittmann
"""

import pickle

import os
import warnings

    
def save(myobject, directory, name):
    
    
    directory_relative = os.path.relpath(directory)
    
    if not os.path.exists(directory_relative):
        os.makedirs(directory_relative)            
        warnings.warn (message = 'Created directory:' + directory_relative)
        
    filepath = os.path.join(directory_relative, name + ".pickle")     
    filename = open(filepath,"wb")
    
    try:    
        pickle.dump(myobject , filename)
        filename.close()
        return(True)
    except:
        return(False)

def load(directory, name):
    
    directory_relative = os.path.relpath(directory)
    
    filepath = os.path.join(directory_relative, name + '.pickle')  
    
    try:
        
        filename = open(filepath,"rb")
        
        if filename:
            obj = pickle.load(filename)
            filename.close()

    except:
        
        warnings.warn(message='No cache found')
        obj = {}
        
    return(obj)
