#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 13:18:40 2018

@author: hfwittmann
"""

from keras import models


import os
import warnings

    
def model_save(mymodel:models.Sequential, directory:str, name:str):
            
    directory_relative = os.path.relpath(directory)
    
    if not os.path.exists(directory_relative):
        os.makedirs(directory_relative)            
        warnings.warn (message = 'Created directory:' + directory_relative)
        
    filepath = os.path.join(directory_relative, name)     
    
    try:
        mymodel.save(filepath)
        return(True)
    except:
        return(False)

def model_load(directory, name):
    
    directory_relative = os.path.relpath(directory)
    
    filepath = os.path.join(directory_relative, name)
    
    model = models.load_model(filepath)
    
    return model