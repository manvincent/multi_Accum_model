#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Module: FitData_Group.py
Author: Vincent Man, v.man@mail.utoronto.ca
Date: 2018 05 24


Description: 
"""

import os

# Specify home directory
homeDir = '<Specify home directory>'
if not os.path.exists(homeDir):
    raise OSError("Specified home directory doesn't exist!")
else:
    os.chdir(homeDir)

import numpy as np
import pandas as pd   
from matplotlib import pyplot as plt
import multiprocessing
from Utils import *
from DefineModel import *
from ComputeFit import *

# Specify input directory for exp data
dataDir = '<Specify where your data are>'
data = pd.read_csv(dataDir + os.sep + 'RF2data.csv')

# Specify output directory for fit parameters
outFitDir = homeDir + os.sep + 'Output' + os.sep + 'fitData_Params'
if not os.path.exists(outFitDir):
    os.mkdir(outFitDir)    


def fitGroup(data, numSeeds=15, outSample=True):
    # Sort data
    # Divide data for out of sample fitting/validation 
    if (outSample): 
        [cisData,transData] = cis_trans(data)
        currData = cisData
    else: 
        currData = data        
    # Check the data length (verify whether out-of-sample)
    print 'Num trials fit against: %d' %(len(currData))
      
    choice_Group = np.array(currData.Response_Type.replace(to_replace=['Gain  ','Cntrl ','NoLoss','NA   '], value=[0,1,2,np.nan]),dtype=float)
    rt_Group = np.array(currData.rt.replace(to_replace=0,value=np.nan),dtype=float)
    cueIdx_Group =  np.array(currData.stim.replace(to_replace=['Gain  ','Cntrl ','NoLoss'], value=[0,1,2]),dtype=float)
    ctxtIdx_Group = np.array(currData.context.replace(to_replace=['FF','OO','FO','OF'], value=[0,1,2,3]), dtype=float)
                   
    numCues = len(np.unique(cueIdx_Group))
    numCtxt = len(np.unique(ctxtIdx_Group))
    
     # Indicate what kind of model is being fitted  
    modelType = "linBias_ctxtDW"

    # Initialize model fitter (grid fitter) 
    initOptimizer = GridOptimizer(choice_Group, rt_Group, 
                              numCues, numCtxt, 'data', cueIdx_Group, ctxtIdx_Group)
   
    # Specify the LL improvement tolerance for parameter search
    tolerance = 0.01
    # Fit model  (using grid parameter search)
    [fitResult, fullSearchSpace] = initOptimizer.inductiveSearch(tolerance, modelType, numSeeds)
    
    # Save fitted parameters
    outFileName = 'fitParams_RF2Group'
    save_obj(fitResult, outFitDir + os.sep + outFileName) 
    
    # Save csv of full search space likelihoods
    fullSearchSpace.columns = ['d','sigma','cueBoost','divWeight','biasIntercept','biasSlopeGain','biasSlopeLoss','Likelihood']
    fullSearchSpace.drop_duplicates(inplace=True)
    fullSearchSpace.to_csv( outFitDir + os.sep + 'fullSearchSpace_LL.csv')
    
    # Print outcomes
    print 'Parameter Estimates: ' + str(fitResult.fitParams)
    print 'Likelihood: ' + str(fitResult.fitlikelihood)
    
    return
          

if __name__ == "__main__":
    fitGroup(data)      
        
