#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Module: FitData_Subject.py
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
from ComputeFit_stats import *


# Specify input directory for exp data
dataDir = '<Specify where your data are>'
data = pd.read_csv(dataDir + os.sep + 'RF2data.csv')

# Specify output directory 
outDir = homeDir + os.sep + 'Output' 
# Output for fit parameters
outFitDir = outDir + os.sep + 'fitData_Params_Subject'
if not os.path.exists(outFitDir):
    os.mkdir(outFitDir)    
# Output for search space
outSearchDir = outDir + os.sep + 'searchSpace_Subject'
if not os.path.exists(outSearchDir):
    os.mkdir(outSearchDir)    
    
# Load in group fit parameters
groupModParams = load_obj(outDir + os.sep + 'fitData_Params/fitParams_RF2Group.pkl')

        
def fitSubject(data, numSeeds=1, outSample=False, prior=True):    
    numSubs = len(np.unique(data.subj_idx))
    for sub in [0]:#range(numSubs):
        subID = np.unique(data.subj_idx)[sub]
        
        # Delinate subject data
        subData = pd.DataFrame()
        subData = data.loc[data.subj_idx == subID]
        
        # Sort data
        # Divide data for out of sample fitting/validation 
        currData = pd.DataFrame()
        if (outSample): 
            [cisData,transData] = cis_trans(subData)
            currData = cisData
        else: 
            currData = subData         
        # Check the data length (verify whether out-of-sample)
        print 'Fitting model for subject: %d' %(subID)
        print 'Num trials fit against: %d' %(len(currData))
          
        choice_Group = np.array(currData.Response_Type.replace(to_replace=['Gain  ','Cntrl ','NoLoss','NA   '], value=[0,1,2,np.nan]),dtype=float)
        rt_Group = np.array(currData.rt.replace(to_replace=0,value=np.nan),dtype=float)
        cueIdx_Group =  np.array(currData.stim.replace(to_replace=['Gain  ','Cntrl ','NoLoss'], value=[0,1,2]),dtype=float)
        ctxtIdx_Group = np.array(currData.context.replace(to_replace=['FF','OO','FO','OF'], value=[0,1,2,3]), dtype=float)
                           
        numCues = len(np.unique(cueIdx_Group))
        numCtxt = len(np.unique(ctxtIdx_Group))

                 # Indicate what kind of model is being fitted  
        modelType = "multiBiasLin_divNorm"

        # Initialize model fitter (grid fitter) 
        initOptimizer = GridOptimizer(choice_Group, rt_Group, 
                              numCues, numCtxt,  'data', 
                              cueIdx_Group, ctxtIdx_Group, prior, groupModParams.fitParams)
        # Specify the LL improvement tolerance for parameter search
        tolerance = 0.01
        # Fit model  (using grid parameter search)
        [fitResult, fullSearchSpace] = initOptimizer.inductiveSearch(tolerance, modelType, numSeeds)
        
        # Save fitted parameters
        outFileName = str(subID) + '_fitParams_RF2'
        save_obj(fitResult, outFitDir + os.sep + outFileName) 
        
        # Print outcomes
        print 'Finished fitting for subject %d, maxLL: %.2f' %(subID, fitResult.fitlikelihood)        
        
        # Save csv of full search space likelihoods
        fullSearchSpace.columns = ['d','sigma','cueBoost','divWeight','biasIntercept','biasSlopeGain','biasSlopeLoss','Likelihood']
        fullSearchSpace.drop_duplicates(inplace=True)
        fullSearchSpace.to_csv(outSearchDir + os.sep + str(subID) + '_fullSearchSpace_LL.csv')            
    return

if __name__ == "__main__":
    fitSubject(data)


     
