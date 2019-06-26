#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Module: ExtractFits_Subject.py
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
from Utils import *
from DefineModel import *
from ComputeFit import *


# Specify input directory for exp data
dataDir = '<Specify where your data are>'
data = pd.read_csv(dataDir + os.sep + 'RF2data.csv')

# Specify output directory 
outDir = homeDir + os.sep + 'Output' 
# Output for fit parameters
outFitDir = outDir + os.sep + 'fitData_Params_Subject'
if not os.path.exists(outFitDir):
    os.mkdir(outFitDir)    
    

        
def extractSubjectFits(data):        
    numSubs = len(np.unique(data.subj_idx))
    # Initilaize pandas dataframe to store subject parameters
    subParamDF = pd.DataFrame(np.nan, 
                              index=range(numSubs), 
                              columns=['subID','d','sigma','cueBoost','divWeight','biasIntercept','biasSlopeGain','biasSlopeLoss','Likelihood'])

    for s in range(numSubs):
        subID = np.unique(data.subj_idx)[s]
            
        # Load fitted parameters
        outFileName = str(subID) + '_fitParams_RF2'
        subParams = load_obj(outFitDir + os.sep + outFileName + '.pkl') 
        
        # Store respective parameters values in pandas dataframe
        subParamDF['subID'][s] = subID                
        subParamDF['d'][s] = subParams.d
        subParamDF['sigma'][s] = subParams.sigma
        subParamDF['cueBoost'][s] = subParams.cueBoost
        subParamDF['divWeight'][s] = subParams.divWeight
        subParamDF['biasIntercept'][s] = subParams.biasIntercept
        subParamDF['biasSlopeGain'][s] = subParams.biasSlopeGain
        subParamDF['biasSlopeLoss'][s] = subParams.biasSlopeLoss
        subParamDF['Likelihood'][s] = subParams.fitlikelihood                     
    # Output csv of all subject's fitted parameter values
    subParamDF.to_csv(outDir + os.sep + 'allSub_modFits.csv')                
    return

       
def modelImplications(data):        
    numSubs = len(np.unique(data.subj_idx))
    # Load csv subject's fitted parameter values 
    subParamDF = pd.read_csv(outDir + os.sep + 'allSub_modFits.csv')
    # Merge model parameters into empirical data
    modData = pd.merge(data,subParamDF,left_on='subj_idx',right_on='subID')
    # Check merge
    if (np.max(modData.subID - modData.subj_idx) == 0):
        print 'Successful dataframe merge without errors'
    else: 
        raise ValueError("Error: dataframes not merged successfully!")
    
    
    appended_data = []
    for s in range(numSubs):
        subID = np.unique(data.subj_idx)[s]
        print 'Estimating for subject %d, ID: %d' %(s, subID)
            
        # Load fitted parameters
        outFileName = str(subID) + '_fitParams_RF2'
        subParams = load_obj(outFitDir + os.sep + outFileName + '.pkl') 
                
        # Compute trialwise accumulator probabilities
        # Conditionalize on (contex), cue type, response, and rt 
        # Returned value is the PTA value of the chosen accumulator given subject's fits, and the task condition
         # Initialize  model structure
        initMod = ModelType(d=subParams.d,
                            sigma=subParams.sigma,
                            NDT= 183.75,
                            barrierDecay= 0.002)
        # Package general model params
        genParams = initMod.returnGen()
                            
        # Initialize object to store model probabiliites 
        numCtxt = len(np.unique(modData.context))
        numCues = len(np.unique(modData.stim))

        currProbUp = np.empty((numCtxt,numCues), dtype=object)
        
        for ctxtNo in range(numCtxt):
            # The value of control is always the same = 0
            currValueCntrl = 0
            # Specify the gain and loss values of each of the four contexts
            if ctxtNo == 0:
                ctxtType = 'LM'
                currValueGain = 0.5
                currValueLoss = 0.5
            elif ctxtNo == 1:
                ctxtType = 'HM'
                currValueGain = 1.0
                currValueLoss = 1.0
            elif ctxtNo == 2:
                ctxtType = 'Prev'
                currValueGain = 0.5
                currValueLoss = 1.0
            elif ctxtNo == 3:
                ctxtType = 'Prom'
                currValueGain = 1.0
                currValueLoss = 0.5
        
        
            for currCue in range(numCues):         
                specParams = initMod.multiBiasLin_divNorm(cueBoost=subParams.cueBoost,
                                                  divWeight=subParams.divWeight,                                                      
                                                  biasIntercept=subParams.biasIntercept,  
                                                  biasSlopeGain=subParams.biasSlopeGain, 
                                                  biasSlopeLoss=subParams.biasSlopeLoss, 
                                                  cueIdx=currCue,
                                                  valueGain=currValueGain,
                                                  valueCntrl=currValueCntrl,
                                                  valueLoss=currValueLoss)
                
                # Define model structure
                simModel = Model(genParams, specParams)
    
                # Compute likelihood of p(specific trial | parameters, model)
                # Run PTA algorithm
                [_, currProbUp[int(ctxtNo), int(currCue)], _] = simModel.get_trial_likelihood()
                


        # Delineate the subject's data                       
        subData = modData.loc[np.where(modData.subID == subID)]
        subData['probAcc'] = np.nan
        
        # Vectorize task conditions and responses in sub data
        choice_sub = np.array(subData.Response_Type.replace(to_replace=['Gain  ','Cntrl ','NoLoss','NA   '], value=[0,1,2,np.nan]),dtype=float)
        rt_sub = np.array(subData.rt.replace(to_replace=0,value=np.nan),dtype=float)
        cueIdx_sub =  np.array(subData.stim.replace(to_replace=['Gain  ','Cntrl ','NoLoss'], value=[0,1,2]),dtype=float)
        ctxtIdx_sub = np.array(subData.context.replace(to_replace=['FF','OO','FO','OF'], value=[0,1,2,3]), dtype=float)

        # Compute PTA values given the RT and the subject's data                      
        subData['probAcc'] =  Likelihood(simModel, cueIdx_sub, rt_sub, 'data').LH(cueIdx_sub, currProbUp, ctxtIdx_sub)
        appended_data.append(subData)

    # Concatenate subject data frames into long format
    modData_estimates = pd.concat(appended_data, axis=0)
    
    # Compute individual differences in accumulator means        
    modData['normFactor'] = modData.divWeight*modData.gainVal + modData.divWeight*modData.cntrlVal + modData.divWeight*modData.lossVal
    modData['meanAcc'] = np.nan
    # Specify RDV update mean equations for each accumulator, conditional on (context), cued option, response
    meanGain_Cued = modData.d * (modData.cueBoost + modData.gainVal / modData.normFactor)
    meanGain_NotCued = modData.d * (1 + modData.gainVal / modData.normFactor)
    meanCntrl_Cued = modData.d * (modData.cueBoost  + modData.cntrlVal / modData.normFactor)
    meanCntrl_NotCued = modData.d * (1  + modData.cntrlVal / modData.normFactor)
    meanLoss_Cued = modData.d * (modData.cueBoost + modData.lossVal / modData.normFactor)
    meanLoss_NotCued  = modData.d * (1 + modData.lossVal / modData.normFactor)
    # Conditionalize on the experimental condition and assign
    modData['meanAcc'][(modData.stim=='Gain  ') & (modData.Response_Type=='Gain  ')] = meanGain_Cued
    modData['meanAcc'][(modData.stim!='Gain  ') & (modData.Response_Type=='Gain  ')] = meanGain_NotCued
    modData['meanAcc'][(modData.stim=='Cntrl ') & (modData.Response_Type=='Cntrl ')] = meanCntrl_Cued
    modData['meanAcc'][(modData.stim!='Cntrl ') & (modData.Response_Type=='Cntrl ')] = meanCntrl_NotCued
    modData['meanAcc'][(modData.stim=='NoLoss') & (modData.Response_Type=='NoLoss')] = meanLoss_Cued
    modData['meanAcc'][(modData.stim!='NoLoss') & (modData.Response_Type=='NoLoss')] = meanLoss_NotCued
    
    # Add model estimates as column
    modData['probAcc'] = modData_estimates.probAcc
    
    # Output to csv
    modData.to_csv(outDir + os.sep + 'allSub_modEstimates_Trial.csv')
     
    return


if __name__ == "__main__":
    extractSubjectFits(data)
    modelImplications(data)


     
