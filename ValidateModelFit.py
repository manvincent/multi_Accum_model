#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Module: ValideModelFit.py
Author: Vincent Man, v.man@mail.utoronto.ca
Date: 2018 05 24


Description: Generates a sample from the best fitting model parameters and compares against experimental data 
"""

import os
# Specify home directory
homeDir = '<Specify home directory>'
if not os.path.exists(homeDir):
    raise OSError("Specified home directory doesn't exist!")
else:
    os.chdir(homeDir)

# Import modules 
import numpy as np
import pandas as pd   
from Utils import *
from DefineModel import * 



# Specify input data
dataDir = '<Specify where your data are>'
data = pd.read_csv(dataDir + os.sep + 'RF2data.csv')


# Specify output directory for fit parameters
outDir = homeDir + os.sep + 'Output' + os.sep + 'fitData_Params'
if not os.path.exists(outDir):
    os.mkdir(outDir)    
    
# Specify model parameters 
modelParams = load_obj(outDir + os.sep + 'fitParams_RF2Group.pkl')

def validateFit(data, outDir, exportData, savePlot, outSample=True, sampleHalf='trans', maxRT=500, numBins=200, numSims=10000):      
    # Specify experimental parameters (number of conditions)
    numCues = len(np.unique(data.stim))
    numCtxt = len(np.unique(data.context))
    numAcc = len(np.unique(data.Response_Type))
    # Load in and format experimental data (to match num trials)
    if outSample:
        [cisData,transData] = cis_trans(data)            
        if sampleHalf == 'cis':
            currData = cisData
        elif sampleHalf == 'trans':
            currData = transData
    else: 
        currData = data         
    # Format data appropriately 
    cueIdx_data =  np.array(currData.stim.replace(to_replace=['Gain  ','Cntrl ','NoLoss'], value=[0,1,2]),dtype=float)


    # Initialize  model
    initMod = ModelType(d=modelParams.d,
                        sigma=modelParams.sigma,
                        NDT=183.75,#modelParams.NDT,
                        barrierDecay=0.002)#modelParams.barrierDecay)
    # Package general model params
    genParams = initMod.returnGen()
    
    # Intialize empty arrays for output samples 
    outSimData = pd.DataFrame()
    ctxtChoice = np.empty(numCtxt, dtype=object)
    ctxtRT = np.empty(numCtxt, dtype=object)
    ctxtCue = np.empty(numCtxt, dtype=object)
    ctxtGain = np.empty(numCtxt, dtype=object)
    ctxtLoss = np.empty(numCtxt, dtype=object)

    # Initilaize array for model predictions
    modPredict = np.empty((numCtxt, numCues, numAcc), dtype=object)  
    for ctxtNo in range(numCtxt):
        # The value of control is always the same = 0
        valueCntrl = np.unique(data.cntrlVal)[0]
        # Specify the gain and loss values of each of the four contexts
        if ctxtNo == 0:
            ctxtType = 'LM'
            valueGain = 0.5
            valueLoss = 0.5
        elif ctxtNo == 1:
            ctxtType = 'HM'
            valueGain = 1.0
            valueLoss = 1.0
        elif ctxtNo == 2:
            ctxtType = 'Prev'
            valueGain = 0.5
            valueLoss = 1.0
        elif ctxtNo == 3:
            ctxtType = 'Prom'
            valueGain = 1.0
            valueLoss = 0.5
         
        # Set up containers to hold simulated trials across all task conditions
        simSampleChoice = np.empty(numCues, dtype=object)
        simSampleRT = np.empty(numCues, dtype=object)
        simSampleHist = np.empty(numCues, dtype=object)
        cueTrials = np.empty(numCues ,dtype=object) 
        for cueIdx in range(numCues):
            # Initialize specific params, given condition and model type
            specParams = initMod.contextBiasLin_divNorm(cueBoost=modelParams.cueBoost,
                                                  divWeight=modelParams.divWeight, 
                                                  biasIntercept=modelParams.biasIntercept, 
                                                  biasSlopeGain=modelParams.biasSlopeGain, 
                                                  biasSlopeLoss=modelParams.biasSlopeLoss, 
                                                  biasSlopePrev=modelParams.biasSlopePrev, 
                                                  biasSlopeProm=modelParams.biasSlopeProm, 
                                                  cueIdx=cueIdx,
                                                  valueGain=valueGain,
                                                  valueCntrl=valueCntrl,
                                                  valueLoss=valueLoss)

            # Select the correct bias values                    
            ctxtBias = specParams[0].reshape((numCtxt, numAcc))
            specParams[0] = ctxtBias[ctxtNo,:]
        
            # Define model structure
            simModel = Model(genParams, specParams)
    
            # Run simulation to create distributions given model
            currDistribution = RunSimulation(
                simModel, numSims).simulate_distrib()
    
            # Specify the mean number of trials per experimental condition                       
            condNTrial = currData[(currData.gainVal==valueGain) & (currData.lossVal==valueLoss) & (cueIdx_data==cueIdx) & (currData.rt!=0)].rt.count() 
   
            # Draw from distributions to create trial sample
            [simSampleChoice[cueIdx], simSampleRT[cueIdx], simSampleHist[cueIdx]] = SimData(
                simModel, currDistribution).createSample(condNTrial)                     
            for a in range(numAcc):
                modPredict[ctxtNo, cueIdx, a] = simSampleHist[cueIdx][a]   

            # Store the number of sampled trials for this cue type 
            cueTrials[cueIdx] = np.repeat(cueIdx,len(simSampleChoice[cueIdx]),axis=0)
    
        # Organize model-predicted simulated data in Pandas dataframe
        ctxtChoice[ctxtNo] = np.concatenate([simSampleChoice[a] for a in range(numCues)])
        ctxtRT[ctxtNo] = np.concatenate([simSampleRT[a] for a in range(numCues)])
        ctxtCue[ctxtNo] = np.concatenate([cueTrials[a] for a in range(numCues)])
        ctxtGain[ctxtNo] = np.repeat(valueGain, len(ctxtCue[ctxtNo]),axis=0)
        ctxtLoss[ctxtNo] = np.repeat(valueLoss, len(ctxtCue[ctxtNo]),axis=0)
    
    # Organize experimental data link to model predictions    
    [allDataHist,ctxtHist,ctxtCueHist] = create_expHist(data, modPredict, outDir, outSample, sampleHalf, exportData, savePlot, numBins = (simModel.maxRT // simModel.timeStep))
    
    # Organize model-predicted simulated data in Pandas dataframe
    outSimData['Response'] = np.concatenate([ctxtChoice[c] for c in range(numCtxt)])
    outSimData['rt'] = np.concatenate([ctxtRT[c] for c in range(numCtxt)])
    outSimData['stim'] = np.concatenate([ctxtCue[c] for c in range(numCtxt)])
    outSimData['gainVal'] = np.concatenate([ctxtGain[c] for c in range(numCtxt)])
    outSimData['lossVal'] = np.concatenate([ctxtLoss[c] for c in range(numCtxt)])
    # Format output data
    outSimData = formatSimData(outSimData)
    # Output model-predicted datat to csv 
    outSimData.to_csv(outDir + os.sep + 'modPredict_simData.csv', index=False)  

    # Print model Likelihood value 
    print 'Model likelihood %.2f' %(modelParams.fitlikelihood)
    print 'Model estimates:'
    print 'd: %.3f' %(modelParams.d)
    print 'sigma: %.3f' %(modelParams.sigma)
    # print 'NDT: %.3f' %(modelParams.NDT)
    # print 'barrierDecay: %.3f' %(modelParams.barrierDecay)
    print 'cueBoost: %.3f' %(modelParams.cueBoost)
    print 'divWeight %.3f' %(modelParams.divWeight)
    print 'biasIntercept: %.3f' %(modelParams.biasIntercept)
    print 'biasSlopeGain: %.3f' %(modelParams.biasSlopeGain)        
    print 'biasSlopeLoss: %.3f' %(modelParams.biasSlopeLoss)    
    print 'biasSlopePrev: %.3f' %(modelParams.biasSlopePrev)        
    print 'biasSlopeProm: %.3f' %(modelParams.biasSlopeProm)    
    
    return 


# Run!
    
if __name__ == "__main__":
    validateFit(data, outDir, exportData=True, savePlot=True)
        

