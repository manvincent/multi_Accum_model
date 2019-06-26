#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Module: ValidatePTA_fitParms.py
Author: Vincent Man, v.man@mail.utoronto.ca
Date: 2018 05 24


Description: 
Validates the protability table algorithm (from addm; Gabriela Tavares) for 
the multi-alternative race model, given a specified set of parameters values 
(here using the best fitting parameters from the model).
Simulates using the maximum likelihood algorithm to find model prediction 
distribution, then runs PTA on the same model. Outputs validation plots 

"""


import os

# Specify home directory
homeDir = '<Specify home directory>'
if not os.path.exists(homeDir):
    raise OSError("Specified home directory doesn't exist!")
else:
    os.chdir(homeDir)

# Specify output directory for fit parameters
outDir = homeDir + os.sep + 'Output' + os.sep + 'fitData_Params'
if not os.path.exists(outDir):
    os.mkdir(outDir)    


import numpy as np
import pandas as pd   
from matplotlib import pyplot as plt
from Utils import *

# Specify input data
dataDir = '<Specify where your data are>'
data = pd.read_csv(dataDir + os.sep + 'RF1data.csv')

# Specify model parameters 
modelParams = load_obj(outDir + os.sep + 'fitParams_RF1Group.pkl')

def validatePTA_fitParams(data, outDir, savePlot=True, outSample = True, sampleHalf = 'trans', numCues = 3, numSims=10000):   
                  
    # Organize model-predicted data 
    # Initialize  model
    initMod = ModelType(d=modelParams.fitParams[0],
                        sigma=modelParams.fitParams[1],
                        NDT=modelParams.fitParams[2],
                        barrierDecay=modelParams.fitParams[3])
    # Package general model params
    genParams = initMod.returnGen()
    
    numAcc = len(np.unique(data.Choice[np.isfinite(data.Choice)]))
    modPredict = np.empty((len(np.unique(data.gainVal)), len(np.unique(data.Cue)), numAcc), dtype=object)  
    for ctxtNo in range(np.unique(data.gainVal).size):
        ctxtVal = np.unique(data.gainVal)[ctxtNo]
        # Specify the accumulator values
        valueGain = valueLoss = ctxtVal
        valueCntrl = 0.0
        
        # Set up containers to hold simulated trials across all task conditions
        simSampleChoice = np.empty(numCues, dtype=object)
        simSampleRT = np.empty(numCues, dtype=object)
        simSampleHist = np.empty(numCues, dtype=object)
        for cueIdx in range(numCues):
            if cueIdx == 0:
                cueType = 'Gain'
            elif cueIdx == 1:
                cueType = 'Cntrl'
            elif cueIdx == 2:
                cueType = 'Loss'
                
            # Initialize specific params, given condition and model type
            specParams = initMod.cueBoostZeroRace(cueBoost=modelParams.fitParams[4],
                                                  bias=modelParams.fitParams[5],                                           
                                                  cueIdx=cueIdx,
                                                  valueGain=valueGain,
                                                  valueCntrl=valueCntrl,
                                                  valueLoss=valueLoss)
            # Define model structure
            simModel = Model(genParams, specParams)
                
            # Run simulation to create distributions given model
            currDistribution = RunSimulation(
                simModel, numSims).simulate_distrib()
    


            # Run PTA algorithm
            tic = time.time()
            [currPrStates, currProbUp, currProbDown] = simModel.get_trial_likelihood()
            print 'model statestep : %.2f' %(simModel.stateStep)
            print 'wall time : %.2f' %(time.time() - tic)
                           
            # Normalise PTA probabilities so they add up to 1
            currProbUp=np.divide(currProbUp, np.nansum(currProbUp))
            # Transform PTA probabilities into normalised counts
            currCountUp = (currProbUp * sum(~np.isnan(currDistribution.simDistrChoice)))
            # Put PTA estimates on correct timescale
            timeVec = np.arange(np.round(simModel.maxRT / simModel.timeStep).astype(int)) * simModel.timeStep
            
            #print 'LHvec: ' + str(LHvec)

            # Plot Results
            binWidth =  simModel.timeStep
            plt.figure()
            greenLine = plt.bar(currDistribution.bins[:-1], currDistribution.simChoiceHist[0], color="g", width=binWidth, alpha=0.4, label='Gain')
            blueLine = plt.bar(currDistribution.bins[:-1], currDistribution.simChoiceHist[1], color="b", width=binWidth, alpha=0.4, label='Control')        
            redLine = plt.bar(currDistribution.bins[:-1], currDistribution.simChoiceHist[2], color="r", width=binWidth,  alpha=0.4, label='No Loss')
            plt.plot(timeVec,currCountUp[0], color="g", linewidth=8, linestyle="-", alpha=0.75)
            plt.plot(timeVec,currCountUp[1], color="b", linewidth=8, linestyle="--", alpha=0.75)                              
            plt.plot(timeVec,currCountUp[2], color="r",linewidth=8, linestyle="-.", alpha=0.75)
            plt.legend(handles=[greenLine,blueLine,redLine],loc=2)
            plt.xlabel('Reaction Time (ms)')
            plt.ylabel('Simulation Count')
            plt.rcParams["font.family"] = "Arial"
            plt.rcParams['font.size'] = 18
            if savePlot:  
                plt.savefig(outDir + os.sep + 'PTAvalidate_ctxt%.1f_' %(ctxtVal) + cueType + 'cue.png' %(ctxtVal),bbox_inches='tight')

    return

validatePTA_fitParams(data, outDir)
