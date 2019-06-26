#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Module: exploreData.py
Author: Vincent Man, v.man@mail.utoronto.ca
Date: 2018 05 24

Description: 
"""
import os
import numpy as np
from matplotlib import pyplot as plt
from utilities import *


def create_expHist(data, modPredict, outDir, outSample, sampleHalf, exportData, savePlot, numBins, computeHist = True): 
    # Divide data for out of sample fitting/validation 
    if outSample:
        [cisData,transData] = cis_trans(data)            
        if sampleHalf == 'cis':
            currData = cisData
        elif sampleHalf == 'trans':
            currData = transData
    else: 
        currData = data         

    if exportData:
        currData.to_csv(outDir + os.sep + sampleHalf + '_expData.csv', index=False)  

          
    # Plot/save histograms of the raw data
    if computeHist:        
        [allDataHist,ctxtHist,ctxtCueHist] = plot_RawData(currData, modPredict, outDir, savePlot, numBins) 
        
    return (allDataHist,ctxtHist,ctxtCueHist)


def plot_RawData(data, modPredict, outDir, savePlot, numBins, maxRT=500):
    # Remove missing values from input data 
    expData = data.loc[data.rt != 0]

    # Plot histograms of raw RTs across all context conditions         
    choice_Group = np.array(expData.Response_Type.replace(to_replace=['Gain  ','Cntrl ','NoLoss','NA   '], value=[0,1,2,np.nan]),dtype=float)
    rt_Group = np.array(expData.rt.replace(to_replace=0,value=np.nan),dtype=float)
    cueIdx_Group =  np.array(expData.stim.replace(to_replace=['Gain  ','Cntrl ','NoLoss'], value=[0,1,2]),dtype=float)
    ctxtIdx_Group = np.array(expData.context.replace(to_replace=['FF','OO','FO','OF'], value=[0,1,2,3]), dtype=float)
                 
    numCues = len(np.unique(cueIdx_Group))
    numCtxt = len(np.unique(ctxtIdx_Group))
    numAcc = len(np.unique(choice_Group))


    # Compute histograms collapsing across cues (just conditional on choice)
    allDataHist = np.empty(numAcc, dtype=object)    
    for a in range(numAcc):
        [allDataHist[a],bins] =  np.histogram(rt_Group[choice_Group == a], range = (0,maxRT), bins = numBins)  
    
    # Plot across cues       
    binWidth = maxRT // numBins
    plt.figure()             
    plt.xlim(0, maxRT)
    greenLine = plt.bar(bins[:-1],allDataHist[0], color='g', alpha=0.4, width=binWidth, label='Gain Resp')
    blueLine = plt.bar(bins[:-1],allDataHist[1], color='b', alpha=0.4, width=binWidth, label='Neutral Resp')
    redLine = plt.bar(bins[:-1],allDataHist[2], color='r', alpha=0.4, width=binWidth, label='No Loss Resp')
    plt.legend(handles=[greenLine,blueLine,redLine],loc=2)
    plt.xlabel('Reaction Time (msec)')
    plt.ylabel('Trial Count')
    if savePlot:       
        plt.savefig(outDir + os.sep + 'allCues_Hist_AcrossCtxt.png',bbox_inches='tight')
        
    ctxtHist = np.empty((len(np.unique(ctxtIdx_Group)),numAcc), dtype=object)
    ctxtCueHist = np.empty((len(np.unique(ctxtIdx_Group)), len(np.unique(cueIdx_Group)), numAcc), dtype=object)
    for ctxtNo in np.unique(ctxtIdx_Group).astype(int):        
        if ctxtNo == 0:
            ctxtType = 'LM'
            ctxtGainVal = 0.5
            ctxtLossVal = 0.5
        elif ctxtNo == 1:
            ctxtType = 'HM'
            ctxtGainVal = 1.0
            ctxtLossVal = 1.0
        elif ctxtNo == 2:
            ctxtType = 'Prev'
            ctxtGainVal = 0.5
            ctxtLossVal = 1.0
        elif ctxtNo == 3:
            ctxtType = 'Prom'
            ctxtGainVal = 1.0
            ctxtLossVal = 0.5
            
           
        # Specify context-specific data points
        expData_ctxt = expData.loc[np.logical_and(expData.gainVal == ctxtGainVal,expData.lossVal == ctxtLossVal)]
    
        # Parse data
        choice_Group_ctxt = np.array(expData_ctxt.Response_Type.replace(to_replace=['Gain  ','Cntrl ','NoLoss',], value=[0,1,2]),dtype=float)        
        rt_Group_ctxt = np.array(expData_ctxt.rt,dtype=float)
        cueIdx_Group_ctxt =  np.array(expData_ctxt.stim.replace(to_replace=['Gain  ','Cntrl ','NoLoss'], value=[0,1,2]),dtype=float)    
                        
         # Create histograms from the data 
        for a in range(numAcc):
            [ctxtHist[ctxtNo,a],bins] = np.histogram(rt_Group_ctxt[choice_Group_ctxt == a], range = (0,maxRT), bins = numBins)

              
        plt.figure()
        plt.xlim(0, maxRT)
        greenLine = plt.bar(bins[:-1],ctxtHist[ctxtNo,0], color='g', alpha=0.4, width=binWidth, label='Gain Resp')
        blueLine = plt.bar(bins[:-1],ctxtHist[ctxtNo,1], color='b', alpha=0.4, width=binWidth, label='Neutral Resp')
        redLine = plt.bar(bins[:-1],ctxtHist[ctxtNo,2], color='r', alpha=0.4, width=binWidth, label='No Loss Resp')
        plt.legend(handles=[greenLine,blueLine,redLine],loc=2)
        plt.xlabel('Reaction Time (msec)')
        plt.ylabel('Trial Count')
        if savePlot:
            plt.savefig(outDir + os.sep + '%s_allCues_Hist.png' %(ctxtType), bbox_inches='tight')
        

        # Compute separate histograms and plots for each cue type           
        for cueIdx in np.unique(cueIdx_Group).astype(int):
            
            if cueIdx == 0:
                cueType = 'Gain'
            elif cueIdx == 1:
                cueType = 'Cntrl'
            elif cueIdx == 2:
                cueType = 'Loss'
                                          
            cueType_rt = rt_Group_ctxt[cueIdx_Group_ctxt == cueIdx]
            cueType_choice = choice_Group_ctxt[cueIdx_Group_ctxt == cueIdx]
            
            # Create histograms for each choice type, for each cue and each context         
            for a in range(numAcc):
                [ctxtCueHist[ctxtNo,cueIdx,a],bins] =  np.histogram(cueType_rt[cueType_choice == a], range = (0,maxRT), bins = numBins)            

                    
            plt.figure()             
            plt.xlim(0, maxRT)
            greenLine = plt.bar(bins[:-1],ctxtCueHist[ctxtNo,cueIdx,0], color='g', alpha=0.4, width=binWidth, label='Gain Resp')
            blueLine = plt.bar(bins[:-1],ctxtCueHist[ctxtNo,cueIdx,1], color='b', alpha=0.4, width=binWidth, label='Neutral Resp')
            redLine = plt.bar(bins[:-1],ctxtCueHist[ctxtNo,cueIdx,2], color='r', alpha=0.4, width=binWidth, label='No Loss Resp')
            plt.legend(handles=[greenLine,blueLine,redLine],loc=2)
            plt.xlabel('Reaction Time (msec)')
            plt.ylabel('Trial Count')
            plt.plot(bins[:-1], modPredict[ctxtNo, cueIdx, 0], color="g", linewidth = 2)
            plt.plot(bins[:-1], modPredict[ctxtNo, cueIdx, 1], color="b", linewidth = 2)
            plt.plot(bins[:-1], modPredict[ctxtNo, cueIdx, 2], color="r", linewidth = 2)
            if savePlot:  
                plt.savefig(outDir + os.sep + '%s_' %(ctxtType) + cueType + 'cue_Hist_AcrossCtxt.png',bbox_inches='tight')

    return (allDataHist,ctxtHist,ctxtCueHist)   

