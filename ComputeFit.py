"""
Module: ComputeFit.py
Author: Vincent Man, v.man@mail.utoronto.ca
Date: 2018 05 24


Note: This modules needs to be modified if changing the model to fit
Currently coded for the m-SSM reported in the main text (exp 2)

"""
import numpy as np
import pandas as pd
from sklearn.utils.extmath import cartesian
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
import time
from DefineModel_Exp2 import *
from Utils import *

def unwrap_self(modFit, paramList, paramSet, paramIdx): 
    # print str(paramIdx)
    currParams = paramSet[paramIdx]
    return modFit.fitModel(currParams)

class GridOptimizer(object):
    def __init__(self, inputChoice, inputRT,
                 numCues, numCtxt, fitType, *args):
        """
        fitType should be specified as 'data' or 'sim'
        """

        self.inputChoice = inputChoice
        self.inputRT = inputRT
        self.numCues = numCues
        self.numCtxt = numCtxt
        self.fitType = fitType
        if fitType == 'data': 
            self.cueIdx_data = args[0]
            self.ctxtIdx_data = args[1]            
            self.prior = args[2]
            if (self.prior):
                self.priorParams = args[3]
  
    
    def inductiveSearch(self, tolerance, modelType, numSeeds): 
        # Initialize clock: 
        searchTic = time.time()    
        
        # Initialize arrays to store full set of search parameter values and LL results 
        fullParamSet = list() 
        fullLLSet = list()
        # Initialize array to store the best parameters and likelihood for each seed
        fullBestParams = np.empty(numSeeds,dtype=object)
        fullMaxLL = np.empty(numSeeds,dtype=object)
        
        # Set up initial search grid 
        boundList = paramBounds(modelType) 
        # Define range of middle values (linspace / uniform between lower and upper bounds)
        midRange = cartesian([np.linspace(boundList[p][0]+(boundList[p][1]-boundList[p][0])/3, 
                                          boundList[p][1], 3, 
                                          endpoint=False) for p in range(len(boundList))])        
        # Shuffle middle value ranges
        np.random.shuffle(midRange)
        
        # Specify first n middle range values 
        if (self.prior):
            # Set the first seed to search the prior parameters            
            midRange[0] = self.priorParams
            # Set the second seed to search the middle point
            midRange[1] = [np.divide(boundList[p][1] - boundList[p][0],2,dtype=float) for p in range(len(boundList))]     
        else: 
            # Set the first seed to always search the middle point between bounds
            midRange[0] = [np.divide(boundList[p][1] - boundList[p][0],2,dtype=float) for p in range(len(boundList))]
        
        print midRange[0]
    
        for seedNo in range(numSeeds):
            print 'Searching with Seed #%d' %(seedNo + 1)
            # Initialize clock
            seedTic = time.time() 

            # Select starting middle value for this seed
            seedParamMid = midRange[seedNo,:] 
            # Compute the (max) step size 
            stepSizeSet = [np.amax(np.ediff1d(np.insert(boundList[p],1,seedParamMid[p]))) for p in range(len(boundList))]
            # Set up parameter grid space (cartesian product)            
            paramSet = cartesian([np.insert(boundList[p],1,seedParamMid[p]) for p in range(len(boundList))])
            
            # Initialize arrays to store seed search parameter values and LL results
            seedParamSet = list() 
            seedLLSet = list()

            # Start the generational search 
            genNo = 0
            maxLL = 1e-250
            while True:       
                genTime = time.time() - seedTic
                # Append current parameters and likelihoods to full list
                seedParamSet.append(paramSet)            

                # Print generation number and highest likelihood so far
                print 'Evaluating Generation %d' %(genNo) + ', current max LL %.3f. ' %(maxLL) + 'Elapsed time: %.2f' %(genTime)            
                # Find likelihood of data given model for all parameters in this generation
                parallelResults = Parallel(n_jobs=num_cores)(delayed(unwrap_self)(self, paramList, paramSet, paramIdx) 
                                for (paramList,paramIdx) in zip([self] * len(paramSet), 
                                                          range(len(paramSet))))

                # Store likelihoods from this generation            
                fitLL = np.array(parallelResults) 
                # Append likelihoods from this generation to full list
                seedLLSet.append(fitLL)
                
                # Extract best fits        
                genMaxIdx = np.argmax(fitLL)            
                genMaxLL = fitLL[genMaxIdx]

                if (genMaxLL >= maxLL) and (np.abs(np.divide((genMaxLL - maxLL),maxLL,dtype=float)) < tolerance):                      
                    fitTime = time.time() - seedTic
                    print 'Seed %d finished at Gen %d' %(seedNo + 1, genNo)
                    print 'Seed minimum found: %.3f' %(genMaxLL)
                    print 'Seed Fit time: %.2f' %(fitTime)
                    break
               
                # Generate a new set of parameter values at a more refined level   
                newStepSize = np.divide(stepSizeSet,2,dtype=float)         
                [paramSet, stepSizeSet] = self.adaptiveParamSet(paramSet[genMaxIdx], newStepSize, boundList)
                
                # Continue the next generation of parameters 
                maxLL = genMaxLL                                         
                genNo += 1
            
            # Store best fits for this seed     
            fullBestParams[seedNo] = paramSet[genMaxIdx,:]
            fullMaxLL[seedNo] = genMaxLL
    
            # Append this seed's parameters and likelihoods
            fullParamSet.append(seedParamSet)
            fullLLSet.append(seedLLSet)
        
        # Print search time 
        searchTime = time.time() - searchTic    
        print 'Search finished in %.2f seconds' %(searchTime)

        # Store all searched parameters and resulting fits across seeds
        fullSearchSpace = pd.DataFrame(np.concatenate(np.concatenate(fullParamSet)))
        fullSearchSpace['Likelihood'] = np.concatenate(np.concatenate(fullLLSet))        
        
        # Identify the global maximum across all seed's best estiamtes
        globalMaxIdx = np.argmax(fullMaxLL)
        print 'Global maximum found: %.3f' %(fullMaxLL[globalMaxIdx])
        fitResult = fitParamContain(fullBestParams[globalMaxIdx], fullMaxLL[globalMaxIdx]).multiBiasLin_divNorm_Params()
               
        return fitResult, fullSearchSpace
                          
    def adaptiveParamSet(self, currParamSet, newStepSize, boundList):
        # Define new upper and lower bounds for this genereation
        newUpperParamBound = np.add(currParamSet, newStepSize)
        newLowerParamBound = np.subtract(currParamSet, newStepSize)
        
        # Fix new parameter values outside of bounds to bound limits
        for p in range(len(boundList)):
            newUpperParamBound[p] = boundList[p][1] if newUpperParamBound[p]>boundList[p][1] else newUpperParamBound[p]
            newLowerParamBound[p] = boundList[p][0] if newLowerParamBound[p]<boundList[p][0] else newLowerParamBound[p]
        # Set up new cross-product across parameter values 
        newParamSet = cartesian(zip(newLowerParamBound , currParamSet, newUpperParamBound))
        return(newParamSet, newStepSize)
        
    def fitModel(self, currParams):
        # Initialize  model structure
        initMod = ModelType(d=0.001,#currParams[0],
                            sigma=currParams[1],
                            NDT= 183.75,#currParams[2],
                            barrierDecay= 0.002) #currParams[3])
        # Package general model params
        genParams = initMod.returnGen()
                            
        # Initialize object to store model probabiliites 
        currProbUp = np.empty((self.numCtxt, self.numCues), dtype=object)
        
        for ctxtNo in range(self.numCtxt):
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
        
        
            for currCue in range(self.numCues):         
                specParams = initMod.multiBiasLin_divNorm(cueBoost=currParams[2],
                                                  divWeight=currParams[3],                                                      
                                                  biasIntercept=currParams[4],  
                                                  biasSlopeGain=currParams[5], 
                                                  biasSlopeLoss=currParams[6], 
                                                  cueIdx=currCue,
                                                  valueGain=currValueGain,
                                                  valueCntrl=currValueCntrl,
                                                  valueLoss=currValueLoss)
                # Select the correct mean accumulator values
#                    numAcc = genParams[6]
#                    ctxtMeans = specParams[1].reshape((self.numCtxt, numAcc))
#                    specParams[1] = ctxtMeans[ctxtNo,:]
                
                # Define model structure
                simModel = Model(genParams, specParams)
    
                # Compute likelihood of p(specific trial | parameters, model)
                # Run PTA algorithm
                [_, currProbUp[int(ctxtNo), int(currCue)], _] = simModel.get_trial_likelihood()
    
        # Find likelihood of data given model
        dataLikelihood = Likelihood(
            simModel, self.inputChoice, self.inputRT, self.fitType).LH(self.cueIdx_data, currProbUp, self.ctxtIdx_data)

       # Aggregate across entire data set (all numCues) and convert to log likelihood
        logLikelihood = np.sum(np.log(dataLikelihood))
        return logLikelihood



class Likelihood(object):

    def __init__(self, model, inputChoice, inputRT, fitType):
        self.model = model
        self.inputChoice = inputChoice
        self.inputRT = inputRT
        self.fitType = fitType

    def LH(self, cueIdx, currProbUp, *args):

        ctxtIdx = args[0]

        # Compute total number of (surviving) trials
        totalTrials = np.sum(np.isfinite(self.inputChoice))
        # Create mask of only surviving (non-nan) trials
        trialNAmask = np.isfinite(self.inputChoice)
       
        # Get vector of all stim cue and context indices with non-nan responses
        ctxtIndices = ctxtIdx[trialNAmask].astype(int)
        cueIndices = cueIdx[trialNAmask].astype(int)

        # Get vector of all non-nan choice
        choiceIndices = self.inputChoice[trialNAmask].astype(int)
        # Get vector of respective PTA RT indices (bins) all non-nan empirical RTs
        nonNanRT = self.inputRT[trialNAmask]
        timeVec = np.arange(np.round(self.model.maxRT / self.model.timeStep).astype(int)) * self.model.timeStep
        rtIndices = [np.abs(np.floor(timeVec - nonNanRT[i])).argmin()for i in range(len(nonNanRT))]
        
        # Create vector of likelihood values by extracting from respective
        # indices in PTA upper crossing            
        LHvec = np.empty(totalTrials, dtype=float)
        for tr in range(totalTrials):
            LHvec[tr] = currProbUp[ctxtIndices[tr], cueIndices[tr]][choiceIndices[tr]][rtIndices[tr]]

        # Convert all zero probabilities to very small value
        LHvec[np.isnan(LHvec)] = 1e-250
        LHvec[LHvec < 1e-250] = 1e-250
            
        return LHvec        
