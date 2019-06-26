"""
Module: DefineModel_Exp2.py
Author: Vincent Man, v.man@mail.utoronto.ca
Date: 2018 05 24


Description:

"""
import numpy as np
from sklearn.utils.extmath import cartesian
from itertools import compress



def paramBounds(modelType):
    # Define parameter lower and upper bounds 
    d_bounds = (0.001, 0.05)
    sigma_bounds = (0.001, 0.05)
    NDT_bounds = (0.0, 250.0)
    barrierDecay_bounds = (0.0001, 0.05)
    cueBoost_bounds = (1.0, 40.0)
    divWeight_bounds = (0.001,1.0)
    dwIntercept_bounds = (0.001,0.1)
    dwSlopeGain_bounds = dwSlopeLoss_bounds = dwSlopePrev_bounds = dwSlopeProm_bounds =(-0.03,0.03)
    bias_bounds = biasGain_bounds = biasCntrl_bounds = biasLoss_bounds = (0.0, 0.9)    
    biasIntercept_bounds = (0.0,0.5)
    biasSlopeGain_bounds = biasSlopeLoss_bounds = biasSlopePrev_bounds = biasSlopeProm_bounds = biasPrevLossInt_bounds = biasPromGainInt_bounds = (0.0,0.25)
    allBounds = [d_bounds, sigma_bounds,\
                NDT_bounds, barrierDecay_bounds,\
                cueBoost_bounds,\
                divWeight_bounds,\
                dwIntercept_bounds, dwSlopeGain_bounds, dwSlopeLoss_bounds,
                biasGain_bounds, biasCntrl_bounds, biasLoss_bounds,\
                biasIntercept_bounds, biasSlopeGain_bounds, biasSlopeLoss_bounds,\
                biasSlopePrev_bounds, biasSlopeProm_bounds,
                dwSlopePrev_bounds, dwSlopeProm_bounds]
    
    # Subset parameter list depending on model type 
    paramMask = np.ones(len(allBounds), dtype=np.bool)
        
    if modelType == "multiBiasLin":
        paramMask[5:12] = paramMask[15:19] = False
    elif modelType == 'multiBias_divNorm':
        paramMask[6:9] = paramMask[12:19] = False
    elif modelType == 'multiBiasLin_divNorm':
        paramMask[6:12] =  paramMask[15:19] = False
    elif modelType == 'contextBiasLin_divNorm':
        paramMask[6:12] = paramMask[17:19] = False
    elif modelType == 'linBias_linDW':
        paramMask[5] = paramMask[9:12] = paramMask[15:19] = False
    elif modelType == 'linBias_linDW_Acc':
        paramMask[5] = paramMask[9:12] = paramMask[15:19] = False        
    elif modelType == 'linBias_ctxtDW':
        paramMask[5] = paramMask[7:12] = paramMask[15:17] = False        
    else:
        raise ValueError("Error: Model Type not recognized.")        
    
    # Fix NDT and barrierDecay values
    paramMask[2] = paramMask[3] = False
    # Compress parameter bounds based on model type
    currBounds = list(compress(allBounds, paramMask))    
    return currBounds
 

class ModelType(object):

    def __init__(self,
                 d, sigma, NDT, barrierDecay,
                 barrierStart=1, lowerBound=0, numAcc=3):
        """
        Args are parameters that generalize across all instances (trials) of model
        Args:
            d: float, speed of signal accumulation
            sigma: float, sd. of the error normal distribution
            bias: float, initial value of the accumulator(s) [in units of RDV]
            NDT: float, intial NDT where only noise drives accumulators [in ms]            
            barrierDecay: float, rate of barrier decay
            barrierStart: pos int, starting value of threshold
            lowerBound: int, lower boundary value
            numAcc: int, specify number of accumulators            
            
        """

        if barrierStart <= 0:
            raise ValueError("Error: barrier parameter must larger than zero.")

        self.d = d
        self.sigma = sigma
        self.NDT = NDT
        self.barrierDecay = barrierDecay
        self.barrierStart = barrierStart
        self.lowerBound = lowerBound
        self.numAcc = numAcc

    def returnGen(self):
        # Pack up and return general parameters
        genParams = np.empty(7, dtype=object)
        genParams[0] = self.d
        genParams[1] = self.sigma
        genParams[2] = self.NDT
        genParams[3] = self.barrierDecay
        genParams[4] = self.barrierStart
        genParams[5] = self.lowerBound
        genParams[6] = self.numAcc
        return genParams
    
    def multiBiasLin_divNorm(self, biasIntercept, biasSlopeGain, biasSlopeLoss, cueBoost, divWeight, cueIdx, valueGain, valueCntrl, valueLoss):
        """
        Args are parameters that change between instances (trials) of model
        Args: 
            cueIdx, int, specify which accumulator is cued on this sim trial
                (denoted by RDVaccum index)
            valueGain: value of Gain alternative for simulation
            valueCntrl: value of Cntrl alternative for simulation
            valueLoss: value of Loss alternative for simulation
                (enter as positive value)
        """
        # Specify number of parameters
        self.numParams = 9

        # Model the starting value with a linear model
        # Dummy code the three cue conditions (gain, cntrl, noloss)
        # The cntrl cue condition is placed in the intercept, 
        # so only two slopes / dummy arrays need be specified        
        biasArray = biasIntercept + biasSlopeGain * np.array([1,0,0]) + biasSlopeLoss * np.array([0,0,1])        
        if any(biasArray > self.barrierStart):
            raise ValueError("Error: bias parameter must be smaller than "
                             "barrier parameter.")


        # Specify which accumulator is cued (gets boost) on this trial
        cueCon = np.ones(3, dtype=float)
        cueCon[cueIdx] = cueBoost

        # Specify the denominator in divisive normalisation 
        normFactor = np.sum([divWeight*valueGain, divWeight*valueCntrl, divWeight*valueLoss],dtype=float)
        # Specify RDV update mean equations for each accumulator            
        meanGain = self.d * (cueCon[0] + np.divide(valueGain,normFactor,dtype=float))
        meanCntrl = self.d * (cueCon[1] + np.divide(valueCntrl,normFactor,dtype=float))
        meanLoss = self.d * (cueCon[2] + np.divide(valueLoss,normFactor,dtype=float))
        # Assign to array
        meanArray = np.array([meanGain, meanCntrl, meanLoss], dtype=float)

        # Package the parameters
        # Variant-specific parameters
        specParams = np.empty(6, dtype=object)
        specParams[0] = biasArray
        specParams[1] = meanArray
        specParams[2] = cueIdx
        specParams[3] = valueGain
        specParams[4] = valueCntrl
        specParams[5] = valueLoss
        return specParams            

    def linBias_linDW(self, biasIntercept, biasSlopeGain, biasSlopeLoss, cueBoost, dwIntercept, dwSlopeGain, dwSlopeLoss, cueIdx, valueGain, valueCntrl, valueLoss):
        # Specify number of parameters
        self.numParams = 11

        # Model the starting value with a linear model
        # Dummy code the three cue conditions (gain, cntrl, noloss)
        # The cntrl cue condition is placed in the intercept, 
        # so only two slopes / dummy arrays need be specified        
        biasArray = biasIntercept + biasSlopeGain * np.array([1,0,0]) + biasSlopeLoss * np.array([0,0,1])        
        if any(biasArray > self.barrierStart):
            raise ValueError("Error: bias parameter must be smaller than "
                             "barrier parameter.")


        # Specify which accumulator is cued (gets boost) on this trial
        cueCon = np.ones(3, dtype=float)
        cueCon[cueIdx] = cueBoost

        # Specify the denominator in divisive normalisation 
        divWeight = dwIntercept + dwSlopeGain * np.array([1,0,0]) + dwSlopeLoss * np.array([0,0,1])

        normFactor = np.sum([divWeight[0]*valueGain, divWeight[1]*valueCntrl, divWeight[2]*valueLoss],dtype=float)
        # Specify RDV update mean equations for each accumulator            
        meanGain = self.d * (cueCon[0] + np.divide(valueGain,normFactor,dtype=float))
        meanCntrl = self.d * (cueCon[1] + np.divide(valueCntrl,normFactor,dtype=float))
        meanLoss = self.d * (cueCon[2] + np.divide(valueLoss,normFactor,dtype=float))
        # Assign to array
        meanArray = np.array([meanGain, meanCntrl, meanLoss], dtype=float)

        # Package the parameters
        # Variant-specific parameters
        specParams = np.empty(6, dtype=object)
        specParams[0] = biasArray
        specParams[1] = meanArray
        specParams[2] = cueIdx
        specParams[3] = valueGain
        specParams[4] = valueCntrl
        specParams[5] = valueLoss
        return specParams   

    def linBias_linDW_Acc(self, biasIntercept, biasSlopeGain, biasSlopeLoss, cueBoost, dwIntercept, dwSlopeGain, dwSlopeLoss, cueIdx, valueGain, valueCntrl, valueLoss):
        # Specify number of parameters
        self.numParams = 11

        # Model the starting value with a linear model
        # Dummy code the three cue conditions (gain, cntrl, noloss)
        # The cntrl cue condition is placed in the intercept, 
        # so only two slopes / dummy arrays need be specified        
        biasArray = biasIntercept + biasSlopeGain * np.array([1,0,0]) + biasSlopeLoss * np.array([0,0,1])        
        if any(biasArray > self.barrierStart):
            raise ValueError("Error: bias parameter must be smaller than "
                             "barrier parameter.")


        # Specify which accumulator is cued (gets boost) on this trial
        cueCon = np.ones(3, dtype=float)
        cueCon[cueIdx] = cueBoost

        # Specify the denominator in divisive normalisation 
        divWeight = dwIntercept + dwSlopeGain * np.array([1,0,0]) + dwSlopeLoss * np.array([0,0,1])

        # Specify RDV update mean equations for each accumulator            
        meanGain = self.d * (cueCon[0] + np.divide(valueGain,np.sum([divWeight[0]*valueGain, divWeight[0]*valueCntrl, divWeight[0]*valueLoss]),dtype=float))
        meanCntrl = self.d * (cueCon[1] + np.divide(valueCntrl,np.sum([divWeight[1]*valueGain, divWeight[1]*valueCntrl, divWeight[1]*valueLoss]),dtype=float))
        meanLoss = self.d * (cueCon[2] + np.divide(valueLoss,np.sum([divWeight[2]*valueGain, divWeight[2]*valueCntrl, divWeight[2]*valueLoss]),dtype=float))
        # Assign to array
        meanArray = np.array([meanGain, meanCntrl, meanLoss], dtype=float)

        # Package the parameters
        # Variant-specific parameters
        specParams = np.empty(6, dtype=object)
        specParams[0] = biasArray
        specParams[1] = meanArray
        specParams[2] = cueIdx
        specParams[3] = valueGain
        specParams[4] = valueCntrl
        specParams[5] = valueLoss
        return specParams 

    def linBias_ctxtDW(self, biasIntercept, biasSlopeGain, biasSlopeLoss, cueBoost, dwIntercept, dwSlopePrev, dwSlopeProm, cueIdx, valueGain, valueCntrl, valueLoss):
        # Specify number of parameters
        self.numParams = 11

        # Model the starting value with a linear model
        # Dummy code the three cue conditions (gain, cntrl, noloss)
        # The cntrl cue condition is placed in the intercept, 
        # so only two slopes / dummy arrays need be specified        
        biasArray = biasIntercept + biasSlopeGain * np.array([1,0,0]) + biasSlopeLoss * np.array([0,0,1])        
        if any(biasArray > self.barrierStart):
            raise ValueError("Error: bias parameter must be smaller than "
                             "barrier parameter.")


        # Specify which accumulator is cued (gets boost) on this trial
        cueCon = np.ones(3, dtype=float)
        cueCon[cueIdx] = cueBoost

        # Specify the denominator in divisive normalisation 
        divWeight = dwIntercept + dwSlopePrev * np.array([0,0,1,0]) + dwSlopeProm * np.array([0,0,0,1])
        normFactor = np.add(np.multiply(divWeight,valueGain),np.multiply(divWeight,valueCntrl),np.multiply(divWeight,valueLoss))


        # Specify RDV update mean equations for each accumulator            
        meanArray = np.empty((4,3), dtype=object)        
        # Indices: 0-LM, 1-HM, 2-Prev, 3-Prom
        # Means for LM
        meanArray[0,0] = self.d * (cueCon[0] + np.divide(valueGain,normFactor[0],dtype=float))
        meanArray[0,1] = self.d * (cueCon[1] + np.divide(valueGain,normFactor[0],dtype=float))        
        meanArray[0,2] = self.d * (cueCon[2] + np.divide(valueLoss,normFactor[0],dtype=float))
        # Means for HM 
        meanArray[1,0] = self.d * (cueCon[0] + np.divide(valueGain,normFactor[1],dtype=float))
        meanArray[1,1] = self.d * (cueCon[1] + np.divide(valueGain,normFactor[1],dtype=float))        
        meanArray[1,2] = self.d * (cueCon[2] + np.divide(valueLoss,normFactor[1],dtype=float))
        # Means for Prev
        meanArray[2,0] = self.d * (cueCon[0] + np.divide(valueGain,normFactor[2],dtype=float))
        meanArray[2,1] = self.d * (cueCon[1] + np.divide(valueGain,normFactor[2],dtype=float))        
        meanArray[2,2] = self.d * (cueCon[2] + np.divide(valueLoss,normFactor[2],dtype=float))
        # Means for Prom 
        meanArray[3,0] = self.d * (cueCon[0] + np.divide(valueGain,normFactor[3],dtype=float))
        meanArray[3,1] = self.d * (cueCon[1] + np.divide(valueGain,normFactor[3],dtype=float))        
        meanArray[3,2] = self.d * (cueCon[2] + np.divide(valueLoss,normFactor[3],dtype=float))


        # Package the parameters
        # Variant-specific parameters
        specParams = np.empty(6, dtype=object)
        specParams[0] = biasArray
        specParams[1] = meanArray
        specParams[2] = cueIdx
        specParams[3] = valueGain
        specParams[4] = valueCntrl
        specParams[5] = valueLoss
        return specParams                        

    def contextBiasLin_divNorm(self, biasIntercept, biasSlopeGain, biasSlopeLoss, biasSlopeProm, biasSlopePrev, cueBoost, divWeight, cueIdx, valueGain, valueCntrl, valueLoss):
        """
        Args are parameters that change between instances (trials) of model
        Args: 
            cueIdx, int, specify which accumulator is cued on this sim trial
                (denoted by RDVaccum index)
            valueGain: value of Gain alternative for simulation
            valueCntrl: value of Cntrl alternative for simulation
            valueLoss: value of Loss alternative for simulation
                (enter as positive value)
        """
        # Specify number of parameters
        self.numParams = 11

        # Model the starting value with a linear model
        # Dummy code the three cue conditions (gain, cntrl, noloss)
        # The cntrl cue condition is placed in the intercept, 
        # so only two slopes / dummy arrays need be specified        
        biasArray = biasIntercept + \
        biasSlopeGain * np.tile([1,0,0],4) + \
        biasSlopeLoss * np.tile([0,0,1],4) + \
        biasSlopePrev  * np.concatenate([np.repeat(0,3),np.repeat(0,3),np.repeat(1,3),np.repeat(0,3)]) + \
        biasSlopeProm * np.concatenate([np.repeat(0,3),np.repeat(0,3),np.repeat(0,3),np.repeat(1,3)]) 
            
        if any(biasArray > self.barrierStart):
            raise ValueError("Error: bias parameter must be smaller than "
                             "barrier parameter.")


        # Specify which accumulator is cued (gets boost) on this trial
        cueCon = np.ones(3, dtype=float)
        cueCon[cueIdx] = cueBoost

        # Specify the denominator in divisive normalisation 
        normFactor = np.sum([divWeight*valueGain, divWeight*valueCntrl, divWeight*valueLoss],dtype=float)
        # Specify RDV update mean equations for each accumulator            
        meanGain = self.d * (cueCon[0] + np.divide(valueGain,normFactor,dtype=float))
        meanCntrl = self.d * (cueCon[1] + np.divide(valueCntrl,normFactor,dtype=float))
        meanLoss = self.d * (cueCon[2] + np.divide(valueLoss,normFactor,dtype=float))
        # Assign to array
        meanArray = np.array([meanGain, meanCntrl, meanLoss], dtype=float)

        # Package the parameters
        # Variant-specific parameters
        specParams = np.empty(6, dtype=object)
        specParams[0] = biasArray
        specParams[1] = meanArray
        specParams[2] = cueIdx
        specParams[3] = valueGain
        specParams[4] = valueCntrl
        specParams[5] = valueLoss
        return specParams        
    


class fitParamContain():
    def __init__(self, fitParams, fitlikelihood):
        self.fitParams = fitParams    
        self.fitlikelihood = fitlikelihood
        
        self.d = self.fitParams[0]
        self.sigma = self.fitParams[1]
        #self.NDT = self.fitParams[2]
        #self.barrierDecay = self.fitParams[3]

    def multiBiasLin_divNorm_Params(self):        
        self.cueBoost = self.fitParams[2]
        self.divWeight = self.fitParams[3]
        self.biasIntercept = self.fitParams[4]
        self.biasSlopeGain = self.fitParams[5]
        self.biasSlopeLoss = self.fitParams[6]
        return self      

    def linBias_linDW_Params(self):  
        self.cueBoost = self.fitParams[2]
        self.dwIntercept = self.fitParams[3]
        self.dwSlopeGain = self.fitParams[4]
        self.dwSlopeLoss = self.fitParams[5]
        self.biasIntercept = self.fitParams[6]
        self.biasSlopeGain = self.fitParams[7]
        self.biasSlopeLoss = self.fitParams[8]
        return self      

    def contextBiasLin_divNorm_Params(self):        
        self.cueBoost = self.fitParams[2]
        self.divWeight = self.fitParams[3]
        self.biasIntercept = self.fitParams[4]
        self.biasSlopeGain = self.fitParams[5]
        self.biasSlopeLoss = self.fitParams[6]
        self.biasSlopePrev = self.fitParams[7]
        self.biasSlopeProm = self.fitParams[8]
        return self        
    
    def contextBiasInterac_divNorm_Params(self):        
        self.cueBoost = self.fitParams[2]
        self.divWeight = self.fitParams[3]
        self.biasIntercept = self.fitParams[4]
        self.biasSlopeGain = self.fitParams[5]
        self.biasSlopeLoss = self.fitParams[6]
        self.biasSlopePrev = self.fitParams[7]
        self.biasSlopeProm = self.fitParams[8]
        self.biasPrevLossInt = self.fitParams[9]
        self.biasPromGainInt = self.fitParams[10]
        return self        

class genParamContain(): 
    def __init__(self, currParams):
        self.genParams = currParams

        self.d = self.genParams[0]
        self.sigma = self.genParams[1]
        #self.NDT = self.genParams[2]
        #self.barrierDecay = self.genParams[3]

    def multiBiasLin_divNorm_Params(self):
        self.cueBoost = self.fitParams[2]
        self.divWeight = self.fitParams[3]
        self.biasIntercept = self.fitParams[4]
        self.biasSlopeGain = self.fitParams[5]
        self.biasSlopeLoss = self.fitParams[6]
        return self        

    def linBias_linDW_Params(self):  
        self.cueBoost = self.fitParams[2]
        self.dwIntercept = self.fitParams[3]
        self.dwSlopeGain = self.fitParams[4]
        self.dwSlopeLoss = self.fitParams[5]
        self.biasIntercept = self.fitParams[6]
        self.biasSlopeGain = self.fitParams[7]
        self.biasSlopeLoss = self.fitParams[8]
        return self 

    def linBias_ctxtDW_Params(self):  
        self.cueBoost = self.fitParams[2]
        self.dwIntercept = self.fitParams[3]
        self.biasIntercept = self.fitParams[4]
        self.biasSlopeGain = self.fitParams[5]
        self.biasSlopeLoss = self.fitParams[6]
        self.dwSlopePrev = self.fitParams[7]
        self.dwSlopeProm = self.fitParams[8]
        return self        

    def contextBiasLin_divNorm_Params(self):        
        self.cueBoost = self.fitParams[2]
        self.divWeight = self.fitParams[3]
        self.biasIntercept = self.fitParams[4]
        self.biasSlopeGain = self.fitParams[5]
        self.biasSlopeLoss = self.fitParams[6]
        self.biasSlopePrev = self.fitParams[7]
        self.biasSlopeProm = self.fitParams[8]
        return self    

    def contextBiasInterac_divNorm_Params(self):        
        self.cueBoost = self.fitParams[2]
        self.divWeight = self.fitParams[3]
        self.biasIntercept = self.fitParams[4]
        self.biasSlopeGain = self.fitParams[5]
        self.biasSlopeLoss = self.fitParams[6]
        self.biasSlopePrev = self.fitParams[7]
        self.biasSlopeProm = self.fitParams[8]
        self.biasPrevLossInt = self.fitParams[9]
        self.biasPromGainInt = self.fitParams[10]
