"""
Module: DefineModel_Exp1.py
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
    cueBoost_bounds = (1.0, 30.0)
    divWeight_bounds = (0.001,1.0)
    bias_bounds = biasGain_bounds = biasCntrl_bounds = biasLoss_bounds = (0.0, 0.9)    
    biasIntercept_bounds = (0.0,0.65)
    biasSlopeGain_bounds = biasSlopeLoss_bounds = (0.0,0.35)
    allBounds = [d_bounds, sigma_bounds, NDT_bounds, barrierDecay_bounds,
                cueBoost_bounds, divWeight_bounds, bias_bounds, biasGain_bounds, biasCntrl_bounds,
                biasLoss_bounds, biasIntercept_bounds, biasSlopeGain_bounds, biasSlopeLoss_bounds]
    
    # Subset parameter list depending on model type 
    paramMask = np.ones(len(allBounds), dtype=np.bool)
        
    if modelType == "multiBias":
        paramMask[5:7] = paramMask[10:13] = False
    elif modelType == "singleBias":
        paramMask[5] = paramMask[7:13] = False
    elif modelType == "multiBiasLin":
        paramMask[5:10] = False
    elif modelType == 'multiBias_divNorm':
        paramMask[6] = paramMask[10:13] = False
    elif modelType == 'multiBias_Linear_divNorm':
        paramMask[6:10] = False
    else:
        raise ValueError("Error: Model Type not recognized.")        
    
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

    def singleBias(self, bias, cueBoost, cueIdx, valueGain, valueCntrl, valueLoss):
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
        self.numParams = 6

        # Set all accumulators to have same starting value        
        biasArray = np.empty(3, dtype=float)
        biasArray[:] = bias
        if any(biasArray > self.barrierStart):
            raise ValueError("Error: bias parameter must be smaller than "
                             "barrier parameter.")

        # Specify which accumulator is cued (gets boost) on this trial
        cueCon = np.ones(3, dtype=float)
        cueCon[cueIdx] = cueBoost

        # Specify RDV update mean equations for each accumulator
        meanGain = self.d * (cueCon[0] + valueGain -
                             np.mean([valueCntrl, valueLoss]))
        meanCntrl = self.d * \
            (cueCon[1] + valueCntrl - np.mean([valueGain, valueLoss]))
        meanLoss = self.d * (cueCon[2] + valueLoss -
                             np.mean([valueGain, valueCntrl]))
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

    def multiBias(self, biasGain, biasCntrl, biasLoss, cueBoost, cueIdx, valueGain, valueCntrl, valueLoss):
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
        self.numParams = 8

        # Set all accumulators to have same starting value
        biasArray = np.empty(3, dtype=float)
        biasArray[0] = biasGain
        biasArray[1] = biasCntrl
        biasArray[2] = biasLoss
        if any(biasArray > self.barrierStart):
            raise ValueError("Error: bias parameter must be smaller than "
                             "barrier parameter.")

        # Specify which accumulator is cued (gets boost) on this trial
        cueCon = np.ones(3, dtype=float)
        cueCon[cueIdx] = cueBoost

        # Specify RDV update mean equations for each accumulator
        meanGain = self.d * (cueCon[0] + valueGain -
                             np.mean([valueCntrl, valueLoss]))
        meanCntrl = self.d * \
            (cueCon[1] + valueCntrl - np.mean([valueGain, valueLoss]))
        meanLoss = self.d * (cueCon[2] + valueLoss -
                             np.mean([valueGain, valueCntrl]))
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

    def multiBias_divNorm(self, biasGain, biasCntrl, biasLoss, cueBoost, divWeight, cueIdx, valueGain, valueCntrl, valueLoss):
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

        # Set all accumulators to have same starting value
        biasArray = np.empty(3, dtype=float)
        biasArray[0] = biasGain
        biasArray[1] = biasCntrl
        biasArray[2] = biasLoss
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

    def multiBiasLin(self, biasIntercept, biasSlopeGain, biasSlopeLoss, cueBoost, cueIdx, valueGain, valueCntrl, valueLoss):
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
        self.numParams = 8
        
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

        # Specify RDV update mean equations for each accumulator
        meanGain = self.d * (cueCon[0] + valueGain -
                             np.mean([valueCntrl, valueLoss]))
        meanCntrl = self.d * \
            (cueCon[1] + valueCntrl - np.mean([valueGain, valueLoss]))
        meanLoss = self.d * (cueCon[2] + valueLoss -
                             np.mean([valueGain, valueCntrl]))
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

    def multiBias_Linear_divNorm(self, biasIntercept, biasSlopeGain, biasSlopeLoss, cueBoost, divWeight, cueIdx, valueGain, valueCntrl, valueLoss):
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

    


class fitParamContain():
    def __init__(self, fitParams, fitlikelihood):
        self.fitParams = fitParams    
        self.fitlikelihood = fitlikelihood
        
        self.d = self.fitParams[0]
        self.sigma = self.fitParams[1]
        self.NDT = self.fitParams[2]
        self.barrierDecay = self.fitParams[3]

    def multiBiasParams(self):
        self.cueBoost = self.fitParams[4]
        self.biasGain = self.fitParams[5]
        self.biasCntrl = self.fitParams[6]
        self.biasLoss = self.fitParams[7]
        return self

    def multiBiasDivNormParams(self):
        self.cueBoost = self.fitParams[4]
        self.divWeight = self.fitParams[5]
        self.biasGain = self.fitParams[6]
        self.biasCntrl = self.fitParams[7]
        self.biasLoss = self.fitParams[8]
        return self        

    def multiBiasLinParams(self):
        self.cueBoost = self.fitParams[4]
        self.biasIntercept = self.fitParams[5]
        self.biasSlopeGain = self.fitParams[6]
        self.biasSlopeLoss = self.fitParams[7]
        return self
        
    def multiBiasLinDivNormParams(self):
        self.cueBoost = self.fitParams[4]
        self.divWeight = self.fitParams[5]
        self.biasIntercept = self.fitParams[6]
        self.biasSlopeGain = self.fitParams[7]
        self.biasSlopeLoss = self.fitParams[8]
        return self        

    def singleBiasParams(self): 
        self.cueBoost = self.fitParams[4]
        self.bias = self.fitParams[5]
        return self
   
class genParamContain(): 
    def __init__(self, currParams):
        self.genParams = currParams
        
        self.d = self.genParams[0]
        self.sigma = self.genParams[1]
        self.NDT = self.genParams[2]
        self.barrierDecay = self.genParams[3]

    def multiBiasParams(self):
        self.cueBoost = self.genParams[4]
        self.biasGain = self.genParams[5]
        self.biasCntrl = self.genParams[6]
        self.biasLoss = self.genParams[7]
        return self

    def multiBiasDivNormParams(self):
        self.cueBoost = self.fitParams[4]
        self.divWeight = self.fitParams[5]
        self.biasGain = self.fitParams[6]
        self.biasCntrl = self.fitParams[7]
        self.biasLoss = self.fitParams[8]
        return self        

    def multiBiasLinParams(self):
        self.cueBoost = self.fitParams[4]
        self.biasIntercept = self.fitParams[5]
        self.biasSlopeGain = self.fitParams[6]
        self.biasSlopeLoss = self.fitParams[7]
        return self

    def multiBiasLinDivNormParams(self):
        self.cueBoost = self.fitParams[4]
        self.divWeight = self.fitParams[5]
        self.biasIntercept = self.fitParams[6]
        self.biasSlopeGain = self.fitParams[7]
        self.biasSlopeLoss = self.fitParams[8]
        return self        


    def singleBiasParams(self): 
        self.cueBoost = self.genParams[4]
        self.bias = self.genParams[5]
        return self



