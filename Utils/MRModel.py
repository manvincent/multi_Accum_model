#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 20:34:17 2018
Module: MRModel.py
Author: Vincent Man, v.man@mail.utoronto.ca
Date: 2018 05 24

Acknowledgements
Core model engine, adapted from ADDM (GNU General Public License)
https://github.com/goptavares/aDDM-Toolbox
Original author:  Gabriela Tavares  gtavares@caltech.edu
Please see this paper for details on PTA algorithm designed by Gabriela Tavares: 

Tavares, G., Perona, P., & Rangel, A. (2017). 
The attentional Drift Diffusion Model of simple perceptual decision-making. 
Frontiers in neuroscience, 11, 468.
https://www.frontiersin.org/articles/10.3389/fnins.2017.00468/full

@author: vincentman
"""


from __future__ import absolute_import, division
import numpy as np
from scipy.stats import norm


class SimTrial(object):

    def __init__(self, choice, RT, barrierVec, RDVvecs, timeVec):
        """
        Hold evolved RDV and barrier of simulated trial for plotting
        Args:
            choice: response on this trial
            RT: response time in ms
            cueIdx: int, specify which accumulator was cued on this sim trial
                (denoted by RDVaccum index)
            simGain: hold gain value specified for simulation
            simCntrl: hold cntrl value specified for simulation
            simLoss: hold loss value specified for simulation
            barrierVec: float vector, barrier values across time
            RDVvecs: float array, RDV for each accumulator across time
            timeVec: float vector, time axis values
        """
        self.choice = choice
        self.RT = RT
        self.barrierVec = barrierVec
        self.RDVvecs = RDVvecs
        self.timeVec = timeVec    


class Model(object):

    def __init__(self, genParams, specParams, stateStep=0.01, timeStep=20, maxRT=500): 
        """
        Initialize the race model object.  
        
        Note: Highly reocmmend a max stateStep of 0.01 -- coarser stateSteps repoduce unreliable fits
        """ 
        # Parameters general across all variants of the multialternative race model   
        self.d = genParams[0]
        self.sigma = genParams[1]
        self.NDT = genParams[2]        
        self.barrierDecay = genParams[3]
        self.barrierStart = genParams[4]
        self.lowerBound = genParams[5]
        self.numAcc = genParams[6]
        # Specific parameters
        self.biasArray = specParams[0]
        self.meanArray = specParams[1]
        self.cueIdx = specParams[2]
        self.valueGain = specParams[3]
        self.valueCntrl = specParams[4]
        self.valueLoss = specParams[5]
        # Parameters for modelling algorithm (not part of computational model)
        self.stateStep = stateStep
        self.timeStep = timeStep
        self.maxRT = maxRT


    def simulate_trial(self):
        """
        Generate a race model trial given the item values
            
        Returns:
            SimTrial: simulation trial (w/ trajectories for plotting)

        """

        # Specify starting RDV for each accumulator
        RDVaccum = np.empty(3, dtype=float)
        for a in range(self.numAcc):
            RDVaccum[a] = self.biasArray[a]
       
        # Save trajectories for plots
        barrierVec = []
        RDVvecs = [[] for i in range(self.numAcc)]
        timeVec = []

        time = 0
        elapsedNDT = 0
        while True:
            # Compute where the absorbing barrier is
            barrier = self.barrierStart / (1 + self.barrierDecay * time)
            # Append barrier value to array for simulation plot
            barrierVec.append(barrier)
            # Append time to array for simulation plot
            timeVec.append(time * self.timeStep)

            # NDT period
            if elapsedNDT < self.NDT // self.timeStep:
                meanAccum = np.zeros(3)
                elapsedNDT += 1
            else:
                meanAccum = self.meanArray

            # Sample the change in RDV from the distribution.
            for a in range(self.numAcc):
                if time > 0:
                    RDVaccum[a] += np.random.normal(meanAccum[a], self.sigma)
                if RDVaccum[a] < self.lowerBound:
                    RDVaccum[a] = self.lowerBound
                # Append to array for simulation plot
                RDVvecs[a].append(RDVaccum[a])

            # If RDV has not hit barrier by max RT, terminate trial and collect
            # Response
            if (time * self.timeStep) >= self.maxRT:
                RT = np.nan
                choice = np.nan
                break

            # If any RDV hits one of the barriers, the trial is over.
            if any(RDVaccum >= barrier):
                #print 'trial in bounds'
                RT = time * self.timeStep
                winAccum = np.nonzero(RDVaccum == np.amax(RDVaccum))
                if len(winAccum[0]) > 1:
                    choice = np.random.choice(winAccum[0])
                else:
                    choice = winAccum[0][0]
                RDVvecs[choice][-1] = barrier
                break
                   
            # Update the time step
            time += 1        
        return SimTrial(choice, RT, barrierVec, RDVvecs, timeVec)

    def get_trial_likelihood(self):
        """
        Computes likelihood of data from a single race trial using the PTA
        Args:
            (note: the timeStep was defined with raceModel.__init__)
        Returns:
            Likelihood obtained for given trial and model
        """
    
        # Get the number of time steps for this trial.
        numTimeSteps = np.round(self.maxRT // self.timeStep).astype(int)
        if numTimeSteps < 1:
            raise RuntimeError(u"Trial response time is smaller than time "
                               "step.")

        # Compute the locations of absorbing barrier
        barrierUp = np.ones(numTimeSteps) * self.barrierStart
        barrierDown = np.ones(numTimeSteps) * self.lowerBound
        for time in range(1, numTimeSteps):
            barrierUp[time] = self.barrierStart / \
                (1 + self.barrierDecay * time)

        # Obtain correct state step.
        # This step ensures the number of bins is an integer
        numStateBins = np.ceil(self.barrierStart / self.stateStep)
        # This step ensures that the denoted states includes specified barrier
        # value
        stateStep = self.barrierStart / (numStateBins + 0.5)

        # The vertical axis is divided into states.
        states = np.arange(barrierDown[0] + (stateStep / 2),
                           barrierUp[0] - (stateStep / 2) + stateStep,
                           stateStep)

        # Compute matrix of delta RDV across state space
        changeMatrix = np.subtract(states.reshape(states.size, 1), states)
        # Compute vectors of delta RDV needed to hit boundaries
        changeUp = np.subtract(barrierUp, states.reshape(states.size, 1))
        changeDown = np.subtract(barrierDown, states.reshape(states.size, 1))

        # Initialize structure to store state space across accumulators
        prStates = np.zeros((self.numAcc, states.size, numTimeSteps), dtype=float)        
        probUpCrossing = np.zeros((self.numAcc, numTimeSteps), dtype=float)        
        probDownCrossing = np.zeros((self.numAcc, numTimeSteps), dtype=float)
        # Initialize state space for each accumulator
        for a in range(self.numAcc):
            # Find the state corresponding to the bias parameter.
            biasStateIdx = (np.abs(states - self.biasArray[a])).argmin()
            # Initial probability for all states to zero, except the bias
            # state, for which the initial probability is one.
            prStates[a] = np.zeros((states.size, numTimeSteps))            
            prStates[a, biasStateIdx, 0] = 1
            # Initialize probability of crossing each barrier
            # over the time of the trial.
            probUpCrossing[a] = np.zeros(numTimeSteps)
            probDownCrossing[a] = np.zeros(numTimeSteps)

        # Initialize NDT
        elapsedNDT = 0
        # Iterate over time steps
        for time in range(1,numTimeSteps):
            # We use a normal distribution to model changes in RDV
            # stochastically. The mean of the distribution (the change most
            # likely to occur) is calculated from the model parameter d and
            # from the item values, except during non-decision time, in which
            # the mean is zero.
            if elapsedNDT < self.NDT // self.timeStep:
                meanAccum = np.zeros(self.numAcc, dtype=float)
                elapsedNDT += 1
            else:
                meanAccum = self.meanArray

            # Update the probability of the states that remain inside the
            # barriers. The probability of being in state B is the sum, over
            # all states A, of the probability of being in A at the previous
            # time step times the probability of changing from A to B. We
            # multiply the probability by the stateStep to ensure that the area
            # under the curves for the probability distributions probUpCrossing
            # and probDownCrossing add up to 1.
            for a in range(self.numAcc):     

                # Recode crossing probabilities to remove extreme values for poor models
                probUpCrossing[np.logical_or.reduce((~np.isfinite(probUpCrossing),probUpCrossing<1e-250,probUpCrossing>1))] = 1e-250
                probDownCrossing[np.logical_or.reduce((~np.isfinite(probDownCrossing),probDownCrossing<1e-250,probDownCrossing>1))] = 1e-250                
                
                # Probability of any alternative having crossed the upper/lower bound            
                prAltCross = np.sum([np.sum([probUpCrossing[i, 0:time-1] for i in range(self.numAcc) if i != a], axis=0) -
                                np.prod([probUpCrossing[i, 0:time-1] for i in range(self.numAcc) if i != a], axis=0)], dtype = np.float64)                

                # Legacy - weigh by alternative accumulators only on previous time point 
                # prAltCross = np.sum([np.sum([probUpCrossing[i, time-1] for i in range(self.numAcc) if i != a], axis=0) -
                #                 np.prod([probUpCrossing[i, time-1] for i in range(self.numAcc) if i != a], axis=0)], dtype = np.float64)    
                
                prWeight = (1 - prAltCross)
                prStatesNew = (stateStep * (np.dot(norm.pdf(changeMatrix, meanAccum[a], self.sigma), prStates[a, :, time - 1]) * prWeight))
                
                # Legacy - nor reweighting the prob updates by how likely alternatives crossed previously
                #prStatesNew = (stateStep * (np.dot(norm.pdf(changeMatrix, meanAccum[a], self.sigma), prStates[a, :, time - 1])))

            
                # Recode state probabilities to remove extreme values for poor models
                prStatesNew[np.logical_or.reduce((~np.isfinite(prStatesNew),prStatesNew<1e-250,prStatesNew>1))] = 1e-250                
                # Set probabilities of states out of decision space to zero 
                prStatesNew[(states >= barrierUp[time])] = 0
                prStatesNew[(states < barrierDown[time])] = 0

                # Legacy code for PTA that re-spreads at each timepoint of lower bound
                # Find the state corresponding to the lower reflecting bound
                # barrierDownIdx = np.abs(states - barrierDown[time]).argmin()
                # prStatesNew[(barrierDownIdx)] = np.dot(prStates[:, time - 1],
                # (prStatesNew[(barrierDownIdx)] + norm.cdf(changeDown[:,
                # time], meanAccum, self.sigma)))

                # Calculate the probabilities of crossing the up barrier and the
                # down barrier. This is given by the sum, over all states A, of the
                # probability of being in A at the previous timestep times the
                # probability of crossing the barrier if A is the previous
                # state.
                tempUpCross = (np.dot(prStates[a, :, time - 1],
                    (1 - norm.cdf(changeUp[:, time], meanAccum[a], self.sigma))) * prWeight)

                tempDownCross = (np.dot(prStates[a, :, time - 1],
                    norm.cdf(changeDown[:, time], meanAccum[a], self.sigma)) * prWeight)

                
                # # Compute renormalization constants
                # sumIn = np.sum(prStates[a, :, time - 1])
                # sumCurrent=np.sum(prStatesNew) + tempUpCross + tempDownCross

                # # Legacy code (see above)
                # # sumCurrent = np.sum(prStatesNew) + tempUpCross

                # # Renormalize to cope with numerical approximations.
                # if not sumCurrent == 0:
                #     prStatesNew=prStatesNew  * sumIn / sumCurrent
                #     tempUpCross=tempUpCross * sumIn / sumCurrent
                #     tempDownCross=tempDownCross  * sumIn / sumCurrent

                # Update the probabilities of each state and the probabilities of
                # crossing each barrier at this timestep.
                prStates[a, :, time]=prStatesNew
                probUpCrossing[a, time]=tempUpCross
                probDownCrossing[a, time]=tempDownCross
        
        # Remove nan, -inf, inf :
        prStates[np.logical_or.reduce((~np.isfinite(prStates),prStates<1e-250,prStates>1))] = 1e-250
        probUpCrossing[np.logical_or.reduce((~np.isfinite(probUpCrossing),probUpCrossing<1e-250,probUpCrossing>1))] = 1e-250
        probDownCrossing[np.logical_or.reduce((~np.isfinite(probDownCrossing),probDownCrossing<1e-250,probDownCrossing>1))] = 1e-250      
        return(prStates, probUpCrossing, probDownCrossing)
