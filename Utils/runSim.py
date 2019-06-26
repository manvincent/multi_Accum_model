"""
Module: runSim.py
Author: Vincent Man, v.man@mail.utoronto.ca
Date: 2018 05 24

"""
import time
import numpy as np
from scipy.stats import rv_histogram


class SimData(object):

    def __init__(self, model, simDistrib):
        """
        Takes in simulated distributions and creates datasets of given number of trials
        Args: 
            model: object, the specified computational model
            simDistrib: class object, the simulated distribution from which to sample to generate datset
                        Breaks down into following relevant attributes: 
                        bins: 

            numTrials: int, the number of trials to generate (# trials per exp. condition)
        """
        self.model = model
        self.simDistrChoice = simDistrib.simDistrChoice
        self.bins = simDistrib.bins
        self.simChoiceHist = simDistrib.simChoiceHist

    def createSample(self, numTrials):
         # Determine proportion of each response type
        [unique, counts] = np.unique(self.simDistrChoice, return_counts=True)        
        totalSims = np.sum(~np.isnan(self.simDistrChoice))
        
        simPDF = np.empty(self.model.numAcc, dtype=object)
        numChoices = np.zeros(self.model.numAcc, dtype=int)
        sampleRT = np.empty(self.model.numAcc, dtype=object)
        simSampleHist = np.empty(self.model.numAcc, dtype=object)
        for a in range(self.model.numAcc):
            # Convert simulation histograms into PDFs (normalized so sum = 1 for
            # each accumulator)
            simPDF[a] = rv_histogram([self.simChoiceHist[a], self.bins])
            # Compute number of sim trials for each response type 
            if a in unique:
                numChoices[a] = np.floor(np.divide(counts[np.where(unique==a)[0]], totalSims, dtype=float) * numTrials)
            # For each response type, draw the corresponding number of sim trials from respective PDFs   
            sampleRT[a] = simPDF[a].rvs(size=numChoices[a])
            # Compute histograms for each accumulator's RTs
            [simSampleHist[a],_] = np.histogram(sampleRT[a],
                                                    bins=self.model.maxRT // self.model.timeStep,
                                                    range=(0, self.model.maxRT))


        # Concatenate generated data across response types
        simSampleChoice = np.repeat(np.arange(self.model.numAcc),numChoices)
        simSampleRT = np.concatenate(sampleRT)

        return simSampleChoice, simSampleRT, simSampleHist


class SimDistrib(object):

    def __init__(self, simDistrRT, simDistrChoice, runSimsTime,
                 simDistrRDV, simChoiceRT, simChoiceHist, bins):
        """
        Hold arrays of results from simulated distributions
        Args:
            simDistrRT: float array, hold all RTs across alternatives
            simDistrChoice: int array, hold all choices across alternatives
            runSimsTime, float, hold the elapsed time (sec) to run sims for this distribution

            simDistrRDV: array, first index denotes accumulator ID, second index holds
                        final RDV values for that accumulator 
            simChoiceRT: float array, index denotes accumulator ID
                        holds RTs for only the trials chosen by that accumulator 
            simChoiceHist: int array, index denotes accumulator ID
                        holds histogram for freq counts of trials chosen by that accumulator 
            bins: array, stores the bins (time axis) for histogram objects
        """
        self.simDistrRT = simDistrRT
        self.simDistrChoice = simDistrChoice
        self.runSimsTime = runSimsTime
        self.simDistrRDV = simDistrRDV
        self.simChoiceRT = simChoiceRT
        self.simChoiceHist = simChoiceHist
        self.bins = bins


class RunSimulation(object):

    def __init__(self, model, numSims):
        """
        Initialize the simulation distribution object
        Args are parameters specific to the distribution at hand, and
        are a function of:
        Args:
            model: object, input the model that drives simulation
            numSims: int, number of simulations per single distribution object
            specParams: list, model-specific parameters
        """
        self.model = model
        self.numSims = numSims

    def simulate_distrib(self):
        # Initiate the timer
        startTime = time.time()

        # Initialize containers to store sim trial results
        simDistrRT = np.empty(self.numSims, dtype=float)
        simDistrChoice = np.empty(self.numSims, dtype=float)
        simDistrRDV = np.empty((self.model.numAcc, self.numSims), dtype=float)
        simChoiceRT = np.empty(self.model.numAcc, dtype=object)
        simChoiceHist = np.empty(self.model.numAcc, dtype=object)

        for simIdx in range(self.numSims):
            # Simulate a trial from these model and exp. parameters
            sim = self.model.simulate_trial()
            # Store simulated trial results
            simDistrRT[simIdx] = sim.RT
            simDistrChoice[simIdx] = sim.choice
            for a in range(self.model.numAcc):
                simDistrRDV[a] = sim.RDVvecs[a][-1]

        for a in range(self.model.numAcc):

            # Parse separate RT arrays for each accumlator
            simChoiceRT[a] = simDistrRT[(simDistrChoice == a)]

            # Compute histograms for each accumulator's RTs
            [simChoiceHist[a], bins] = np.histogram(simChoiceRT[a],
                                                    bins=self.model.maxRT // self.model.timeStep,
                                                    range=(0, self.model.maxRT))

        # Record elapsed time for simulations
        runSimsTime = time.time() - startTime
        return SimDistrib(simDistrRT, simDistrChoice, runSimsTime,
                          simDistrRDV, simChoiceRT, simChoiceHist, bins)
