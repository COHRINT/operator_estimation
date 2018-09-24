from __future__ import division
import matplotlib.pyplot as plt
import sys
import numpy as np
from hmmlearn import hmm
from copy import deepcopy
import warnings
import os
import time
import copy
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data_sim')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))
from DynamicsProfiles import *
from gaussianMixtures import GM

class HMM_Classification():

    def __init__(self):
        pass
        #  self.models = loadModels(modelFileName)

    def buildModels(self, dataSet, saveFileName='hmm_train.npy'):

        histModels = {}
        warnings.filterwarnings("ignore")
        for i in tqdm(range(5),ncols=100):
            allTypeData = []
            allTypeLengths = []
            currentSet = dataSet[i]
            for j in range(len(currentSet)):
                allTypeLengths.append(len(currentSet[j,:]))


            currentSet=np.reshape(currentSet,(currentSet.shape[0]*currentSet.shape[1],1))

            allModels = []
            BICScores = []
            for n_states in range(2,10):
                modelTest = hmm.GaussianHMM(n_components=n_states).fit(currentSet,allTypeLengths)
                allModels.append(modelTest)
                logLikelihood,posteriors = modelTest.score_samples(currentSet,allTypeLengths)
                bic = np.log(len(currentSet))*n_states - 2*logLikelihood
                BICScores.append(bic)

            bestModel = allModels[np.argmin(BICScores)]
            best = {}
            best['transition'] = bestModel.transmat_.tolist()
            best['prior'] = bestModel.startprob_.tolist()

            means = bestModel.means_.tolist()
            var = bestModel.covars_.tolist()
            obs = []
            for j in range(len(means)):
                obs.append(GM(means[j],var[j],1))

            best['obs'] = obs

            histModels['Cumuliform'+str(i)] = best

        np.save(saveFileName,histModels)

    def buildDataSet(self, num_sets=100):
        #  models=[Cumuliform]
        subs=[str(i) for i in range(5)]
        allSeries=np.empty((5,num_sets,100))
        for i in range(5):
            model=Cumuliform(genus=i,weather=False)
            b=copy.deepcopy(model.intensityModel)
            for j in range(num_sets):
                c=b+np.random.normal(0,2,(len(b)))
                for k in range(len(c)):
                    c[k]=max(c[k],1e-5)
                allSeries[i,j,:]=c
        return allSeries


    def continueForward(self, newData, model, prevAlpha=[-1,-1]):
        x0 = model['prior']
        pxx = model['transition']
        pyx = model['obs']

        numStates = len(x0)
        if prevAlpha[0] == -1:
            prevAlpha=x0

        newAlpha = [-1]*numStates
        for xcur in range(numStates):
            newAlpha[xcur] = 0
            for xprev in range(numStates):
                newAlpha[xcur] += prevAlpha[xprev]*pxx[xcur][xprev]
            newAlpha[xcur] = newAlpha[xcur]*pyx[xcur].pointEval(newData)
        return newAlpha

    def testHMM(self,num_tar):
        modelFileName = 'hmm_train.npy'
        models = np.load(modelFileName).item()

        genNames = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']
        correct=0
        #  for i in range(num_tar):
        for i in tqdm(range(num_tar),ncols=100):
            genus=np.random.randint(5)
            species = Cumuliform(genus = genus,weather=False)
            data = species.intensityModel


            alphas = {}
            for i in genNames:
                alphas[i] = [-1,-1]

            probs = {}
            for i in genNames:
                probs[i] = .2

            while max(probs.values())<0.9:
                for d in data:
                    #update classification probs
                    for i in genNames:
                        alphas[i] = self.continueForward(d, models[i], alphas[i])
                        probs[i] = probs[i]*sum(alphas[i])

                    #normalize probs
                    suma = sum(probs.values())
                    for i in genNames:
                        probs[i] = probs[i]/suma
            chosen=np.argmax(probs.values())
            if genus==chosen:
                correct+=1
        print (correct/num_tar)

if __name__ == '__main__':
    hc=HMM_Classification()
    commands=[]
    for i in range(1,len(sys.argv)):
        commands.append(sys.argv[i])

    if 'train' in commands:
        dataSet=hc.buildDataSet(100)
        hc.buildModels(dataSet)

    if 'test' in commands:
        hc.testHMM(100)
