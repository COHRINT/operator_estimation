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

    def buildModels(self, dataSet, num_tar=5,saveFileName='hmm_train.npy'):
        histModels = {}
        warnings.filterwarnings("ignore")
        for i in tqdm(range(num_tar),ncols=100):
            allTypeData = []
            allTypeLengths = []
            currentSet = dataSet[i]
            for j in range(len(currentSet)):
                allTypeData.append(currentSet[j])
                allTypeLengths.append(len(currentSet[j]))

            allTypeData=np.concatenate(allTypeData)
            allTypeData=np.reshape(allTypeData,(-1,1))

            model = hmm.GaussianHMM(n_components=4).fit(allTypeData,allTypeLengths)

            model_store = {}
            model_store['transition'] = model.transmat_.tolist()
            model_store['prior'] = model.startprob_.tolist()

            means = model.means_.tolist()
            var = model.covars_.tolist()
            obs = []
            for j in range(len(means)):
                obs.append(GM(means[j],var[j],1))

            model_store['obs'] = obs

            histModels['Cumuliform'+str(i)] = model_store
        if num_tar==10:
            np.save('hmm_train_10.npy',histModels)
        else:
            np.save(saveFileName,histModels)

    def buildDataSet(self, num_sets=100, num_tar=5):
        if num_tar==10:
            allSeries=[[],[],[],[],[],[],[],[],[],[]]
            var=0.1
        else:
            allSeries=[[],[],[],[],[]]
            var=2
        for i in range(num_tar):
            model=Cumuliform(genus=i,weather=False)
            b=copy.deepcopy(model.intensityModel)
            for j in range(num_sets):
                c=b+np.random.normal(0,var,(len(b)))
                for k in range(len(c)):
                    c[k]=max(c[k],1e-5)
                allSeries[i].append(c)
        return allSeries


    def continueForward(self, newData, model, prevAlpha=[-1,-1]):
        x0 = model['prior']
        pxx = model['transition']
        pyx = model['obs']

        numStates = len(x0)
        if prevAlpha[0] == -1:
            prevAlpha=x0

        newAlpha = [0]*numStates
        for xcur in range(numStates):
            for xprev in range(numStates):
                newAlpha[xcur] += prevAlpha[xprev]*pxx[xcur][xprev]
            newAlpha[xcur] = newAlpha[xcur]*pyx[xcur].pointEval(newData)
        return newAlpha

    def testHMM(self,num_events,num_tar=5):
        if num_tar==10:
            modelFileName = 'hmm_train_10.npy'
        else:
            modelFileName = 'hmm_train.npy'
        models = np.load(modelFileName).item()

        genNames=[]
        for i in range(num_tar):
            genNames.append('Cumuliform'+str(i))
        #  genNames = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']
        correct=0
        #  for i in range(num_tar):
        for i in tqdm(range(num_events),ncols=100):
            genus=np.random.randint(5)
            #  genus=0
            species = Cumuliform(genus = genus,weather=False)
            data = species.intensityModel
            var=0.5
            data=data+np.random.normal(0,var,(len(data)))


            alphas = {}
            for i in genNames:
                alphas[i] = [-1,-1]

            probs = {}
            for i in genNames:
                probs[i] = .2
            
            count=0
            while max(probs.values())<0.4:
                    #update classification probs
                    for i in genNames:
                        alphas[i] = self.continueForward(data[count], models[i], alphas[i])
                        #  print alphas[i]
                        probs[i] = probs[i]*sum(alphas[i])

                    #normalize probs
                    suma = sum(probs.values())
                    for i in genNames:
                        probs[i] = probs[i]/suma
                    count+=1
            #  print np.max(probs.values())
            chosen=np.argmax(probs.values())
            #  print genus,chosen
            if genus==chosen:
                correct+=1
        print (correct/num_tar)

if __name__ == '__main__':
    hc=HMM_Classification()
    commands=[]
    for i in range(1,len(sys.argv)):
        commands.append(sys.argv[i])

    if 'train' in commands:
        dataSet=hc.buildDataSet(500,10)
        hc.buildModels(dataSet,10)

    if 'test' in commands:
        hc.testHMM(500,10)
