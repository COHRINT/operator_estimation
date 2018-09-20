from __future__ import division
import matplotlib.pyplot as plt
import sys
import numpy as np
import h5py
from hmmlearn import hmm
from copy import deepcopy
import warnings
import os
from gaussianMixtures import GM, Gaussian
import time
sys.path.append("../../../scenario_simulator/modules/")
from DynamicsProfiles import *

class HMM_Classification():

    def __init__(self, modelFileName='../data/histModels_fams.npy',genus=0):
        pass
        #  self.models = loadModels(modelFileName)
        #  self.species = Cumuliform(genus=genus, weather=False)
        #  self.family_names = ['Stratiform', 'Cirriform', 'Stratocumuliform', 'Cumuliform', 'Cumulonibiform']
        #  self.genus_names = ['Cumuliform0', 'Cumuliform1', 'Cumuliform2', 'Cumuliform3', 'Cumuliform4']

    def buildModels(self, dataSet, saveFileName='../data/histModels_final.npy', sizes=300):
        #  models = ['Stratiform','Cirriform','Stratocumuliform','Cumuliform','Cumulonibiform']
        models = ['Cumuliform']

        histModels = {}
        warnings.filterwarnings("ignore")
        for mod in models:
            print("Now building model {} from historical data.".format(mod))
            allFamData = []
            allFamLengths = []
            for i in range(5):
                print("Genus: {}".format(i))
                allTypeData = []
                allTypeLengths = []
                h = dataSet[mod][str(i)]
                for j in range(sizes):
                    allFamData.append(h[j][0:-1].tolist())
                    allFamLengths.append(len(h[j][0:-1]))
                    allTypeData.append(h[j][0:-1].tolist())
                    allTypeLengths.append(len(h[j][0:-1]))


                for k in range(len(allTypeData)):
                    for j in range(len(allTypeData[k])):
                        allTypeData[k][j] = [allTypeData[k][j]]

                allTypeData = np.concatenate(allTypeData)

                allModels = []
                allBIC = []
                for j in range(2,9):
                    a = hmm.GaussianHMM(n_components=j).fit(allTypeData,allTypeLengths)
                    allModels.append(a)
                    llh,posteriors = a.score_samples(allTypeData)
                    bic = 0
                    for k in range(len(allTypeData)):
                        bic += np.log(len(allTypeData[k]))*j - 2*llh
                    bic = bic/len(allTypeData)
                    allBIC.append(bic)

                bestModel = allModels[np.argmin(allBIC)]
                best = {}
                best['transition'] = bestModel.transmat_.tolist()
                best['prior'] = bestModel.startprob_.tolist()

                means = bestModel.means_.tolist()
                var = bestModel.covars_.tolist()
                obs = []
                for j in range(len(means)):
                    obs.append(GM(means[j],var[j],1))

                best['obs'] = obs

                nam = mod + str(i)
                print("Completed Profile: {}".format(nam))
                histModels[nam] = best


#              for k in range(len(allFamData)):
#                  for j in range(len(allFamData[k])):
#                      allFamData[k][j] = [allFamData[k][j]]
#              allFamData = np.concatenate(allFamData)

#              allModels = []
#              allBIC = []
#              for j in range(2,15):
#                  a = hmm.GaussianHMM(n_components=j).fit(allFamData,allFamLengths)
#                  allModels.append(a)
#                  llh,posteriors = a.score_samples(allFamData[0])
#                  bic = 0
#                  for k in range(len(allFamData)):
#                      bic += np.log(len(allFamData[k]))*j - 2*llh
#                  bic = bic/len(allFamData)
#                  allBIC.append(bic)
#              bestModel = allModels[np.argmin(allBIC)]
#              best = {}
#              best['transition'] = bestModel.transmat_.tolist()
#              best['prior'] = bestModel.startprob_.tolist()


#              means = bestModel.means_.tolist()
#              var = bestModel.covars_.tolist()
#              obs = []
#              for j in range(len(means)):
#                  obs.append(GM(means[j],var[j],1))

#              best['obs'] = obs

#              nam = mod
#              print("Completed Profile: {}".format(nam))
#              histModels[nam] = best

        #  saveFile = open(saveFileName,'w')
        np.save(saveFileName,histModels)


    def buildDataSet(self, size=300, weather=True):

        #get an intensity from each
        #  models = [Stratiform, Cirriform, Stratocumuliform, Cumuliform, Cumulonibiform]
        models = [Cumuliform]
        subs = [str(i) for i in range(5)]

        allSeries = {}

        baseSeries = {}
        for mod in models:
            #baseSeries[mod.__name__] = {}
            allSeries[mod.__name__] = {}
            for i in range(5):
                a = mod(genus = i,weather=weather)
                allSeries[mod.__name__][str(i)] = []
                for j in range(size):
                    b = deepcopy(a.intensityModel)
                    c = b+np.random.normal(0,2,(len(b)))
                    for k in range(len(c)):
                        c[k] = max(c[k],1e-5)

                    allSeries[mod.__name__][str(i)].append(c)

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


    def humanTesting(self):
        modelFileName = '../data/histModels_final.npy'
        models = np.load(modelFileName).item()
        #  print models

        #  genus = 'Cirriform0'
        species = Cirriform(genus = 0,weather=False)
        data = species.intensityModel

        #  famNames = ['Stratiform','Cirriform','Stratocumuliform','Cumuliform','Cumulonibiform']
        genNames = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']
        
        obsMod = {}
        obsMod['Cumuliform0'] = [.7,.1,.2,.2,.2,.6,.2,.2,.6,.2,.2,.6,.2,.2,.6]
        obsMod['Cumuliform1'] = [.2,.2,.6,.7,.1,.2,.2,.2,.6,.2,.2,.6,.2,.2,.6]
        obsMod['Cumuliform2'] = [.2,.2,.6,.2,.2,.6,.7,.1,.2,.2,.2,.6,.2,.2,.6]
        obsMod['Cumuliform3'] = [.2,.2,.6,.2,.2,.6,.2,.2,.6,.7,.1,.2,.2,.2,.6]
        obsMod['Cumuliform4'] = [.2,.2,.6,.2,.2,.6,.2,.2,.6,.2,.2,.6,.7,.1,.2]

        #  obsMod = {}
        #  obsMod['Stratiform'] = [.7,.1,.2,.2,.2,.6,.2,.2,.6,.2,.2,.6,.2,.2,.6]
        #  obsMod['Cirriform'] = [.2,.2,.6,.7,.1,.2,.2,.2,.6,.2,.2,.6,.2,.2,.6]
        #  obsMod['Stratocumuliform'] = [.2,.2,.6,.2,.2,.6,.7,.1,.2,.2,.2,.6,.2,.2,.6]
        #  obsMod['Cumuliform'] = [.2,.2,.6,.2,.2,.6,.2,.2,.6,.7,.1,.2,.2,.2,.6]
        #  obsMod['Cumulonibiform'] = [.2,.2,.6,.2,.2,.6,.2,.2,.6,.2,.2,.6,.7,.1,.2]

        alphas = {}
        for i in genNames:
            alphas[i] = [-1,-1]

        probs = {}
        for i in genNames:
            probs[i] = 1

        #for each bit of data
        for d in data:
            #update classification probs
            for i in genNames:
                alphas[i] = self.continueForward(d, models[i], alphas[i])
                probs[i] = probs[i]*sum(alphas[i])

            #normalize probs
            suma = sum(probs.values())
            for i in genNames:
                probs[i] = probs[i]/suma

            #show to human
            #get human observation
            print(probs)
            ob = int(raw_input("Which observation would you like to make?"))

            #  ob = -1
            #print('up')
            if ob != -1:
                #apply bayes rule
                for i in genNames:
                    probs[i] = probs[i]*obsMod[i][ob]

            #normalize probs
            suma = sum(probs.values())
            for i in genNames:
                probs[i] = probs[i]/suma

            self.probabilities = probs


if __name__ == '__main__':
    hc=HMM_Classification()
    commands=[]
    for i in range(1,len(sys.argv)):
        commands.append(sys.argv[i])

    if 'train' in commands:
        filename='../data/histModels_final.npy'
        dataSet=hc.buildDataSet(100,weather=False)
        hc.buildModels(dataSet,filename,100)

    if 'test' in commands:
        hc.humanTesting()
