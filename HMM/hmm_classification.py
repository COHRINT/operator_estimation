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
from scipy.misc import logsumexp
from sklearn.preprocessing import normalize

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

    def elnsum(self,elnx,elny):
        if np.isnan(elnx) or np.isnan(elny):
            if np.isnan(elnx):
                return elny
            else:
                return elnx
        else:
            if elnx>elny:
                return elnx + self.eln(1+np.exp(elny-elnx))
            else:
                return elny + self.eln(1+np.exp(elnx-elny))

    def eln(self,x):
        if x==0:
            return np.NaN
        elif x>0:
            return np.log(x)
        else:
            #  raise(NotImplementedError)
            return 0.0

    def elnproduct(self,elnx,elny):
        if np.isnan(elnx) or np.isnan(elny):
            return np.NaN
        else:
            return elnx + elny

    #  def continueForward(self, newData, model, prevAlpha=[-1,-1]):
    #      x0 = model['prior']
    #      pxx = model['transition']
    #      pyx = model['obs']

    #      numStates = len(x0)
    #      if prevAlpha[0] == -1:
    #          prevAlpha=x0

    #      newAlpha = [0]*numStates
    #      for xcur in range(numStates):
    #          for xprev in range(numStates):
    #              newAlpha[xcur] += prevAlpha[xprev]*pxx[xcur][xprev]
    #          newAlpha[xcur] = newAlpha[xcur]*pyx[xcur].pointEval(newData)
        #  return newAlpha

    def continueForward(self, newData, model, prevAlpha=[-1,-1]):
        x0 = model['prior']
        pxx = model['transition']
        pyx = model['obs']

        numStates = len(x0)
        #  print x0
        if prevAlpha[0] == -1:
            prevAlpha=x0
            #  print
            #  print x0

        newAlpha = [0]*numStates
        for xcur in range(numStates):
            for xprev in range(numStates):
                newAlpha[xcur]+=prevAlpha[xprev]*pxx[xcur][xprev]
                #  print pxx[xcur][xprev]
                #  print newAlpha
            newAlpha[xcur]=newAlpha[xcur]*pyx[xcur].pointEval(newData)
            #  print pyx[xcur].getMeans(),pyx[xcur].getVars(),pyx[xcur].pointEval(newData)
            #  print newData
        suma=sum(newAlpha)
        #  print suma
        if suma!=0:
            for state in range(len(newAlpha)):
                newAlpha[state]/=suma
            #  suma=np.finfo(float).eps
            suma=self.eln(suma)
        else:
            for state in range(len(newAlpha)):
                newAlpha[state]=1/len(x0)
            #  suma=-np.inf
            suma=-800
            #  suma=np.NaN
        #  print newAlpha,suma
        return newAlpha,suma

    #  def continueForward(self, newData, model, prevAlpha=[-1,-1]):
    #      x0 = model['prior']
    #      pxx = model['transition']
    #      pyx = model['obs']

    #      numStates = len(x0)
    #      #  print x0
    #      if prevAlpha[0] == -1:
    #          prevAlpha=[0]*numStates
    #          for j in range(numStates):
    #              prevAlpha[j]=self.eln(x0[j])
    #      #  print len(prevAlpha)

    #      newAlpha = [0]*numStates
    #      for xcur in range(numStates):
    #          for xprev in range(numStates):
    #              newAlpha[xcur] = self.elnproduct(newAlpha[xcur],np.exp(self.elnproduct(prevAlpha[xprev],self.eln(pxx[xcur][xprev]))))
    #          newAlpha[xcur] = self.elnproduct(self.eln(newAlpha[xcur]),self.eln(pyx[xcur].pointEval(newData)))
    #      return newAlpha

    def expNormalize(self,alpha_logs):
        #  print alpha_logs.shape
        #  sys.exit()
        numStates=alpha_logs.shape[1]
        numTypes=alpha_logs.shape[0]

        alpha_logs=alpha_logs.reshape((1,numStates*numTypes))
        prob_norm=np.zeros((alpha_logs.shape[0],alpha_logs.shape[1]))
        shift=np.nanmax(alpha_logs)
        #  print shift
        normalizer=0
        for j in range(numStates*numTypes):
            if np.isnan(alpha_logs[0,j]):
                pass
            else:
                normalizer+=np.exp(alpha_logs[0,j]-shift)
        #  print normalizer

        for j in range(numStates*numTypes):
            if np.isnan(alpha_logs[0,j]):
                pass
            else:
                prob_norm[0,j]=np.exp(alpha_logs[0,j]-shift)/normalizer

        prob_norm=prob_norm.reshape((numTypes,numStates))
        #  self.all_hmm=np.append(self.all_hmm,np.expand_dims(prob_norm,2),axis=2)
        #  print prob_norm
        #  print self.all_hmm
        #  sys.exit()
        prob_norm=np.sum(prob_norm,axis=1)

        #  print prob_norm,sum(prob_norm)
        return prob_norm

    #  def expNormalize(self,constants):
    #      shift=np.nanmax(constants)

    def graph_HMMs(self,genus,chosen):
        #gut check if HMMs are working
        #ONLY WORKS WITH 10 TARGETS
        self.all_hmm=self.all_hmm[:,:,1:]
        fig,ax=plt.subplots(nrows=5,ncols=2,tight_layout=True,figsize=((8,10)))

        x=range(self.all_hmm.shape[2])
        for j in range(2):
            for i in range(5):
                states=[]
                for k in x:
                    if sum(self.all_hmm[5*j+i,:,k])==0:
                        states.append(np.NaN)
                    else:
                        states.append(np.argmax(self.all_hmm[5*j+i,:,k]))
                if (5*j+i)==genus:
                    ax[i,j].set_facecolor('lightcoral')
                elif (5*j+i)==chosen:
                    ax[i,j].set_facecolor('cyan')
                #  ax[i,j].plot(x,np.argmax(self.all_hmm[5*j+i,:,:],0))
                ax[i,j].plot(x,states)
                ax[i,j].set_xlim(0,self.all_hmm.shape[2])
                ax[i,j].set_ylim(-.5,3.5)
                ax[i,j].set_xlabel('Data Point')
                ax[i,j].set_ylabel('State')
                ax[i,j].set_title('Target Type '+str(5*j+i))

        fig,ax=plt.subplots(nrows=5,ncols=2,tight_layout=True,figsize=((8,10)))
        x=range(self.all_hmm.shape[2])
        for j in range(2):
            for i in range(5):
                for k in range(4):
                    ax[i,j].plot(x,self.all_hmm[5*j+i,k,:],label=str(k))

                ax[i,j].set_xlim(0,self.all_hmm.shape[2])
                ax[i,j].set_ylim(-.05,1.05)
                ax[i,j].set_xlabel('Data Point')
                ax[i,j].set_ylabel('Probs')
                ax[i,j].set_title('Target Type '+str(5*j+i))
                ax[i,j].legend()

        #  plt.show()
        #  sys.exit()


    def testHMM(self,num_events,num_tar=5):
        if num_tar==10:
            modelFileName = 'hmm_train_10.npy'
            confidenceName = 'hmm_con_10.npy'
        else:
            modelFileName = 'hmm_train.npy'
            confidenceName = 'hmm_con.npy'
        models = np.load(modelFileName).item()

        genNames=[]
        for i in range(num_tar):
            genNames.append('Cumuliform'+str(i))
        correct=0
        confidence=np.zeros((num_tar,num_tar))
        log_like_total=np.zeros((1,2))
        right=np.zeros((1,100))
        wrong=np.zeros((1,100))
        for k in tqdm(range(num_events),ncols=100):
            genus=np.random.randint(num_tar)
            #  genus=0
            species = Cumuliform(genus = genus,weather=False)
            data = species.intensityModel
            if num_tar==10:
                var=0.5
            else:
                var=2
            data=data+np.random.normal(0,var,(len(data)))


            alphas = {}
            for i in genNames:
                alphas[i] = [-1,-1]

            norm_const = np.zeros((num_tar,len(data)))

            probs = {}
            for i in genNames:
                probs[i] = .2
            
            count=0
            log_like=[]
            store_probs=np.zeros((1,num_tar))
            self.all_hmm=np.zeros((num_tar,4,1))
            #  print
            big_alphas=np.zeros((num_tar,4))
            #  while (max(probs.values())<0.9):
            for j in range(len(data)):
                if count<100:
                    #update classification probs
                    big_count=0
                    for i in genNames:
                        #  alphas[i] = self.continueForward(data[count], models[i], alphas[i])
                        alphas[i],norm_const[genNames.index(i),j] = self.continueForward(data[count], models[i], alphas[i])
                        big_alphas[big_count,:]=alphas[i]
                        big_count+=1

                    #normalize probs
                    self.all_hmm=np.append(self.all_hmm,np.expand_dims(big_alphas,2),axis=2)
                    prob_norm=self.expNormalize(norm_const[:,:j+1])
                    #  print prob_norm
                    #  sys.exit()
                    big_count=0
                    #  print norm_const[:,j],genus
                    #  normalized_like=np.prod(norm_const[:,:j+1],axis=1)
                    #  normalized_like=sum(norm_const[:,:j+1])
                    #  print normalized_like
                    for i in genNames:
                        probs[i]=prob_norm[big_count]
                        big_count+=1
                    count+=1
                    store_probs=np.append(store_probs,np.expand_dims(np.array(probs.values()),axis=0),axis=0)

                    #  alpha_total=0
                    #  for i in genNames:
                    #      alpha_total+=np.nansum(alphas[i])/count
                    #  alpha_norm=alpha_total
                    #  #  print log_like
                    #  log_like.append(alpha_norm)
                    log_like.append(sum(norm_const[:,j]))
                else:
                    break
            prob_norm=self.expNormalize(norm_const)
            #  print prob_norm
            #  print norm_const[0,:j+1]
            #  print norm_const[genus,:j+1]
            #  print normalized_like[0],normalized_like[genus]
            #  print np.argmax(prob_norm),genus
            #  sys.exit()
            chosen=np.argmax(probs.values())
            #  print alphas
            #  print genus,chosen
            confidence[genus,chosen]+=1
            #  print np.array([log_like])
            if genus==chosen:
                correct+=1
                right=np.append(right,np.array([log_like]),axis=0)
            else:
                wrong=np.append(wrong,np.array([log_like]),axis=0)
            #  self.graph_HMMs(genus,chosen)
            #  plt.figure()
            #  plt.plot(range(store_probs.shape[0]-1),np.argmax(store_probs[1:],1),'o-')
            #  plt.plot(range(store_probs.shape[0]-1),[genus]*(store_probs.shape[0]-1))
            #  plt.xlim((0,store_probs.shape[0]))
            #  plt.ylim((-0.5,num_tar-.5))
            #  plt.show()
            #  sys.exit()


        confidence=normalize(confidence,axis=1,norm='l1')
        np.save(confidenceName,confidence)
        print confidence
        print (correct/num_events)
        right_mean=np.mean(right,axis=0)
        right_std=np.std(right,axis=0)
        wrong_mean=np.mean(wrong,axis=0)
        wrong_std=np.std(wrong,axis=0)
        plt.figure()
        plt.plot(range(len(right_mean)),right_mean,label='correct')
        plt.fill_between(range(len(right_mean)),right_mean+right_std,right_mean-right_std,alpha=0.5)
        plt.plot(range(len(wrong_mean)),wrong_mean,label='wrong')
        plt.fill_between(range(len(wrong_mean)),wrong_mean+wrong_std,wrong_mean-wrong_std,alpha=0.5)
        plt.legend()

        plt.show()

if __name__ == '__main__':
    hc=HMM_Classification()
    commands=[]
    for i in range(1,len(sys.argv)):
        commands.append(sys.argv[i])

    if 'train' in commands:
        dataSet=hc.buildDataSet(500,10)
        hc.buildModels(dataSet,10)

    if 'test' in commands:
        hc.testHMM(100,10)
