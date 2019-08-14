from __future__ import division

import numpy as np
import scipy.stats
from scipy.special import gamma, psi, polygamma
import random
import copy
import math
import sys
import os
import itertools
import warnings
import time
from tqdm import tqdm
np.set_printoptions(precision=2)

from data_sim.DynamicsProfiles import *
from HMM.hmm_classification import HMM_Classification
from gaussianMixtures import GM

warnings.filterwarnings("ignore",category=RuntimeWarning)
warnings.filterwarnings("ignore",category=UserWarning)

__author__ = "Jeremy Muesing"
__version__ = "2.3.0"
__maintainer__ = "Jeremy Muesing"
__email__ = "jeremy.muesing@colorado.edu"
__status__ = "maintained"

class Human():
    def DirPrior(self,num_tar,human="good"):
        #init for confusion matrix
        self.pred_obs=[]
        self.real_obs=[]
        self.human_correct=[]

        #theta 1
        self.table=[5,2,0.5,8]
        self.theta1=copy.deepcopy(self.table)
        #  self.theta1=scipy.stats.dirichlet.mean(alpha=self.table)
        table_real=self.table+np.random.uniform(-1,1,4)
        table_real[table_real<0]=0.1
        self.theta1_correct=scipy.stats.dirichlet.mean(alpha=self.table)

        #theta2 param tied
        self.theta2=np.zeros((4,4))
        for i in range(4):
            self.theta2[i,:]=self.table
        for i in range(4):
            self.theta2[i,i]*=2

        #theta2 prior full case
        table_full=np.zeros((num_tar,2*num_tar,2*num_tar))
        base_table=np.ones((num_tar,2*num_tar))
        for i in range(num_tar):
            base_table[i,2*i]*=5
            for j in range(num_tar):
                if i==j:
                    base_table[i,2*j+1]*=0.5
                else:
                    base_table[i,2*j+1]*=2
                    base_table[i,2*j]*=0.5
        for i in range(2*num_tar):
            table_full[:,:,i]=base_table
        for i in range(2*num_tar):
            table_full[:,i,i]*=3
        table_full=np.swapaxes(table_full,1,2)
        
        self.theta1_full=base_table
        self.theta1_ind=base_table
        self.theta2_full=table_full

        # theta2 real
        # note: this is the only one with real theta distributions, the above are alphas
        table_real=np.zeros((num_tar,2*num_tar,2*num_tar))
        base_table_real=np.ones((num_tar,2*num_tar))
        base_table_real*=5
        human_rates=np.array([[10,0.4,1.67,0.4],[9,0.7,2,0.9],[8,1,2.7,1.3],[7,1.2,3.4,1.7],[6,1.3,4,2.1],[5,1.5,5,2.5]])
        for i in range(num_tar):
            #tp
            if num_tar==10:
                base_table_real[i,2*i]*=1.2*human_rates[human][0]
            else:
                base_table_real[i,2*i]*=human_rates[human][0]
            for j in range(num_tar):
                if i==j:
                    #fn
                    base_table_real[i,2*j+1]*=human_rates[human][1]
                else:
                    #tn
                    if num_tar==10:
                        base_table_real[i,2*j+1]*=1.6*human_rates[human][2]
                    else:
                        base_table_real[i,2*j+1]*=human_rates[human][2]
                    #fp
                    base_table_real[i,2*j]*=human_rates[human][3]
            #  if human=='good':
            #      #tp
            #      base_table_real[i,2*i]*=10
            #      for j in range(5):
            #          if i==j:
            #              #fn
            #              base_table_real[i,2*j+1]*=0.4
            #          else:
            #              #tn
            #              base_table_real[i,2*j+1]*=1.67
            #              #fp
            #              base_table_real[i,2*j]*=0.4
            #  elif human=='bad':
            #      #tp
            #      base_table_real[i,2*i]*=5
            #      for j in range(5):
            #          if i==j:
            #              #fn
            #              base_table_real[i,2*j+1]*=1.5
            #          else:
            #              #tn
            #              base_table_real[i,2*j+1]*=5
            #              #fp
            #              base_table_real[i,2*j]*=2.5
        #  print base_table_real
        for i in range(2*num_tar):
            table_real[:,:,i]=base_table_real
        for i in range(2*num_tar):
            #repeat
            table_real[:,i,i]*=3
        table_real=np.swapaxes(table_real,1,2)
        table_real+=np.random.uniform(-1,1,(num_tar,2*num_tar,2*num_tar))
        table_real[table_real<0]=0.1
        #  print table_real.shape
        #  sys.exit()
        self.theta2_correct=np.zeros((2*num_tar*num_tar,2*num_tar))
        #  self.table_compare=np.zeros((2*num_tar*num_tar,2*num_tar))
        for X in range(num_tar):
            for prev_obs in range(2*num_tar):
                self.theta2_correct[X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=table_real[X,prev_obs,:])
                #  self.table_compare[X*2*num_tar+prev_obs,:]=table_real[X,prev_obs,:]

    def HumanObservations(self,num_tar,real_target,obs,real=True):
        if len(obs)>0:
            prev_obs=obs[-1]
            obs.append(np.random.choice(range(2*num_tar),p=self.theta2_correct[real_target*2*num_tar+prev_obs,:]))
            #DEBUG
            #  print real_target
        else:
            obs_type=np.random.choice(range(4),p=self.theta1_correct)
            #tp
            if obs_type==0:
                obs.append(2*real_target)
            #fn
            elif obs_type==2:
                obs.append(2*real_target+1)
            else:
                choices=range(2*num_tar)
                # the first gets rid of the tp, 2nd the fn
                del choices[2*real_target]
                del choices[2*real_target]
                #fp
                if obs_type==1:
                    for i in range(num_tar):
                        if i!=real_target:
                            choices.remove(2*i+1)
                    obs.append(np.random.choice(choices))
                #tn
                elif obs_type==3:
                    for i in range(num_tar):
                        if i!=real_target:
                            choices.remove(2*i)
                    obs.append(np.random.choice(choices))

        # confusion matrix for human
        if real:
            if obs[-1]%2==0:
                self.pred_obs.append(0)
                if (obs[-1]/2)==real_target:
                    self.real_obs.append(0)
                    self.human_correct.append(1)
                else:
                    self.real_obs.append(1)
                    self.human_correct.append(0)
            else:
                self.pred_obs.append(1)
                if (int(obs[-1]/2))==real_target:
                    self.real_obs.append(0)
                    self.human_correct.append(0)
                else:
                    self.real_obs.append(1)
                    self.human_correct.append(1)

        return obs

    def HumanAnswer(self,num_tar,real_target,obs,theta1,theta2):
        def pick_obs(obs,obs_type,real_target,num_tar):
            #tp
            if obs_type==0:
                obs.append(2*real_target)
            #fn
            elif obs_type==2:
                obs.append(2*real_target+1)
            else:
                choices=range(2*num_tar)
                # the first gets rid of the tp, 2nd the fn
                del choices[2*real_target]
                del choices[2*real_target]
                #fp
                if obs_type==1:
                    for i in range(num_tar):
                        if i!=real_target:
                            choices.remove(2*i+1)
                    obs.append(np.random.choice(choices))
                #tn
                elif obs_type==3:
                    for i in range(num_tar):
                        if i!=real_target:
                            choices.remove(2*i)
                    obs.append(np.random.choice(choices))
            return obs

        if len(obs)==0:
            obs_type=np.random.choice(range(4),p=theta1)
            obs=pick_obs(obs,obs_type,real_target,num_tar)
        else:
            prev_obs=obs[-1]
            if prev_obs/2==real_target:
                obs_type=np.random.choice(range(4),p=theta2[0,:])
            elif math.floor(prev_obs/2)==real_target:
                obs_type=np.random.choice(range(4),p=theta2[1,:])
            elif prev_obs%2==0:
                obs_type=np.random.choice(range(4),p=theta2[2,:])
            elif prev_obs%2==1:
                obs_type=np.random.choice(range(4),p=theta2[3,:])
            obs=pick_obs(obs,obs_type,real_target,num_tar)

        return obs

    def HumanAnswer_full(self,num_tar,real_target,obs,theta1,theta2):
        if len(obs)==0:
            prob=theta1[real_target,:]
            obs=[np.random.choice(range(2*num_tar),p=prob)]
        else:
            prev_obs=obs[-1]
            prob=theta2[real_target*2*num_tar+prev_obs,:]
            obs.append(np.random.choice(range(2*num_tar),p=prob))
        return obs

class DataFusion(Human):
    def __init__(self,num_tar):
        self.hmm=HMM_Classification()
        #  self.num_samples=5000
        #  self.burn_in=1000
        if num_tar==10:
            modelFileName = 'HMM/hmm_train_10.npy'
        else:
            modelFileName = 'HMM/hmm_train.npy'
        self.hmm_models = np.load(modelFileName).item()
        self.names=[]
        for i in range(num_tar):
            self.names.append('Cumuliform'+str(i))
        #  self.names = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']
        self.alphas={}
        for i in self.names:
            self.alphas[i] = [-1,-1]
        self.sampling_data=True
        self.norm_const=np.zeros((num_tar,100))
        if num_tar==10:
            self.confidence=np.load('HMM/hmm_con_10.npy')
        else:
            self.confidence=np.load('HMM/hmm_con.npy')

    def make_data(self,genus,num_tar,graph=False):
        model=Cumuliform(genus=genus,weather=False)
        if num_tar==5:
            intensity_data=model.intensityModel+np.random.normal(0,2,(len(model.intensityModel)))
        elif num_tar==10:
            intensity_data=model.intensityModel+np.random.normal(0,0.5,(len(model.intensityModel)))
        for j in range(len(intensity_data)):
            intensity_data[j]=max(intensity_data[j],1e-5)
        self.intensity_data=intensity_data

        if graph:
            # without noise
            plt.figure()
            for genus in range(5):
                model=Cumuliform(genus=genus,weather=False)
                intensity_data=model.intensityModel
                plt.plot(range(100),intensity_data,label=genus)
            plt.xlabel('Time (frames)')
            plt.ylabel('Intensity (Units)')
            plt.title('Family: Cumuliform')
            plt.legend()
            plt.show()

            # with noise
            plt.figure()
            for genus in range(5):
                model=Cumuliform(genus=genus,weather=False)
                intensity_data=model.intensityModel+np.random.normal(0,2,(len(model.intensityModel)))
                for j in range(len(intensity_data)):
                    intensity_data[j]=max(intensity_data[j],1e-5)
                plt.plot(range(100),intensity_data,label=genus)
            plt.xlabel('Time (frames)')
            plt.ylabel('Intensity (Units)')
            plt.title('Family: Cumuliform')
            plt.legend()
            plt.show()

    def updateProbsML(self):
        probs={}
        data=self.intensity_data[self.frame]
        #forward algorithm
        for i in self.names:
            #  self.alphas[i]=self.hmm.continueForward(data,self.hmm_models[i],self.alphas[i])
            self.alphas[i],self.norm_const[self.names.index(i),self.frame] = self.hmm.continueForward(data, self.hmm_models[i], self.alphas[i])
        #  self.probs[i]=self.probs[i]*sum(self.alphas[i])
        prob_norm=self.hmm.expNormalize(self.norm_const[:,:self.frame+1])
        for i in self.names:
            probs[i]=prob_norm[self.names.index(i)]
        return probs
        #noramlize
        #  suma=sum(self.probs.values())
        #  for i in self.names:
        #      self.probs[i]/=suma

    def sampling_full(self,num_tar,obs,num_samples=5000,burn_in=1000):
        postX=copy.deepcopy(self.probs)
        # only learning theta2 on 2+ observations
        if len(obs)>1:
            # initialize Dir sample
            theta1_static=np.empty((num_tar,2*num_tar))
            theta2_static=np.empty((2*num_tar*num_tar,2*num_tar))
            all_post=np.zeros((int((num_samples-burn_in)/5),1,num_tar))
            self.all_theta1=np.zeros((int((num_samples-burn_in)/5),num_tar,2*num_tar))
            self.all_theta2=np.zeros((int((num_samples-burn_in)/5),2*num_tar*num_tar,2*num_tar))
            for X in range(num_tar):
                theta1_static[X,:]=scipy.stats.dirichlet.mean(alpha=self.theta1_full[X,:])
                for prev_obs in range(2*num_tar):
                    theta2_static[X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=self.theta2_full[X,prev_obs,:])

            # begin gibbs sampling
            theta2=copy.copy(theta2_static)
            theta1=copy.copy(theta1_static)
            for n in range(num_samples):
                # calc X as if we knew theta2
                for i in self.names:
                    # likelihood from theta1
                    likelihood=theta1[self.names.index(i),obs[0]]
                    # likelihood from theta2
                    for value in obs[1:]:
                        likelihood*=theta2[self.names.index(i)*2*num_tar+obs[obs.index(value)-1],value]
                    postX[i]=self.probs[i]*likelihood
                # normalize
                suma=sum(postX.values())
                for i in self.names:
                    postX[i]=np.log(postX[i])-np.log(suma) 
                    postX[i]=np.exp(postX[i])
                # store every 5th sample
                if n%5==0:
                    all_post[int((n-burn_in)/5),:,:]=postX.values()
                # sample from X
                X=np.random.choice(range(num_tar),p=postX.values())
                alphas1=copy.copy(self.theta1_full)
                alphas2=copy.copy(self.theta2_full)
                theta1=copy.copy(theta1_static)
                theta2=copy.copy(theta2_static)
                # clac theta1 as if we knew X
                alphas1[X,obs[0]]+=1
                theta1[X,:]=np.random.dirichlet(alphas1[X,:])
                # calc theta2 as if we knew X
                for i in range(len(obs)-1):
                    alphas2[X,obs[i],obs[i+1]]+=1
                for j in range(theta2.shape[1]):
                    theta2[X*2*num_tar+j,:]=np.random.dirichlet(alphas2[X,j,:])
                if n%5==0:
                    self.all_theta1[int((n-burn_in)/5),:,:]=theta1
                    self.all_theta2[int((n-burn_in)/5),:,:]=theta2


            # take max likelihood of X for next obs
            post_probs=np.mean(all_post,axis=0)
            return post_probs[0]

        # using only theat1 on first observation
        else:
            theta1=np.empty((num_tar,2*num_tar))
            for X in range(num_tar):
                theta1[X,:]=scipy.stats.dirichlet.mean(alpha=self.theta1_full[X,:])
            for i in self.names:
                # likelihood from theta1 (not full dist, assuming we know theta1)
                likelihood=theta1[self.names.index(i),obs[0]]
                postX[i]=self.probs[i]*likelihood
            # normalize and set final values
            suma=sum(postX.values())
            for i in self.names:
                postX[i]=np.log(postX[i])-np.log(suma) 
                postX[i]=np.exp(postX[i])
            return postX.values()

    def moment_matching_full(self,num_tar):
        # moment matching of alphas from samples (Minka, 2000)
        sample_counts=np.zeros((2*num_tar*num_tar,2*num_tar))
        for n in range(self.all_theta2.shape[1]):
            sum_alpha=sum(self.theta2_full[int(n/(2*num_tar)),n%(2*num_tar),:])
            for k in range(self.all_theta2.shape[2]):
                samples=self.all_theta2[:,n,k]
                if len(samples)==0:
                    pass
                else:
                    sample_counts[n,k]=len(samples)
                    current_alpha=self.theta2_full[int(n/(2*num_tar)),n%(2*num_tar),k]
                    for x in range(5):
                        sum_alpha_old=sum_alpha-current_alpha+self.theta2_full[int(n/(2*num_tar)),n%(2*num_tar),k]
                        logpk=np.sum(np.log(samples))/len(samples)
                        y=psi(sum_alpha_old)+logpk
                        if y>=-2.22:
                            alphak=np.exp(y)+0.5
                        else:
                            alphak=-1/(y+psi(1))
                        for w in range(5):
                            alphak-=((psi(alphak)-y)/polygamma(1,alphak))
                        self.theta2_full[int(n/(2*num_tar)),n%(2*num_tar),k]=alphak

    def moment_matching_full_small(self,num_tar):
        # moment matching of alphas from samples (Minka, 2000)
        sample_counts=np.zeros((num_tar,2*num_tar))
        for n in range(self.all_theta1.shape[1]):
            sum_alpha=sum(self.theta1_full[int(n/(2*num_tar)),:])
            for k in range(self.all_theta1.shape[2]):
                samples=self.all_theta1[:,n,k]
                if len(samples)==0:
                    pass
                else:
                    sample_counts[n,k]=len(samples)
                    current_alpha=self.theta1_full[int(n/(2*num_tar)),k]
                    for x in range(5):
                        sum_alpha_old=sum_alpha-current_alpha+self.theta1_full[int(n/(2*num_tar)),k]
                        logpk=np.sum(np.log(samples))/len(samples)
                        y=psi(sum_alpha_old)+logpk
                        if y>=-2.22:
                            alphak=np.exp(y)+0.5
                        else:
                            alphak=-1/(y+psi(1))
                        for w in range(5):
                            alphak-=((psi(alphak)-y)/polygamma(1,alphak))
                        self.theta1_full[int(n/(2*num_tar)),k]=alphak

    def sampling_param_tied(self,num_tar,obs,num_samples=5000,burn_in=1000):
        postX=copy.deepcopy(self.probs)
        # only learning theta2 on 2+ observations
        if len(obs)>1:
            # initialize Dir sample
            theta1_static=np.empty((1,4))
            theta2_static=np.empty((4,4))
            all_post=np.zeros((int((num_samples-burn_in)/5),1,num_tar))
            self.theta1_samples=np.zeros((int((num_samples-burn_in)/5),4))
            self.theta2_samples=np.zeros((int((num_samples-burn_in)/5),4,4))
            theta1_static=scipy.stats.dirichlet.mean(alpha=self.theta1)
            for i in range(4):
                theta2_static[i,:]=scipy.stats.dirichlet.mean(alpha=self.theta2[i,:])

            # begin gibbs sampling
            theta1=copy.copy(theta1_static)
            theta2=copy.copy(theta2_static)
            for n in range(num_samples):
                # calc X as if we knew theta2
                for i in self.names:
                    # lieklihood from theta1
                    index=self.select_param(self.names.index(i),obs[0])
                    if index%2==0:
                        likelihood=theta1[index]
                    else:
                        likelihood=theta1[index]/(num_tar-1)
                    # likelihood from theta2
                    count=0
                    for value in obs[1:]:
                        indicies=self.select_param(self.names.index(i),value,obs[count])
                        if indicies[1]%2==0:
                            likelihood*=theta2[indicies[0],indicies[1]]
                        else:
                            likelihood*=(theta2[indicies[0],indicies[1]]/(num_tar-1))
                        count+=1
                    postX[i]=self.probs[i]*likelihood
                # normalize
                suma=sum(postX.values())
                for i in self.names:
                    postX[i]=np.log(postX[i])-np.log(suma) 
                    postX[i]=np.exp(postX[i])
                # store every 5th sample
                if n%5==0:
                    all_post[int((n-burn_in)/5),:,:]=postX.values()
                # sample from X
                X=np.random.choice(range(num_tar),p=postX.values())
                alphas1=copy.copy(self.theta1)
                alphas2=copy.copy(self.theta2)
                theta1=copy.copy(theta1_static)
                theta2=copy.copy(theta2_static)
                # calc theta1 as i we knew it
                alphas1[self.select_param(X,obs[0])]+=1
                theta1=np.random.dirichlet(alphas1)
                # calc theta2 as is we knew X
                for i in range(len(obs)-1):
                    indicies=self.select_param(X,obs[i+1],obs[i])
                    alphas2[indicies[0],indicies[1]]+=1
                for j in range(4):
                    theta2[j,:]=np.random.dirichlet(alphas2[j,:])
                if n%5==0:
                    self.theta1_samples[int((n-burn_in)/5),:]=theta1
                    self.theta2_samples[int((n-burn_in)/5),:,:]=theta2

            # storing data for graphs
            if max(postX.values())<0.5:
                self.X_samples=all_post

            # take max likelihood of X for next obs
            post_probs=np.mean(all_post,axis=0)
            #DEBUG
            #  print obs
            #  print post_probs[0]
            return post_probs[0]

        # using only theat1 on first observation
        else:
            theta1=scipy.stats.dirichlet.mean(alpha=self.theta1)
            for i in self.names:
                # likelihood from theta1 (not full dist, assuming we know theta1)
                index=self.select_param(self.names.index(i),obs[0])
                likelihood=theta1[index]
                postX[i]=self.probs[i]*likelihood
            # normalize and set final values
            suma=sum(postX.values())
            for i in self.names:
                postX[i]=np.log(postX[i])-np.log(suma) 
                postX[i]=np.exp(postX[i])
            return postX.values()

    def moment_matching(self,graph=False):
        # moment matching of alphas from samples (Minka, 2000)
        if graph:
            fig,ax=plt.subplots(nrows=1,ncols=2,figsize=((8,4)),tight_layout=True)
        sample_counts=np.zeros((4,4))
        for n in range(4):
            sum_alpha=sum(self.theta2[n,:])
            for k in range(4):
                samples=self.theta2_samples[:,n,k]
                if len(samples)==0:
                    pass
                else:
                    sample_counts[n,k]=len(samples)
                    current_alpha=self.theta2[n,k]
                    for x in range(5):
                        sum_alpha_old=sum_alpha-current_alpha+self.theta2[n,k]
                        logpk=np.sum(np.log(samples))/len(samples)
                        y=psi(sum_alpha_old)+logpk
                        if y>=-2.22:
                            alphak=np.exp(y)+0.5
                        else:
                            alphak=-1/(y+psi(1))
                        for w in range(5):
                            alphak-=((psi(alphak)-y)/polygamma(1,alphak))
                        self.theta2[n,k]=alphak
                    if graph:
                        if (n==0) and (k==0):
                            ax[0].hist(samples,bins=20,density=True)
                            x=np.linspace(0,1)
                            ax[0].plot(x,scipy.stats.beta.pdf(x,alphak,sum(self.theta2[n,:])-alphak))
                            ax[0].set_xlabel(r'$\theta_2$')
                            ax[0].set_ylabel(r'$p(\theta_2)$')
                            ax[0].set_title("Moment Matching for TP,TP")
                        elif (n==1) and (k==0):
                            ax[1].hist(samples,bins=20,density=True)
                            x=np.linspace(0,1)
                            ax[1].plot(x,scipy.stats.beta.pdf(x,alphak,sum(self.theta2[n,:])-alphak))
                            ax[1].set_xlabel(r'$\theta_2$')
                            ax[1].set_ylabel(r'$p(\theta_2)$')
                            ax[1].set_title("Moment Matching for FP,TP")
        if graph:
            fig.savefig('figures/moment_matching.png',bbox_inches='tight',pad_inches=0)

    def moment_matching_small(self):
        # moment matching of alphas from samples (Minka, 2000)
        sample_counts=np.zeros((1,4))
        #  for n in range(4):
        sum_alpha=sum(self.theta1)
        for k in range(4):
            samples=self.theta1_samples[:,k]
            if len(samples)==0:
                pass
            else:
                sample_counts[0,k]=len(samples)
                current_alpha=self.theta1[k]
                for x in range(5):
                    sum_alpha_old=sum_alpha-current_alpha+self.theta1[k]
                    logpk=np.sum(np.log(samples))/len(samples)
                    y=psi(sum_alpha_old)+logpk
                    if y>=-2.22:
                        alphak=np.exp(y)+0.5
                    else:
                        alphak=-1/(y+psi(1))
                    for w in range(5):
                        alphak-=((psi(alphak)-y)/polygamma(1,alphak))
                    self.theta1[k]=alphak

    def sampling_ind(self,num_tar,obs,num_samples=5000,burn_in=1000):
        postX=copy.deepcopy(self.probs)
        # initialize Dir sample
        theta1_static=np.empty((num_tar,2*num_tar))
        all_post=np.zeros((int((num_samples-burn_in)/5),1,num_tar))
        self.theta1_ind_samples=np.zeros((int((num_samples-burn_in)/5),num_tar,2*num_tar))
        for X in range(num_tar):
            theta1_static[X,:]=scipy.stats.dirichlet.mean(alpha=self.theta1_ind[X,:])

        # begin gibbs sampling
        theta1=copy.copy(theta1_static)
        for n in range(num_samples):
            # calc X as if we knew theta2
            for i in self.names:
                likelihood=1
                for value in obs:
                    # likelihood from theta1
                    likelihood*=theta1[self.names.index(i),value]
                postX[i]=self.probs[i]*likelihood
            # normalize
            suma=sum(postX.values())
            for i in self.names:
                postX[i]=np.log(postX[i])-np.log(suma) 
                postX[i]=np.exp(postX[i])
            # store every 5th sample
            if n%5==0:
                all_post[int((n-burn_in)/5),:,:]=postX.values()
            # sample from X
            X=np.random.choice(range(num_tar),p=postX.values())
            alphas1=copy.copy(self.theta1_ind)
            theta1=copy.copy(theta1_static)
            # clac theta1 as if we knew X
            for i in range(len(obs)):
                alphas1[X,obs[i]]+=1
            theta1[X,:]=np.random.dirichlet(alphas1[X,:])
            if n%5==0:
                self.theta1_ind_samples[int((n-burn_in)/5),:]=theta1

        # take max likelihood of X for next obs
        post_probs=np.mean(all_post,axis=0)
        return post_probs[0]

    def moment_matching_ind(self,num_tar):
        # moment matching of alphas from samples (Minka, 2000)
        sample_counts=np.zeros((num_tar,2*num_tar))
        for n in range(self.theta1_ind_samples.shape[1]):
            sum_alpha=sum(self.theta1_ind[int(n/(2*num_tar)),:])
            for k in range(self.theta1_ind_samples.shape[2]):
                samples=self.theta1_ind_samples[:,n,k]
                if len(samples)==0:
                    pass
                else:
                    sample_counts[n,k]=len(samples)
                    current_alpha=self.theta1_ind[int(n/(2*num_tar)),k]
                    for x in range(5):
                        sum_alpha_old=sum_alpha-current_alpha+self.theta1_ind[int(n/(2*num_tar)),k]
                        logpk=np.sum(np.log(samples))/len(samples)
                        y=psi(sum_alpha_old)+logpk
                        if y>=-2.22:
                            alphak=np.exp(y)+0.5
                        else:
                            alphak=-1/(y+psi(1))
                        for w in range(5):
                            alphak-=((psi(alphak)-y)/polygamma(1,alphak))
                        self.theta1_ind[int(n/(2*num_tar)),k]=alphak

    def select_param(self,target,current_obs,prev_obs=None):
        # translate an observation about a target into its type of obs
        def select_index(tar,obs):
            if tar*2==obs:
                #tp
                index=0
            elif obs%2==0:
                #fp
                index=1
            if tar*2+1==obs:
                #fn
                index=2
            elif obs%2==1:
                #tn
                index=3
            return index
        if (prev_obs) or (prev_obs==0):
            index1=select_index(target,prev_obs)
            index2=select_index(target,current_obs)
            return [index1,index2]
        else:
            index=select_index(target,current_obs)
            return index


    def VOI(self,num_tar,obs,threshold):
        post=copy.deepcopy(self.probs.values())
        R=np.zeros((num_tar,num_tar*2))
        VOI=np.zeros(num_tar)
        obs_probs=np.empty((2*num_tar*num_tar,2*num_tar))
        # create our p(o'|o,X,theta_2)
        for X in range(num_tar):
            for prev_obs in range(2*num_tar):
                obs_probs[X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=self.theta2_full[X,prev_obs,:])
        # we must marginalize out the target types
        sum_tar=np.zeros(2*num_tar)
        for X in range(num_tar):
            # don't forget to mul by our probs
            sum_tar=np.sum([sum_tar,post[X]*obs_probs[obs[-1]+X*2*num_tar,:]],axis=0)
        # normalize
        obs_probs_no_state=sum_tar/np.sum(sum_tar)

        # small samples of what would happend if an observation was really given
        for i in range(num_tar*2):
            theory_obs=copy.copy(obs)
            theory_obs.append(i)
            post=self.sampling_param_tied(num_tar,theory_obs,150,10)
            # reward if it gets it right, punish if wrong
            if max(post)>threshold:
                if i%2==1:
                    R[:,i]=-num_tar
                    R[np.argmax(post),i]=10*num_tar
                # prefer affirmative classification
                else:
                    R[:,i]=-.5*num_tar
                    R[np.argmax(post),i]=20*num_tar
        #  print R

        # expected reward if the human gave any observation
        E_no_obs=0
        for n in range(num_tar):
            E_no_obs+=np.sum(np.multiply(R[n,:],obs_probs_no_state))
        # sum over the possible answers for each target, ask regardless of answer
        R_act=np.zeros((num_tar,num_tar))
        for n in range(num_tar):
            R_act[:,n]=np.sum([R[:,2*n],R[:,2*n+1]])
        # expected reward if we make the human talk about a single target
        for n in range(num_tar):
            E_with_obs=(obs_probs_no_state[2*n]+obs_probs_no_state[2*n+1])*np.sum(R_act,axis=0)[n]
            VOI[n]=E_with_obs-E_no_obs
        #  print VOI
        if max(VOI)>0:
            return np.argmax(VOI)
        else:
            return None

    def VOI_thetas(self):
        theta2=np.empty((4,4))
        theta1=scipy.stats.dirichlet.mean(alpha=self.theta1)
        for i in range(4):
            theta2[i,:]=scipy.stats.dirichlet.mean(alpha=self.theta2[i,:])
        #  theta1=np.empty((num_tar,2*num_tar))
        #  theta2=np.empty((2*num_tar*num_tar,2*num_tar))
        #  for X in range(num_tar):
        #      theta1[X,:]=scipy.stats.dirichlet.mean(alpha=self.theta1_full[X,:])
        #      for prev_obs in range(2*num_tar):
        #          theta2[X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=self.theta2_full[X,prev_obs,:])
        return [theta1,theta2]

    def VOI_tied(self,num_tar,threshold,post,theta1,theta2):
        num_samples=100

        right=1
        wrong=-1
        R_human=np.zeros(num_tar)
        R_bot=np.zeros(num_tar)
        #  print self.confidence
        for X in range(num_tar):
            # HMM sample
            classification=np.random.choice(range(num_tar),size=num_samples,p=self.confidence[X])
            for n in range(num_samples):
                # human sample
                obs=[]
                post_sample=copy.copy(post)
                while max(post_sample.values())<threshold:
                    if len(obs)==0:
                        obs=self.HumanAnswer(num_tar,X,obs,theta1,theta2)
                    else:
                        obs=self.HumanAnswer(num_tar,X,obs,theta1,theta2)
                    for i in self.names:
                        if len(obs)==1:
                            if obs[0]/2==self.names.index(i):
                                post_sample[i]*=theta1[0]
                            elif math.floor(obs[0]/2)==self.names.index(i):
                                post_sample[i]*=theta1[1]
                            elif obs[0]%2==0:
                                post_sample[i]*=theta1[2]
                            elif obs[0]%2==1:
                                post_sample[i]*=theta1[3]
                        else:
                            if obs[-2]/2==self.names.index(i):
                                ind_0=0
                            elif math.floor(obs[-2]/2)==self.names.index(i):
                                ind_0=1
                            elif obs[-2]%2==0:
                                ind_0=2
                            elif obs[-2]%2==1:

                                ind_0=3
                            if obs[-1]/2==self.names.index(i):
                                ind_1=0
                            elif math.floor(obs[-1]/2)==self.names.index(i):
                                ind_1=1
                            elif obs[-1]%2==0:
                                ind_1=2
                            elif obs[-1]%2==1:
                                ind_1=3
                            post_sample[i]*=theta2[ind_0,ind_1]
                    # normalize
                    suma=sum(post_sample.values())
                    for i in self.names:
                        post_sample[i]=np.log(post_sample[i])-np.log(suma) 
                        post_sample[i]=np.exp(post_sample[i])
                if np.argmax(post_sample.values())==X:
                    R_human[X]+=right
                else:
                    R_human[X]+=wrong
                if classification[n]==X:
                    R_bot[X]+=right
                else:
                    R_bot[X]+=wrong
        #  print R_human*post.values()
        #  print R_bot*post.values()
        #  print sum(R_human*post.values())/num_samples,sum(R_bot*post.values())/num_samples
        VOI=(sum(R_human*post.values())/num_samples)-(sum(R_bot*post.values())/num_samples)
        #  print VOI
        #  sys.exit()
        if VOI>.45:
            return 1
        else:
            return 0

    def VOI_full(self,num_tar,threshold,post,theta1,theta2):
        num_samples=100

        right=1
        wrong=-1
        R_human=np.zeros(num_tar)
        R_bot=np.zeros(num_tar)
        #  print self.confidence
        for X in range(num_tar):
            # HMM sample
            classification=np.random.choice(range(num_tar),size=num_samples,p=self.confidence[X])
            for n in range(num_samples):
                # human sample
                obs=[]
                post_sample=copy.copy(post)
                while max(post_sample.values())<threshold:
                    if len(obs)==0:
                        obs=self.HumanAnswer_full(num_tar,X,obs,theta1,theta2)
                    else:
                        obs=self.HumanAnswer_full(num_tar,X,obs,theta1,theta2)
                    for i in self.names:
                        if len(obs)==1:
                            post_sample[i]*=theta1[self.names.index(i),obs[0]]
                        else:
                            post_sample[i]*=theta2[self.names.index(i)*2*num_tar+obs[-2],obs[-1]]
                    # normalize
                    suma=sum(post_sample.values())
                    for i in self.names:
                        post_sample[i]=np.log(post_sample[i])-np.log(suma) 
                        post_sample[i]=np.exp(post_sample[i])
                if np.argmax(post_sample.values())==X:
                    R_human[X]+=right
                else:
                    R_human[X]+=wrong
                if classification[n]==X:
                    R_bot[X]+=right
                else:
                    R_bot[X]+=wrong
        #  print R_human*post.values()
        #  print R_bot*post.values()
        #  print sum(R_human*post.values())/num_samples,sum(R_bot*post.values())/num_samples
        VOI=(sum(R_human*post.values())/num_samples)-(sum(R_bot*post.values())/num_samples)
        #  print VOI
        #  sys.exit()
        if VOI>.45:
            return 1
        else:
            return 0

             
if __name__ == '__main__':
    num_tar=10
    a=DataFusion(num_tar)
    a.DirPrior(num_tar)
    #  print a.HumanAnswer2(5,2,[0,4])
    a.probs={}
    for i in a.names:
        a.probs[i]=1/num_tar
    for i in tqdm(range(100),ncols=100):
        a.VOI2(num_tar,0.9,a.probs)
