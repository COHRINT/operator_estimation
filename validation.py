from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mgimg
import scipy.stats
from scipy.special import gamma, psi, polygamma
import random
import copy
import sys
import os
import itertools
import warnings
import time
import yaml
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
np.set_printoptions(precision=2)

from data_sim.DynamicsProfiles import *
from HMM.hmm_classification import HMM_Classification
from gaussianMixtures import GM
from graphing import Graphing

warnings.filterwarnings("ignore",category=RuntimeWarning)
warnings.filterwarnings("ignore",category=UserWarning)

__author__ = "Jeremy Muesing"
__version__ = "2.2.0"
__maintainer__ = "Jeremy Muesing"
__email__ = "jeremy.muesing@colorado.edu"
__status__ = "maintained"

class Human():

    def DirPrior(self,num_tar):
        #init for confusion matrix
        self.pred_obs=[]
        self.real_obs=[]

        #theta 1
        self.table=[5,2,0.5,8]
        self.theta1=scipy.stats.dirichlet.mean(alpha=self.table)

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

        self.theta2_full=table_full

        # theta2 real
        # note: this is the only one with real theta distributions, the above are alphas
        table_real=np.zeros((num_tar,2*num_tar,2*num_tar))
        base_table_real=np.ones((num_tar,2*num_tar))
        base_table_real*=5
        for i in range(5):
            #tp
            base_table_real[i,2*i]*=10
            for j in range(5):
                if i==j:
                    #fn
                    base_table_real[i,2*j+1]*=0.4
                else:
                    #tn
                    base_table_real[i,2*j+1]*=1.67
                    #fp
                    base_table_real[i,2*j]*=0.4
        for i in range(2*num_tar):
            table_real[:,:,i]=base_table_real
        for i in range(2*num_tar):
            #repeat
            table_real[:,i,i]*=3
        table_real=np.swapaxes(table_real,1,2)
        table_real+=np.random.uniform(-1,1,(num_tar,2*num_tar,2*num_tar))
        table_real[table_real<0]=0.1
        self.theta2_correct=np.zeros((2*num_tar*num_tar,2*num_tar))
        #  self.table_compare=np.zeros((2*num_tar*num_tar,2*num_tar))
        for X in range(num_tar):
            for prev_obs in range(2*num_tar):
                self.theta2_correct[X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=table_real[X,prev_obs,:])
                #  self.table_compare[X*2*num_tar+prev_obs,:]=table_real[X,prev_obs,:]

    def HumanObservations(self,num_tar,real_target,obs):
        if len(obs)>0:
            prev_obs=obs[-1]
            obs.append(np.random.choice(range(2*num_tar),p=self.theta2_correct[real_target*2*num_tar+prev_obs,:]))
            #DEBUG
            #  print real_target
        else:
            obs_type=np.random.choice(range(4),p=self.theta1)
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
        if obs[-1]%2==0:
            self.pred_obs.append(0)
            if (obs[-1]/2)==real_target:
                self.real_obs.append(0)
            else:
                self.real_obs.append(1)
        else:
            self.pred_obs.append(1)
            if (int(obs[-1]/2))==real_target:
                self.real_obs.append(0)
            else:
                self.real_obs.append(1)

        return obs


class DataFusion(Human):
    def __init__(self):
        self.hmm=HMM_Classification()
        self.num_samples=5000
        self.burn_in=1000
        modelFileName = 'HMM/hmm_train.npy'
        self.hmm_models = np.load(modelFileName).item()
        self.names = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']
        self.alphas={}
        self.sampling_data=True

    def make_data(self,genus):
        model=Cumuliform(genus=genus,weather=False)
        intensity_data=model.intensityModel+np.random.normal(0,2,(len(model.intensityModel)))
        for j in range(len(intensity_data)):
            intensity_data[j]=max(intensity_data[j],1e-5)
        self.intensity_data=intensity_data

    def updateProbsML(self):
        data=self.intensity_data[self.frame]
        #forward algorithm
        for i in self.names:
            self.alphas[i]=self.hmm.continueForward(data,self.hmm_models[i],self.alphas[i])
            self.probs[i]=self.probs[i]*sum(self.alphas[i])
        #noramlize
        suma=sum(self.probs.values())
        for i in self.names:
            self.probs[i]/=suma

    def sampling_full(self,num_tar,obs):
        postX=copy.deepcopy(self.probs)
        # only learning theta2 on 2+ observations
        if len(obs)>1:
            # initialize Dir sample
            sample_check=[]
            theta2_static=np.empty((2*num_tar*num_tar,2*num_tar))
            all_post=np.zeros((int((self.num_samples-self.burn_in)/5),1,num_tar))
            self.all_theta2=np.zeros((int((self.num_samples-self.burn_in)/5),2*num_tar*num_tar,2*num_tar))
            for X in range(num_tar):
                for prev_obs in range(2*num_tar):
                    theta2_static[X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=self.theta2_full[X,prev_obs,:])

            # begin gibbs sampling
            theta2=copy.deepcopy(theta2_static)
            for n in range(self.num_samples):
                # calc X as if we knew theta2
                for i in self.names:
                    # likelihood from theta1 (not full dist, assuming we know theta1)
                    index=self.select_param(self.names.index(i),obs[0])
                    if index%2==0:
                        likelihood=self.theta1[index]
                    else:
                        likelihood=self.theta1[index]/(num_tar-1)
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
                    all_post[int((n-self.burn_in)/5),:,:]=postX.values()
                # sample from X
                X=np.random.choice(range(num_tar),p=postX.values())
                alphas=copy.deepcopy(self.theta2_full)
                theta2=copy.deepcopy(theta2_static)
                # calc theta2 as if we knew X
                for i in range(len(obs)-1):
                    alphas[X,obs[i],obs[i+1]]+=1
                for j in range(theta2.shape[1]):
                    theta2[X*2*num_tar+j,:]=np.random.dirichlet(alphas[X,j,:])
                if n%5==0:
                    self.all_theta2[int((n-self.burn_in)/5),:,:]=theta2


            # take max likelihood of X for next obs
            post_probs=np.mean(all_post,axis=0)
            return post_probs[0]

        # using only theat1 on first observation
        else:
            for i in self.names:
                # likelihood from theta1 (not full dist, assuming we know theta1)
                index=self.select_param(self.names.index(i),obs[0])
                likelihood=self.theta1[index]
                postX[i]=self.probs[i]*likelihood
            # normalize and set final values
            suma=sum(postX.values())
            for i in self.names:
                postX[i]=np.log(postX[i])-np.log(suma) 
                postX[i]=np.exp(postX[i])
            return postX.values()

    def moment_matching_full(self):
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


    def sampling_param_tied(self,num_tar,obs):
        postX=copy.deepcopy(self.probs)
        # only learning theta2 on 2+ observations
        if len(obs)>1:
            # initialize Dir sample
            sample_check=[]
            theta2_static=np.empty((4,4))
            all_post=np.zeros((int((self.num_samples-self.burn_in)/5),1,num_tar))
            self.theta2_samples=np.zeros((int((self.num_samples-self.burn_in)/5),4,4))
            for i in range(4):
                theta2_static[i,:]=scipy.stats.dirichlet.mean(alpha=self.theta2[i,:])

            # begin gibbs sampling
            theta2=copy.deepcopy(theta2_static)
            for n in range(self.num_samples):
                # calc X as if we knew theta2
                for i in self.names:
                    # lieklihood from theta1
                    index=self.select_param(self.names.index(i),obs[0])
                    if index%2==0:
                        likelihood=self.theta1[index]
                    else:
                        likelihood=self.theta1[index]/(num_tar-1)
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
                    all_post[int((n-self.burn_in)/5),:,:]=postX.values()
                # sample from X
                X=np.random.choice(range(num_tar),p=postX.values())
                alphas=copy.deepcopy(self.theta2)
                theta2=copy.deepcopy(theta2_static)
                # calc theta2 as is we knew X
                for i in range(len(obs)-1):
                    indicies=self.select_param(X,obs[i+1],obs[i])
                    #DEBUG
                    #  if (n==500):
                    #      print X,obs[i+1],obs[i],indicies
                    #      print alphas
                    alphas[indicies[0],indicies[1]]+=1
                    #  if (n==500):
                    #      print alphas
                for j in range(4):
                    theta2[j,:]=np.random.dirichlet(alphas[j,:])
                if n%5==0:
                    self.theta2_samples[int((n-self.burn_in)/5),:,:]=theta2

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
            for i in self.names:
                # likelihood from theta1 (not full dist, assuming we know theta1)
                index=self.select_param(self.names.index(i),obs[0])
                likelihood=self.theta1[index]
                postX[i]=self.probs[i]*likelihood
            # normalize and set final values
            suma=sum(postX.values())
            for i in self.names:
                postX[i]=np.log(postX[i])-np.log(suma) 
                postX[i]=np.exp(postX[i])
            return postX.values()

    def moment_matching(self,graph=False):
        # moment matching of alphas from samples (Minka, 2000)
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
                    #DEBUG
                    if graph:
                        if (n==1) and (k==0):
                            plt.figure()
                            plt.hist(samples,bins=20,density=True)
                            x=np.linspace(0,1)
                            plt.plot(x,scipy.stats.beta.pdf(x,alphak,sum(self.theta2[n,:])-alphak))
                            plt.show()
                            #  sys.exit()

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

def load_config(path=None):
    if not path:
        path=os.path.dirname(__file__) + 'config.yaml'
    try:
        with open(path, 'r') as stream:
            cfg=yaml.load(stream)
    except IOError:
        print "No config file found"
        raise
    return cfg

if __name__ == '__main__':
    commands=[]
    for i in range(1,len(sys.argv)):
        commands.append(sys.argv[i])
    num_events=int(commands[0])

    cfg=load_config('config.yaml')
    graph_params=cfg['graphs']
    # checking the config choices
    if not cfg['sim_types']['full_dir']:
        if graph_params['sim_results_full']:
            print "can't make the full sim results"
            raise
    if not cfg['sim_types']['param_tied_dir']:
        if graph_params['sim_results_tied']:
            print "can't make the tied sim results"
            raise
    if not cfg['sim_types']['param_tied_dir']:
        if graph_params['gibbs_val']:
            print "can't make gibbs graph"
            raise
    
    # initializing variables
    num_tar=cfg['num_tar']
    threshold=cfg['threshold']

    if graph_params['sim_results_full']:
        # target confusion matrix
        true_tar_full=[]
        pred_tar_full=[]
        pred_tar_full_ml=[]
        # precision recall
        pred_percent_full=[]
        correct_full=[0]*num_events
        # running average
        correct_percent_full=[]
        correct_ml_full=[0]*num_events
        correct_percent_ml_full=[]
    else:
        # target confusion matrix
        true_tar_full=None
        pred_tar_full=None
        pred_tar_full_ml=None
        # precision recall
        pred_percent_full=None
        correct_full=None
        # running average
        correct_percent_full=None
        correct_ml_full=None
        correct_percent_ml_full=None
    if graph_params['sim_results_tied']:
        # target confusion matrix
        true_tar_tied=[]
        pred_tar_tied=[]
        pred_tar_tied_ml=[]
        # precision recall
        pred_percent_tied=[]
        correct_tied=[0]*num_events
        # running average
        correct_percent_tied=[]
        correct_ml_tied=[0]*num_events
        correct_percent_ml_tied=[]
    else:
        # target confusion matrix
        true_tar_tied=None
        pred_tar_tied=None
        pred_tar_tied_ml=None
        # precision recall
        pred_percent_tied=None
        correct_tied=None
        # running average
        correct_percent_tied=None
        correct_ml_tied=None
        correct_percent_ml_tied=None
    #  # human validation
    #  all_theta2_tied=np.empty((num_events,4,4))
    #  all_theta2_full=np.empty((num_events,2*num_tar*num_tar,2*num_tar))
    if graph_params['gibbs_val']:
        # gibbs validations
        theta2_samples=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    else:
        theta2_samples=None

    # start sim
    if cfg['sim_types']['full_dir']:
        full_sim=DataFusion()
        full_sim.DirPrior(num_tar)
    if cfg['sim_types']['param_tied_dir']:
        param_tied_sim=DataFusion()
        param_tied_sim.DirPrior(num_tar)
    if graph_params['theta_val']:
        alphas_start=copy.deepcopy(param_tied_sim.theta2)
    else:
        alphas_start=None

    #running sim
    for n in tqdm(range(num_events),ncols=100):
        # initialize target type
        genus=np.random.randint(num_tar)
        #  param_tied_sim.make_data(genus)

        # getting a prior from ML
        if cfg['starting_dist']=='assist':
            if (cfg['sim_types']['full_dir']) and (cfg['sim_types']['param_tied_dir']):
                param_tied_sim.make_data(genus)
                param_tied_sim.frame=0
                param_tied_sim.alphas={}
                param_tied_sim.probs={}
                for i in param_tied_sim.names:
                    param_tied_sim.alphas[i]=[-1,-1]
                    param_tied_sim.probs[i]=.2
                while max(param_tied_sim.probs.values())<0.6:
                    param_tied_sim.updateProbsML()
                    param_tied_sim.frame+=1
                full_sim.probs=param_tied_sim.probs
                chosen=np.argmax(param_tied_sim.probs.values())
                if graph_params['sim_results_full']:
                    if genus==chosen:
                        correct_ml_full[n]=1
                    correct_percent_ml_full.append(sum(correct_ml_full)/(n+1))
                    pred_tar_full_ml.append(np.argmax(param_tied_sim.probs.values()))
                if graph_params['sim_results_tied']:
                    if genus==chosen:
                        correct_ml_tied[n]=1
                    correct_percent_ml_tied.append(sum(correct_ml_tied)/(n+1))
                    pred_tar_tied_ml.append(np.argmax(param_tied_sim.probs.values()))

            elif (cfg['sim_types']['full_dir']) and not (cfg['sim_types']['param_tied_dir']):
                full_sim.make_data(genus)
                full_sim.frame=0
                full_sim.alphas={}
                full_sim.probs={}
                for i in full_sim.names:
                    full_sim.alphas[i]=[-1,-1]
                    full_sim.probs[i]=.2
                while max(full_sim.probs.values())<0.6:
                    full_sim.updateProbsML()
                    full_sim.frame+=1
                chosen=np.argmax(full_sim.probs.values())
                if graph_params['sim_results_full']:
                    if genus==chosen:
                        correct_ml_full[n]=1
                    correct_percent_ml_full.append(sum(correct_ml_full)/(n+1))
                    pred_tar_full_ml.append(np.argmax(full_sim.probs.values()))

            elif not (cfg['sim_types']['full_dir']) and (cfg['sim_types']['param_tied_dir']):
                param_tied_sim.make_data(genus)
                param_tied_sim.frame=0
                param_tied_sim.alphas={}
                param_tied_sim.probs={}
                for i in param_tied_sim.names:
                    param_tied_sim.alphas[i]=[-1,-1]
                    param_tied_sim.probs[i]=.2
                while max(param_tied_sim.probs.values())<0.6:
                    param_tied_sim.updateProbsML()
                    param_tied_sim.frame+=1
                chosen=np.argmax(param_tied_sim.probs.values())
                if graph_params['sim_results_tied']:
                    if genus==chosen:
                        correct_ml_tied[n]=1
                    correct_percent_ml_tied.append(sum(correct_ml_tied)/(n+1))
                    pred_tar_tied_ml.append(np.argmax(param_tied_sim.probs.values()))

        elif cfg['starting_dist']=='uniform':
            if (cfg['sim_types']['full_dir']) and (cfg['sim_types']['param_tied_dir']):
                full_sim.probs={}
                param_tied_sim.probs={}
                for i in param_tied_sim.names:
                    full_sim.probs[i]=.2
                    param_tied_sim.probs[i]=.2
                #TODO: this needs to change for different target number
                chosen=np.random.choice([0,0,0,0,1])
                if graph_params['sim_results_full']:
                    correct_ml_full[n]=chosen
                    correct_percent_ml_full.append(sum(correct_ml_full)/(n+1))
                    pred_tar_full_ml.append(np.random.randint(num_tar))
                if graph_params['sim_results_tied']:
                    correct_ml_tied[n]=chosen
                    correct_percent_ml_tied.append(sum(correct_ml_tied)/(n+1))
                    pred_tar_tied_ml.append(np.random.randint(num_tar))
            elif (cfg['sim_types']['full_dir']) and not (cfg['sim_types']['param_tied_dir']):
                full_sim.probs={}
                for i in full_sim.names:
                    full_sim.probs[i]=.2
                if graph_params['sim_results_full']:
                    correct_ml_full[n]=np.random.choice([0,0,0,0,1])
                    correct_percent_ml_full.append(sum(correct_ml_full)/(n+1))
                    pred_tar_full_ml.append(np.random.randint(num_tar))
            elif not (cfg['sim_types']['full_dir']) and (cfg['sim_types']['param_tied_dir']):
                param_tied_sim.probs={}
                for i in param_tied_sim.names:
                    param_tied_sim.probs[i]=.2
                if graph_params['sim_results_tied']:
                    correct_ml_tied[n]=np.random.choice([0,0,0,0,1])
                    correct_percent_ml_tied.append(sum(correct_ml_tied)/(n+1))
                    pred_tar_tied_ml.append(np.random.randint(num_tar))

        obs=[]
        if cfg['sim_types']['param_tied_dir'] and cfg['sim_types']['full_dir']:
            full_sim_probs=full_sim.probs.values()
            param_tied_sim_probs=param_tied_sim.probs.values()
            count_full=0
            count_tied=0
            while (max(full_sim_probs)<threshold) or (max(param_tied_sim_probs)<threshold):
                obs=param_tied_sim.HumanObservations(num_tar,genus,obs)
                if max(full_sim_probs)<threshold:
                    full_sim_probs=full_sim.sampling_full(num_tar,obs)
                    count_full+=1
                if max(param_tied_sim_probs)<threshold:
                    param_tied_sim_probs=param_tied_sim.sampling_param_tied(num_tar,obs)
                    count_tied+=1
            if count_full>1:
                full_sim.moment_matching_full()
            if count_tied>1:
                param_tied_sim.moment_matching()
        elif cfg['sim_types']['full_dir']:
            full_sim_probs=full_sim.probs.values()
            count_full=0
            while (max(full_sim_probs)<threshold):
                obs=full_sim.HumanObservations(num_tar,genus,obs)
                full_sim_probs=full_sim.sampling_full(num_tar,obs)
                count_full+=1

            if count_full>1:
                full_sim.moment_matching_full()
        elif cfg['sim_types']['param_tied_dir']:
            param_tied_sim_probs=param_tied_sim.probs.values()
            count_tied=0
            while (max(param_tied_sim_probs)<threshold):
                obs=param_tied_sim.HumanObservations(num_tar,genus,obs)
                param_tied_sim_probs=param_tied_sim.sampling_full(num_tar,obs)
                count_tied+=1

            if graph_params['gibbs_val']:
                if count_tied>1:
                    param_tied_sim.moment_matching()

                # need a run where all 16 have been sampled, keep storing until a run produces that
                if count_tied>1:
                    for i in range(4):
                        for j in range(4):
                            samples=param_tied_sim.theta2_samples[np.nonzero(param_tied_sim.theta2_samples[:,i,j]),i,j]
                            if len(samples[0])>len(theta2_samples[i*4+j]):
                                theta2_samples[i*4+j]=samples[0]

        if graph_params['sim_results_full']:
            # building graphing parameters
            chosen=max(full_sim_probs)
            pred_percent_full.append(chosen)
            true_tar_full.append(genus)
            pred_tar_full.append(np.argmax(full_sim_probs))
            if genus==np.argmax(full_sim_probs):
                correct_full[n]=1
            correct_percent_full.append(sum(correct_full)/(n+1))
        if graph_params['sim_results_tied']:
            # building graphing parameters
            chosen=max(param_tied_sim_probs)
            pred_percent_tied.append(chosen)
            true_tar_tied.append(genus)
            pred_tar_tied.append(np.argmax(param_tied_sim_probs))
            if genus==np.argmax(param_tied_sim_probs):
                correct_tied[n]=1
            correct_percent_tied.append(sum(correct_tied)/(n+1))
        #  all_theta2_tied[n,:,:]=param_tied_sim.theta2
        #  theta2_full_alphas=np.empty((2*num_tar*num_tar,2*num_tar))
        #  for X in range(num_tar):
        #      for prev_obs in range(2*num_tar):
        #          theta2_full_alphas[X*2*num_tar+prev_obs,:]=full_sim.theta2_full[X,prev_obs,:]
        #  all_theta2_full[n,:,:]=theta2_full_alphas
    if graph_params['theta_val']:
        theta2=param_tied_sim.theta2
        theta2_correct=param_tied_sim.theta2_correct
    else:
        theta2=None
        theta2_correct=None
    if graph_params['gibbs_val']:
        X_samples=param_tied_sim.X_samples
    else:
        X_samples=None
    if graph_params['sim_results_full']:
        real_obs=full_sim.real_obs
        pred_obs=full_sim.pred_obs
        pred_tar_ml=pred_tar_full_ml
    if graph_params['sim_results_tied']:
        real_obs=param_tied_sim.real_obs
        pred_obs=param_tied_sim.pred_obs
        pred_tar_ml=pred_tar_tied_ml
    if not (graph_params['sim_results_tied']) and not (graph_params['sim_results_full']):
        real_obs=None
        pred_obs=None
        pred_tar_ml=None


    # TODO: need pred_tar_ml
    graphs=Graphing(num_events,num_tar,alphas_start,theta2,true_tar_full,
            pred_tar_full,real_obs,pred_obs,pred_tar_ml,correct_percent_full,
            correct_percent_ml_full,correct_full,pred_percent_full,true_tar_tied,
            pred_tar_tied,correct_percent_tied,correct_percent_ml_tied,correct_tied,
            pred_percent_tied,theta2_correct,theta2_samples,X_samples)
    plt.show()
