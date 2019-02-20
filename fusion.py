from __future__ import division
import sys
import os
import numpy as np
import scipy.stats
from scipy import spatial
from scipy.special import gamma, psi, polygamma
import random
import matplotlib.pyplot as plt
import time
import copy

class Fusion():
    def __init__(self):
        self.num_samples=5000
        self.burn_in=1000
        self.threshold=.90
        self.names = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']

    def DirPrior(self,num_tar,human="bad"):
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
            if human=='good':
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
            elif human=='bad':
                #tp
                base_table_real[i,2*i]*=5
                for j in range(5):
                    if i==j:
                        #fn
                        base_table_real[i,2*j+1]*=1.5
                    else:
                        #tn
                        base_table_real[i,2*j+1]*=5
                        #fp
                        base_table_real[i,2*j]*=2.5
        for i in range(2*num_tar):
            table_real[:,:,i]=base_table_real
        for i in range(2*num_tar):
            #repeat
            table_real[:,i,i]*=3
        table_real=np.swapaxes(table_real,1,2)
        table_real+=np.random.uniform(-1,1,(num_tar,2*num_tar,2*num_tar))
        table_real[table_real<0]=0.1
        self.theta2_correct=np.zeros((2*num_tar*num_tar,2*num_tar))
        for X in range(num_tar):
            for prev_obs in range(2*num_tar):
                self.theta2_correct[X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=table_real[X,prev_obs,:])


    def updateProbs(self,obs):
        postX=copy.deepcopy(self.probs)
        num_tar=5
        # only learning theta2 on 2+ observations
        if len(obs)>1:
            # initialize Dir sample
            sample_check=[]
            theta2_static=np.empty((4,4))
            #  all_post=np.zeros((int((self.num_samples-self.burn_in)/5),1,num_tar))
            all_post=np.zeros(int((self.num_samples-self.burn_in)/5))
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
                # sample from X
                X=np.random.choice(range(num_tar),p=postX.values())
                # store every 5th sample
                if n%5==0:
                    all_post[int((n-self.burn_in)/5)]=X
                #      all_post[int((n-self.burn_in)/5),:,:]=postX.values()
                alphas=copy.deepcopy(self.theta2)
                theta2=copy.deepcopy(theta2_static)
                # calc theta2 as is we knew X
                for i in range(len(obs)-1):
                    indicies=self.select_param(X,obs[i+1],obs[i])
                    alphas[indicies[0],indicies[1]]+=1
                for j in range(4):
                    theta2[j,:]=np.random.dirichlet(alphas[j,:])
                if n%5==0:
                    self.theta2_samples[int((n-self.burn_in)/5),:,:]=theta2

            # take max likelihood of X for next obs
            all_post=list(all_post)
            post_probs=[all_post.count(0),all_post.count(1),all_post.count(2),all_post.count(3),all_post.count(4)]
            post_probs=[x/len(all_post) for x in post_probs]
            #  post_probs=np.mean(all_post,axis=0)[0]
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
            post_probs=postX.values()

        self.post_probs=post_probs
        return post_probs

    def moment_matching(self,obs,graph=False):
        if (len(obs)>1) and (max(self.post_probs)>self.threshold):
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
                        #  if graph:
                        #      if (n==0) and (k==0):
                        #          plt.figure()
                        #          plt.hist(samples,bins=20,density=True)
                        #          x=np.linspace(0,1)
                        #          plt.plot(x,scipy.stats.beta.pdf(x,alphak,sum(self.theta2[n,:])-alphak))
                        #          plt.xlabel(r'$\theta_2$')
                        #          plt.ylabel(r'$p(\theta_2)$')
                        #          plt.title("Moment Matching for TP,TP")
                        #          plt.show()
                        #          sys.exit()
            self.reset=True
        else:
            self.reset=False

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
