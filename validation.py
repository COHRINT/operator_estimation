from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import gamma, psi, polygamma
import random
import copy
import sys
import itertools
import warnings
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
np.set_printoptions(precision=2)

from data_sim.DynamicsProfiles import *
from HMM.hmm_classification import HMM_Classification
from gaussianMixtures import GM

warnings.filterwarnings("ignore",category=RuntimeWarning)

__author__ = "Jeremy Muesing"
__version__ = "2.2.0"
__maintainer__ = "Jeremy Muesing"
__email__ = "jeremy.muesing@colorado.edu"
__status__ = "maintained"

class Human():
    def __init__(self):
        pass

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
        for X in range(num_tar):
            for prev_obs in range(2*num_tar):
                self.theta2_correct[X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=table_real[X,prev_obs,:])
        # for graphing
        self.theta_val=self.theta2_correct[15,2]

    def HumanObservations(self,num_tar,real_target,obs):
        if len(obs)>0:
            prev_obs=obs[-1]
            obs.append(np.random.choice(range(2*num_tar),p=self.theta2_correct[real_target*2*num_tar+prev_obs,:]))
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
        # initialize Dir sample
        sample_check=[]
        theta2_static=np.empty((2*num_tar*num_tar,2*num_tar))
        postX=copy.deepcopy(self.probs)
        all_post=np.zeros((int((self.num_samples-self.burn_in)/5),1,num_tar))
        all_theta2=np.zeros((int((self.num_samples-self.burn_in)/5),2*num_tar*num_tar,2*num_tar))
        for X in range(num_tar):
            for prev_obs in range(2*num_tar):
                theta2_static[X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=self.theta2_full[X,prev_obs,:])

        theta2=copy.deepcopy(theta2_static)
        for n in range(self.num_samples):
            for i in self.names:
                index=self.select_param(self.names.index(i),obs[0])
                if index%2==0:
                    likelihood=self.theta1[index]
                else:
                    likelihood=(self.theta1[index]/(num_tar-1))
                # sample from theta2
                if len(obs)>1:
                    for value in obs[1:]:
                        likelihood*=theta2[self.names.index(i)*2*num_tar+obs[obs.index(value)-1],value]
                #  print likelihood
                postX[i]=self.probs[i]*likelihood
            suma=sum(postX.values())
            # normalize
            for i in self.names:
                postX[i]=np.log(postX[i])-np.log(suma) 
                postX[i]=np.exp(postX[i])
            if n%5==0:
                all_post[int((n-self.burn_in)/5),:,:]=postX.values()
            # sample from X
            X=np.random.choice(range(num_tar),p=postX.values())
            alphas=copy.deepcopy(self.theta2_full)
            theta2=copy.deepcopy(theta2_static)
            if len(obs)>1:
                alphas[X,obs[-2],obs[-1]]+=1
                theta2[X*2*num_tar+obs[-2],:]=np.random.dirichlet(alphas[X,obs[-2],:])
                if n%5==0:
                    all_theta2[int((n-self.burn_in)/5),X*2*num_tar+obs[-2],:]=theta2[X*2*num_tar+obs[-2],:]

        if len(obs)>1:
            sample_counts=np.zeros((2*num_tar*num_tar,2*num_tar))
            # estimation of alphas from distributions
            for n in range(all_theta2.shape[1]):
                pk_top_list=[]
                sum_alpha=sum(self.theta2_full[int(n/(2*num_tar)),n%(2*num_tar),:])
                for k in range(all_theta2.shape[2]):
                    samples=all_theta2[np.nonzero(all_theta2[:,n,k]),n,k]
                    if len(samples[0])==0:
                        pass
                    else:
                        sample_counts[n,k]=len(samples[0])
                        pk_top_list.append(np.mean(samples[0]))
                        current_alpha=self.theta2_full[int(n/(2*num_tar)),n%(2*num_tar),k]
                        for x in range(5):
                            sum_alpha_old=sum_alpha-current_alpha+self.theta2_full[int(n/(2*num_tar)),n%(2*num_tar),k]
                            logpk=np.sum(np.log(samples[0]))/len(samples[0])
                            y=psi(sum_alpha_old)+logpk
                            if y>=-2.22:
                                alphak=np.exp(y)+0.5
                            else:
                                alphak=-1/(y+psi(1))
                            #  print "start:",alphak
                            for w in range(5):
                                alphak-=((psi(alphak)-y)/polygamma(1,alphak))
                            self.theta2_full[int(n/(2*num_tar)),n%(2*num_tar),k]=alphak

        post_probs=np.mean(all_post,axis=0)
        for i in self.names:
            self.probs[i]=post_probs[0][self.names.index(i)]


    def sampling_param_tied(self,num_tar,obs):
        # initialize Dir sample
        sample_check=[]
        postX=copy.deepcopy(self.probs)
        theta2_static=np.empty((4,4))
        all_post=np.zeros((int((self.num_samples-self.burn_in)/5),1,num_tar))
        theta2_samples=np.zeros((int((self.num_samples-self.burn_in)/5),4,4))
        for i in range(4):
            theta2_static[i,:]=scipy.stats.dirichlet.mean(alpha=self.theta2[i,:])

        theta2=copy.deepcopy(theta2_static)
        for n in range(self.num_samples):
            for i in self.names:
                # sample from theta1
                index=self.select_param(self.names.index(i),obs[0])
                if index%2==0:
                    likelihood=self.theta1[index]
                else:
                    likelihood=(self.theta1[index]/(num_tar-1))
                # sample from theta2
                if len(obs)>1:
                    count=0
                    for value in obs[1:]:
                        indicies=self.select_param(self.names.index(i),value,obs[count])
                        if indicies[1]%2==0:
                            likelihood*=theta2[indicies[0],indicies[1]]
                        else:
                            likelihood*=(theta2[indicies[0],indicies[1]]/(num_tar-1))
                        count+=1
                #  print likelihood
                postX[i]=self.probs[i]*likelihood
            suma=sum(postX.values())
            # normalize
            for i in self.names:
                postX[i]=np.log(postX[i])-np.log(suma) 
                postX[i]=np.exp(postX[i])
            if n%5==0:
                all_post[int((n-self.burn_in)/num_tar),:,:]=postX.values()
            # sample from X
            X=np.random.choice(range(num_tar),p=postX.values())
            alphas=copy.deepcopy(self.theta2)
            theta2=copy.deepcopy(theta2_static)
            if len(obs)>1:
                indicies=self.select_param(X,obs[-1],obs[-2])
                alphas[indicies[0],indicies[1]]+=1
                theta2[indicies[0],:]=np.random.dirichlet(alphas[indicies[0],:])
                if n%5==0:
                    theta2_samples[int((n-self.burn_in)/5),indicies[0]]=theta2[indicies[0],indicies[1]]

        if len(obs)>1:
            sample_counts=np.zeros((4,4))
            # estimation of alphas from distributions
            for n in range(4):
                pk_top_list=[]
                sum_alpha=sum(self.theta2[n,:])
                for k in range(4):
                    samples=theta2_samples[np.nonzero(theta2_samples[:,k]),k]
                    if len(samples[0])==0:
                        pass
                    else:
                        sample_counts[n,k]=len(samples[0])
                        pk_top_list.append(np.mean(samples[0]))
                        current_alpha=self.theta2[n,k]
                        for x in range(5):
                            sum_alpha_old=sum_alpha-current_alpha+self.theta2[n,k]
                            logpk=np.sum(np.log(samples[0]))/len(samples[0])
                            y=psi(sum_alpha_old)+logpk
                            if y>=-2.22:
                                alphak=np.exp(y)+0.5
                            else:
                                alphak=-1/(y+psi(1))
                            #  print "start:",alphak
                            for w in range(5):
                                alphak-=((psi(alphak)-y)/polygamma(1,alphak))
                            self.theta2[n,k]=alphak

        post_probs=np.mean(all_post,axis=0)
        for i in self.names:
            self.probs[i]=post_probs[0][self.names.index(i)]

    def select_param(self,target,current_obs,prev_obs=None):
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

def KLD(mean_i,mean_j,var_i,var_j):

    dist=.5*((var_i**2/var_j**2)+var_j**2*(mean_j-mean_i)**2-1+np.log(var_j**2/var_i**2))

    return np.absolute(dist)

class Graphing():

    def build_theta2(self,num_tar,alphas):
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
        theta2=np.empty((2*num_tar*num_tar,num_tar*2))
        for i in range(theta2.shape[0]):
            index1=select_index(int(i/(2*num_tar)),i%(2*num_tar))
            for j in range(theta2.shape[1]):
                index2=select_index(int(i/(2*num_tar)),j%(2*num_tar))
                theta2[i,j]=alphas[index1,index2]
        return theta2

    def lagk_correlation(self,data):
        "https://www.itl.nist.gov/div898/handbook/eda/section3/eda35c.htm"
        rhok=[]
        for X in data:
            n=len(X)
            k=1
            numerator=0
            for i in range(n-k):
                numerator+=(X[i]-X.mean())*(X[i+k]-X.mean())
            denominator=0
            for i in range(n):
                denominator+=(X[i]-X.mean())^2
            rhok.append(numerator/denominator)
        return rhok.mean()

    def theta_validation(self,alphas_start,alphas,theta_real):
        starting_params=self.build_theta2(5,alphas_start)
        estimated_params=self.build_theta2(5,alphas)
        mean_start=scipy.stats.dirichlet.mean(alpha=starting_params[15,:])[2]
        std_start=np.sqrt(scipy.stats.dirichlet.var(alpha=starting_params[15,:])[2])
        mean_est=scipy.stats.dirichlet.mean(alpha=estimated_params[15,:])[2]
        std_est=np.sqrt(scipy.stats.dirichlet.var(alpha=estimated_params[15,:])[2])

        plt.figure()
        plt.plot(np.linspace(0,1),1/(std_est*np.sqrt(2*np.pi))*np.exp(-(np.linspace(0,1)-mean_est)**2/(2*std_est**2)),label=r"Estimated $p(\theta_2)$")
        plt.plot(np.linspace(0,1),1/(std_start*np.sqrt(2*np.pi))*np.exp(-(np.linspace(0,1)-mean_start)**2/(2*std_start**2)),label=r"Starting $p(\theta_2)$")
        plt.scatter(theta_real,0,label=r"$\theta_2$")
        plt.legend()

    def experimental_results(self,num_events,true_tar,pred_tar,real_obs,pred_obs,correct_percent,correct_percent_ml,correct,pred_percent):
        plt.figure()
        plt.subplot(221)
        cm=confusion_matrix(true_tar,pred_tar)
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        plt.imshow(cm,cmap='Blues')
        plt.ylabel('True Label')
        plt.xlabel('Given Label')
        plt.title('Target Classification Confusion Matrix')
        for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
            plt.text(j,i,format(100*cm[i,j],'.1f')+'%',horizontalalignment="center",color="white" if cm[i,j]>cm.max()/2 else "black")

        plt.subplot(222)
        cm=confusion_matrix(real_obs,pred_obs)
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        plt.imshow(cm,cmap='Blues')
        plt.ylabel('True Value')
        plt.xlabel('Given Obs')
        plt.title('Human Observations Confusion Matrix')
        plt.xticks([0,1],['pos','neg'])
        plt.yticks([0,1],['pos','neg'])
        for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
            plt.text(j,i,format(100*cm[i,j],'.1f')+'%',horizontalalignment="center",color="white" if cm[i,j]>cm.max()/2 else "black")

        plt.subplot(223)
        plt.plot([n+5 for n in range(num_events-5)],correct_percent[5:], label="w/Human Total Correct")
        plt.plot([n+5 for n in range(num_events-5)],correct_percent_ml[5:], label="wo/Human Total Correct")
        plt.legend()
        plt.xlabel('Number of Targets')
        plt.ylabel('Percent Correct')
        plt.title('Correct Classification')

        plt.subplot(224)
        precision, recall, _ =precision_recall_curve(correct,pred_percent)
        plt.step(recall,precision,where='post')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precission')
        plt.title('Precision Recall Curve')

    def human_validation(self,num_tar,alphas_tied,alphas_full):
        #  total_difference=np.empty([1,50,10])
        #  theta_real_mean=np.empty((50,10))
        #  theta_calc_mean=np.empty((50,10))
        #  theta_real_var=np.empty((50,10))
        #  theta_calc_var=np.empty((50,10))
        theta2_tied=self.build_theta2(5,alphas_tied)
        theta2_full=self.build_theta2(5,alphas_full)
        for X in range(num_tar):
            for prev_obs in range(2*num_tar):
                theta_real_mean[X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=theta2_full[X,prev_obs,:])
                theta_real_var[X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.var(alpha=theta2_full[X,prev_obs,:])
                theta_calc_mean[X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=theta2_tied[X*2*num_tar+prev_obs,:])
                theta_calc_var[X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.var(alpha=theta2_tied[X*2*num_tar+prev_obs,:])
        if (n%int((num_events/10))==0) or n==(num_events-1):
            difference=np.empty([2*num_tar*num_tar,2*num_tar])
            for i in range(difference.shape[0]):
                for j in range(difference.shape[1]):
                    #TODO
                    difference[i,j]=KLD(theta_real_mean[i,j],theta_calc_mean[i,j],theta_real_var[i,j],theta_calc_var[i,j])
            total_difference=np.append(total_difference,np.expand_dims(difference,axis=0),axis=0)

        plt.figure()
        d=np.abs(total_difference[1:,:,:]-np.median(total_difference[1:,:,:]))
        mdev=np.median(d)
        vmax=2*mdev+np.median(total_difference[1:,:,:])
        for i in range(11):
            plt.subplot(1,11,i+1)
            plt.imshow(total_difference[i+1],cmap='hot',vmin=np.min(total_difference[1:,:,:]),vmax=vmax)
            #  plt.imshow(np.log(total_difference[i+1]),cmap='hot',vmin=np.min(np.log(total_difference[1:,:,:])),vmax=np.max(np.log(total_difference[1:,:,:])))
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('%d Targets' % (int(num_tar/10)*i))
            if i==5:
                plt.title('KLD for Dirichlet Distributions')
        cax=plt.axes([0.93,0.25,0.025,0.5])
        plt.colorbar(cax=cax)

    def convergence_validation(self):
        pass


if __name__ == '__main__':
    commands=[]
    for i in range(1,len(sys.argv)):
        commands.append(sys.argv[i])
    num_events=int(commands[0])

    # initializing variables

    # target confusion matrix
    true_tar=[]
    pred_tar=[]
    # precision recall
    pred_percent=[]
    correct=[0]*num_events
    # running average
    correct_percent=[]
    correct_ml=[0]*num_events
    correct_percent_ml=[]

    # start sim
    full_sim=DataFusion()
    full_sim.DirPrior(5)
    param_tied_sim=DataFusion()
    param_tied_sim.DirPrior(5)
    alphas_start=param_tied_sim.theta2
    num_tar=5
    for n in tqdm(range(num_events),ncols=100):
        # initialize target type
        genus=np.random.randint(num_tar)
        #  param_tied_sim.make_data(genus)

        full_sim.probs={}
        param_tied_sim.probs={}
        if commands[1]=='uniform':
            for i in param_tied_sim.names:
                full_sim.probs[i]=.2
                param_tied_sim.probs[i]=.2
            correct_ml[n]=np.random.choice([0,0,0,0,1],p=param_tied_sim.probs.values())
            correct_percent_ml.append(sum(correct_ml)/(n+1))
        elif commands[1]=='assist':
            #  sim.frame=0
            #  for i in sim.names:
            #      sim.alphas[i]=[-1,-1]
            for i in param_tied_sim.names:
                if param_tied_sim.names.index(i)==genus:
                    param_tied_sim.probs[i]=np.random.normal(.75,.25)
                    full_sim.probs[i]=param_tied_sim.probs[i]
                else:
                    param_tied_sim.probs[i]=np.random.normal(.25,.25)
                    full_sim.probs[i]=param_tied_sim.probs[i]
                if param_tied_sim.probs[i]<0:
                    param_tied_sim.probs[i]=0.01
                if full_sim.probs[i]<0:
                    full_sim.probs[i]=0.01
            for i in param_tied_sim.names:
                param_tied_sim.probs[i]/=sum(param_tied_sim.probs.values())
            for i in full_sim.names:
                full_sim.probs[i]/=sum(full_sim.probs.values())

            chosen_ml=max(param_tied_sim.probs.values())
            if genus==param_tied_sim.probs.values().index(chosen_ml):
                correct_ml[n]=1
            correct_percent_ml.append(sum(correct_ml)/(n+1))

        obs=[]
        while (max(full_sim.probs.values())<0.9) or (max(param_tied_sim.probs.values())<0.9):
            #  for i in range(sim.frame,sim.frame+10):
            #      if i<100:
            #          sim.updateProbsML()
            #          sim.frame+=1
            #          print sim.probs
            #  sys.exit()
            obs=param_tied_sim.HumanObservations(5,genus,obs)
            if max(full_sim.probs.values())<0.9:
                full_sim.sampling_full(5,obs)
            if max(param_tied_sim.probs.values())<0.9:
                param_tied_sim.sampling_param_tied(5,obs)
        chosen=max(param_tied_sim.probs.values())
        pred_percent.append(chosen)
        true_tar.append(genus)
        pred_tar.append(param_tied_sim.probs.values().index(chosen))
        if genus==param_tied_sim.probs.values().index(chosen):
            correct[n]=1
        correct_percent.append(sum(correct)/(n+1))


    graphs=Graphing()
    graphs.theta_validation(alphas_start,param_tied_sim.theta2,param_tied_sim.theta_val)
    graphs.experimental_results(num_events,true_tar,pred_tar,param_tied_sim.real_obs,param_tied_sim.pred_obs,
            correct_percent,correct_percent_ml,correct,pred_percent)
    plt.show()
