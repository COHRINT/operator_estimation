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
import yaml
import cPickle as pickle
import warnings
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
np.set_printoptions(precision=2)

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

class Graphing():
    def __init__(self,graph_dic):
        #  self.gif_time=10 #seconds

        num_tar=graph_dic['num_tar']
        num_events=graph_dic['num_events']
        # percent correct
        correct_percent_tied=graph_dic['correct_percent_tied']
        correct_percent_full=graph_dic['correct_percent_full']
        correct_percent_ind=graph_dic['correct_percent_ind']
        correct_percent_ml=graph_dic['correct_percent_ml']
        correct_percent_ml_alone=graph_dic['correct_percent_ml_alone']
        # precision recall
        pred_percent_tied=graph_dic['pred_percent_tied']
        pred_percent_full=graph_dic['pred_percent_full']
        pred_percent_ind=graph_dic['pred_percent_ind']
        pred_percent_ml=graph_dic['pred_percent_ml']
        pred_percent_ml_alone=graph_dic['pred_percent_ml_alone']
        correct_tied=graph_dic['correct_tied']
        correct_full=graph_dic['correct_full']
        correct_ind=graph_dic['correct_ind']
        correct_ml=graph_dic['correct_ml']
        correct_ml_alone=graph_dic['correct_ml_alone']
        # confusion
        true_tar_tied=graph_dic['true_tar_tied']
        pred_tar_tied=graph_dic['pred_tar_tied']
        pred_tar_full=graph_dic['pred_tar_full']
        pred_tar_ind=graph_dic['pred_tar_ind']
        pred_tar_ml=graph_dic['pred_tar_ml']
        pred_tar_ml_alone=graph_dic['pred_tar_ml_alone']
        real_obs=graph_dic['real_obs']
        pred_obs=graph_dic['pred_obs']
        # timing
        tied_times=graph_dic['tied_times']
        tied_number=graph_dic['tied_number']
        tied_match_times=graph_dic['tied_match_times']
        full_times=graph_dic['full_times']
        full_number=graph_dic['full_number']
        full_match_times=graph_dic['full_match_times']
        ind_times=graph_dic['ind_times']
        ind_number=graph_dic['ind_number']
        ind_match_times=graph_dic['ind_match_times']
        # theta val
        theta1=graph_dic['theta1']
        theta1_correct=graph_dic['theta1_correct']
        theta2=graph_dic['theta2']
        theta2_correct=graph_dic['theta2_correct']
        alphas_start=graph_dic['alphas_start']
        alphas1_start=graph_dic['alphas1_start']
        #gibbs val
        theta2_samples=graph_dic['theta2_samples'] 
        X_samples=graph_dic['X_samples']
        # data
        data_dic=graph_dic['data']
        # pass off
        pass_off_average=graph_dic['pass_off_average']
        pass_off=graph_dic['pass_off']

       
        if (theta2_correct is not None) and (alphas_start is not None) and (theta2 is not None):
            print "Making Theta Validation Plots"
            self.theta_validation(num_tar,theta2_correct,alphas_start,theta2)
        if (theta1_correct is not None) and (alphas1_start is not None) and (theta1 is not None):
            print "Making Theta 1 Validation Plots"
            self.theta1_validation(num_tar,theta1_correct,alphas1_start,theta1)
        if (theta2_samples is not None) and (X_samples is not None):
            print "Making Gibbs Validation Plots"
            self.gibbs_validation(num_tar,theta2_samples,X_samples)
            #  self.gibbs_validation(num_tar,theta2_samples)
        if (full_times is not None) and (tied_times is not None) and (ind_times is not None):
            print "Making Timing Comparison"
            self.timing(tied_times,tied_number,tied_match_times,full_times,full_number,
                    full_match_times,ind_times,ind_number,ind_match_times)

        if (true_tar_tied is not None) and (pred_tar_tied is not None) and \
                (pred_tar_full is not None) and (pred_tar_ind is not None) and \
                (pred_tar_ml is not None) and (pred_tar_ml_alone is not None) and \
                (real_obs is not None) and (pred_obs is not None):
            print "Making Confusion Matrixes"
            self.confusion(true_tar_tied,pred_tar_tied,pred_tar_full,pred_tar_ind,pred_tar_ml,
                    pred_tar_ml_alone,real_obs,pred_obs)

        if (pred_percent_tied is not None) and (correct_tied is not None) and \
                (pred_percent_full is not None) and (correct_full is not None) and \
                (pred_percent_ind is not None) and (correct_ind is not None) and \
                (pred_percent_ml is not None) and (correct_ml is not None) and \
                (pred_percent_ml_alone is not None) and (correct_ml_alone is not None):
            print "Making Precision-Recall Graphs"
            self.precision_recall_graph(correct_tied,pred_percent_tied,correct_full,pred_percent_full,
            correct_ind,pred_percent_ind,correct_ml,pred_percent_ml,correct_ml_alone,
            pred_percent_ml_alone)

        if (correct_percent_tied is not None) and (correct_percent_ml is not None) \
                and (correct_percent_full is not None) and (correct_percent_ind is not None):
            print "Making Percent Correct Plot"
            self.percent_correct(num_events,correct_percent_tied,correct_percent_ml,
                    correct_percent_full,correct_percent_ind,correct_percent_ml_alone)
        
        if data_dic is not None:
            print "Making Data Graph"
            self.data_graph(num_tar,data_dic)

        if (pass_off_average is not None) and (pass_off is not None):
            print "Making Pass Off Graph"
            self.pass_off_graph(num_events,pass_off_average,pass_off)
        #  print "Making Theta2 Validation GIF"
        #  self.human_validation()
        #  print "Making Convergence GIF"
        #  self.convergence_validation()



    def lagk_correlation(self,data):
        "https://www.itl.nist.gov/div898/handbook/eda/section3/eda35c.htm"
        rhok=np.empty(len(data))
        mean=np.mean(data)
        for k in range(len(data)):
            n=len(data)
            numerator=0
            for i in range(n-k):
                numerator+=(data[i]-mean)*(data[i+k]-mean)
            denominator=0
            for i in range(n):
                denominator+=(data[i]-mean)**2
            rhok[k]=(numerator/denominator)
        return rhok

    def theta1_validation(self,num_tar,theta1_correct,alphas_start,theta1):
        # need to make a dynamic sized list for all 16 params
        # there are a different number of parameters for each type
        breakdown=[[],[],[],[]]

        # assigning the theta2_correct table to tp,fp,fn,tn values
        for i in range(theta1_correct.shape[0]):
            if 2*int(i/(2*num_tar))==i%(2*num_tar):
                index=0 #tp
            elif i%2==0:
                index=1 #fp
            if 2*int(i/(2*num_tar))+1==i%(2*num_tar):
                index=2 #fn
            elif i%2==1:
                index=3 #tn

            #  if (index==1) or (index==3):
            if (index==1):
                breakdown[index].append((num_tar-1)*theta1_correct[i])
            else:
                breakdown[index].append(theta1_correct[i])

        # maginalizing out dimentions of dirichlet into beta functions
        alpha_beta_start=np.empty((4,2))
        alpha_beta_est=np.empty((4,2))
        for i in range(4):
            alpha_beta_start[i,:]=[alphas_start[i],sum(alphas_start)-alphas_start[i]]
            alpha_beta_est[i,:]=[theta1[i],sum(theta1)-theta1[i]]

        strings=['TP','FP','FN','TN']
        fig,ax=plt.subplots(nrows=1,ncols=4,figsize=((15,4)),tight_layout=True)
        #  fig.suptitle(r'Starting $p(\theta_2)$, Estimated $p(\theta_2)$, and True $\theta_2$',fontweight='bold')
        x=np.linspace(0,1)
        for i in range(4):
            ax[i].plot(x,scipy.stats.beta.pdf(x,alpha_beta_start[i,0],alpha_beta_start[i,1]),label=r"Starting $p(\theta_1)$")
            ax[i].plot(x,scipy.stats.beta.pdf(x,alpha_beta_est[i,0],alpha_beta_est[i,1]),label=r"Estimated $p(\theta_1)$")
            ax[i].scatter(breakdown[i],len(breakdown[i])*[0],label=r'real $\theta$')
            ax[i].set_xlabel(r'$\theta_1$')
            ax[i].set_ylabel('PDF')
            ax[i].set_title(strings[i],fontweight='bold')
            ax[i].legend()
        fig.savefig('figures/theta1_validation.png',bbox_inches='tight',pad_inches=0)

    def theta_validation(self,num_tar,theta2_correct,alphas_start,theta2):
        # need to make a dynamic sized list for all 16 params
        # there are a different number of parameters for each type
        breakdown=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

        # assigning the theta2_correct table to tp,fp,fn,tn values
        for i in range(theta2_correct.shape[0]):
            if 2*int(i/(2*num_tar))==i%(2*num_tar):
                index1=0 #tp
            elif i%2==0:
                index1=1 #fp
            if 2*int(i/(2*num_tar))+1==i%(2*num_tar):
                index1=2 #fn
            elif i%2==1:
                index1=3 #tn

            for j in range(theta2_correct.shape[1]):
                if 2*int(i/(2*num_tar))==j:
                    index2=0
                elif j%2==0:
                    index2=1
                if 2*int(i/(2*num_tar))+1==j:
                    index2=2
                elif j%2==1:
                    index2=3


                if (index2==1) or (index2==3):
                    breakdown[index1*4+index2].append((num_tar-1)*theta2_correct[i,j])
                else:
                    breakdown[index1*4+index2].append(theta2_correct[i,j])

        # maginalizing out dimentions of dirichlet into beta functions
        alpha_beta_start=np.empty((4,4,2))
        alpha_beta_est=np.empty((4,4,2))
        for i in range(4):
            for j in range(4):
                alpha_beta_start[i,j,:]=[alphas_start[i,j],sum(alphas_start[i,:])-alphas_start[i,j]]
                alpha_beta_est[i,j,:]=[theta2[i,j],sum(theta2[i,:])-theta2[i,j]]

        strings=['TP','FP','FN','TN']
        fig,ax=plt.subplots(nrows=4,ncols=4,figsize=((15,15)),tight_layout=True)
        #  fig.suptitle(r'Starting $p(\theta_2)$, Estimated $p(\theta_2)$, and True $\theta_2$',fontweight='bold')
        x=np.linspace(0,1)
        for i in range(4):
            for j in range(4):
                ax[i,j].plot(x,scipy.stats.beta.pdf(x,alpha_beta_start[i,j,0],alpha_beta_start[i,j,1]),label=r"Starting $p(\theta_2)$")
                ax[i,j].plot(x,scipy.stats.beta.pdf(x,alpha_beta_est[i,j,0],alpha_beta_est[i,j,1]),label=r"Estimated $p(\theta_2)$")
                ax[i,j].scatter(breakdown[i*4+j],len(breakdown[i*4+j])*[0],label=r'real $\theta$')
                ax[i,j].set_xlabel(r'$\theta_2$')
                ax[i,j].set_ylabel('PDF')
                ax[i,j].set_title(strings[i]+', '+strings[j],fontweight='bold')
                ax[i,j].legend()
        fig.savefig('figures/theta_validation.png',bbox_inches='tight',pad_inches=0)

    def gibbs_validation(self,num_tar,theta2_samples,X_samples):
    #  def gibbs_validation(self,num_tar,theta2_samples):
        for i in range(16):
            if len(theta2_samples[i])==0:
                print "At least one case produced no samples, must have samples for gibbs graph"
                return
        strings=['TP','FP','FN','TN']
        colors=['g','y','r','c']
        fig1=plt.figure(figsize=(18,12),tight_layout=True)
        fig1.suptitle(r'Signal, Histogram, and Autocorrelation of Gibbs Samples ($\theta_2$)',fontweight='bold')
        for i in tqdm(range(4),ncols=100):
            for j in range(4):
                ax0=plt.subplot2grid((9,12),(2*i+1,3*j),colspan=2)
                ax0.plot(range(len(theta2_samples[4*i+j])),theta2_samples[4*i+j],color=colors[i])
                ax0.set_ylim((0,1))
                ax0.set_title(strings[i]+', '+strings[j],fontweight='bold',loc='right')
                ax0.set_xlabel('Sample #')
                ax0.set_ylabel(r'p($\theta_2$)')
                ax1=plt.subplot2grid((9,12),(2*i+1,3*j+2))
                ax1.hist(theta2_samples[4*i+j],bins=20,range=(0,1),color=colors[i])
                ax1.set_xlabel(r'p($\theta_2$)')
                ax1.set_ylabel('Frequency')
                ax2=plt.subplot2grid((9,12),(2*i+2,3*j),colspan=3)
                corr=self.lagk_correlation(theta2_samples[4*i+j])
                ax2.plot(range(len(corr)),corr,color=colors[i])
                ax2.set_xlabel('Lag')
                ax2.set_ylabel(r'$\rho$')
        fig2=plt.figure(figsize=(16,12),tight_layout=True)
        fig2.suptitle('Signal, Histogram, and Autocorrelation of Gibbs Samples (X)',fontweight='bold')
        for i in tqdm(range(num_tar),ncols=100):
            ax0=plt.subplot2grid((2*(int(num_tar/2)+num_tar%2)+1,6),(2*int(i/2)+1,3*(i%2)),colspan=2)
            ax0.plot(range(len(X_samples[:,:,i])),X_samples[:,:,i])
            ax0.set_ylim((0,1))
            ax0.set_title('X='+str(i),fontweight='bold',loc='right')
            ax0.set_xlabel('Sample #')
            ax0.set_ylabel('p(X)')
            ax1=plt.subplot2grid((2*(int(num_tar/2)+num_tar%2)+1,6),(2*int(i/2)+1,3*(i%2)+2))
            ax1.hist(X_samples[:,:,i],bins=20,range=(0,1))
            ax1.set_xlabel('p(X)')
            ax1.set_ylabel('Frequency')
            ax2=plt.subplot2grid((2*(int(num_tar/2)+num_tar%2)+1,6),(2*int(i/2)+2,3*(i%2)),colspan=3)
            corr=self.lagk_correlation(X_samples[:,:,i])
            ax2.plot(range(len(corr)),corr)
            ax2.set_xlabel('Lag')
            ax2.set_ylabel(r'$\rho$')
        fig1.savefig('figures/gibbs_validation_theta.png',bbox_inches='tight',pad_inches=0)
        fig2.savefig('figures/gibbs_validation_X.png',bbox_inches='tight',pad_inches=0)

        #  fig1=plt.figure(figsize=(9,4),tight_layout=True)
        #  fig1.suptitle(r'Signal, Histogram, and Autocorrelation of Gibbs Samples ($\theta_2$)',fontweight='bold')
        #  for j in range(2):
        #      ax0=plt.subplot2grid((3,6),(1,3*j),colspan=2)
        #      ax0.plot(range(len(theta2_samples[j])),theta2_samples[j])
        #      ax0.set_ylim((0,1))
        #      ax0.set_title('TP, '+strings[j],fontweight='bold',loc='right')
        #      ax0.set_xlabel('Sample #')
        #      ax0.set_ylabel(r'p($\theta_2$)')
        #      ax1=plt.subplot2grid((3,6),(1,3*j+2))
        #      ax1.hist(theta2_samples[j],bins=20,range=(0,1))
        #      ax1.set_xlabel(r'p($\theta_2$)')
        #      ax1.set_ylabel('Frequency')
        #      ax2=plt.subplot2grid((3,6),(2,3*j),colspan=3)
        #      corr=self.lagk_correlation(theta2_samples[j])
        #      ax2.plot(range(len(corr)),corr)
        #      ax2.set_xlabel('Lag')
        #      ax2.set_ylabel(r'$\rho$')

        #  fig1.savefig('figures/gibbs_validation_theta.png',bbox_inches='tight',pad_inches=0)

    def confusion(self,true_tar_tied,pred_tar_tied,pred_tar_full,pred_tar_ind,pred_tar_ml,pred_tar_ml_alone,real_obs,pred_obs):
        fig=plt.figure(figsize=((15,10)),tight_layout=True)
        # Tied
        plt.subplot(231)
        cm=confusion_matrix(true_tar_tied,pred_tar_tied)
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        plt.imshow(cm,cmap='Blues',vmin=0.0,vmax=1.0)
        plt.xticks(range(cm.shape[0]))
        plt.yticks(range(cm.shape[1]))
        plt.ylabel('True Label')
        plt.xlabel('Given Label')
        plt.title('Tied Sim')
        for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
            plt.text(j,i,format(100*cm[i,j],'.1f')+'%',horizontalalignment="center",color="white" if cm[i,j]>cm.max()/2 else "black")

        # Full
        plt.subplot(232)
        cm=confusion_matrix(true_tar_tied,pred_tar_full)
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        plt.imshow(cm,cmap='Blues',vmin=0.0,vmax=1.0)
        plt.xticks(range(cm.shape[0]))
        plt.yticks(range(cm.shape[1]))
        plt.ylabel('True Label')
        plt.xlabel('Given Label')
        plt.title('Full Sim')
        for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
            plt.text(j,i,format(100*cm[i,j],'.1f')+'%',horizontalalignment="center",color="white" if cm[i,j]>cm.max()/2 else "black")

        # Ind
        plt.subplot(233)
        cm=confusion_matrix(true_tar_tied,pred_tar_ind)
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        plt.imshow(cm,cmap='Blues',vmin=0.0,vmax=1.0)
        plt.xticks(range(cm.shape[0]))
        plt.yticks(range(cm.shape[1]))
        plt.ylabel('True Label')
        plt.xlabel('Given Label')
        plt.title('Ind Sim')
        for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
            plt.text(j,i,format(100*cm[i,j],'.1f')+'%',horizontalalignment="center",color="white" if cm[i,j]>cm.max()/2 else "black")

        # ML
        plt.subplot(234)
        cm=confusion_matrix(true_tar_tied,pred_tar_ml)
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        plt.imshow(cm,cmap='Blues',vmin=0.0,vmax=1.0)
        plt.xticks(range(cm.shape[0]))
        plt.yticks(range(cm.shape[1]))
        plt.ylabel('True Label')
        plt.xlabel('Given Label')
        plt.title('HMM Sim')
        for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
            plt.text(j,i,format(100*cm[i,j],'.1f')+'%',horizontalalignment="center",color="white" if cm[i,j]>cm.max()/2 else "black")

        # ML no pass
        plt.subplot(235)
        cm=confusion_matrix(true_tar_tied,pred_tar_ml_alone)
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        plt.imshow(cm,cmap='Blues',vmin=0.0,vmax=1.0)
        plt.xticks(range(cm.shape[0]))
        plt.yticks(range(cm.shape[1]))
        plt.ylabel('True Label')
        plt.xlabel('Given Label')
        plt.title('HMM No Human Sim')
        for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
            plt.text(j,i,format(100*cm[i,j],'.1f')+'%',horizontalalignment="center",color="white" if cm[i,j]>cm.max()/2 else "black")

        # human
        plt.subplot(236)
        cm=confusion_matrix(real_obs,pred_obs)
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        plt.imshow(cm,cmap='Blues',vmin=0.0,vmax=1.0)
        plt.ylabel('True Value')
        plt.xlabel('Given Obs')
        plt.xticks([0,1],['pos','neg'])
        plt.yticks([0,1],['pos','neg'])
        plt.title('Human Operator')
        for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
            plt.text(j,i,format(100*cm[i,j],'.1f')+'%',horizontalalignment="center",color="white" if cm[i,j]>cm.max()/2 else "black")

        fig.savefig('figures/confusion.png',bbox_inches='tight',pad_inches=0)

    def percent_correct(self,num_events,correct_percent,correct_percent_ml,correct_percent_full,correct_percent_ind,correct_percent_ml_alone):
        fig=plt.figure()
        plt.plot([n+5 for n in range(num_events-5)],correct_percent[5:], label="w/Human Total Correct (Tied)",linewidth=4)
        plt.plot([n+5 for n in range(num_events-5)],correct_percent_full[5:], label="w/Human Total Correct (Full)")
        plt.plot([n+5 for n in range(num_events-5)],correct_percent_ind[5:], label="w/Human Total Correct (Ind)")
        plt.plot([n+5 for n in range(num_events-5)],correct_percent_ml[5:], label="Before Human Total Correct")
        plt.plot([n+5 for n in range(num_events-5)],correct_percent_ml_alone[5:], label="No Human Total Correct")
        plt.legend()
        plt.xlabel('Number of Targets')
        plt.ylabel('Percent Correct')
        plt.ylim(0,1.1)
        plt.title('Correct Classification')

        fig.savefig('figures/percent_correct.png',bbox_inches='tight',pad_inches=0)

    def timing(self,tied_times,tied_number,tied_match_times,full_times,full_number,
            full_match_times,ind_times,ind_number,ind_match_times):
        full_mean=np.mean(full_times)
        full_std=np.std(full_times)
        full_match=np.mean(full_match_times)
        full_match_std=np.std(full_match_times)
        full_avg_num=np.mean(full_number)

        ind_mean=np.mean(ind_times)
        ind_std=np.std(ind_times)
        ind_match=np.mean(ind_match_times)
        ind_match_std=np.std(ind_match_times)
        ind_avg_num=np.mean(ind_number)

        tied_mean=np.mean(tied_times)
        tied_std=np.std(tied_times)
        tied_match=np.mean(tied_match_times)
        tied_match_std=np.std(tied_match_times)
        tied_avg_num=np.mean(tied_number)

        fig1=plt.figure(figsize=((10,4)),tight_layout=True)
        plt.subplot(121)
        plt.bar(range(3),[ind_mean,tied_mean,full_mean],yerr=[ind_std,full_std,tied_std])
        plt.xticks(range(3),('Ind','Tied','Full'))
        plt.title('Average Time for Gibbs Sampling (5000 samples)')
        plt.ylabel('Seconds')

        plt.subplot(122)
        plt.bar(range(3),[ind_match,tied_match,full_match],yerr=[ind_match_std,tied_match_std,
            full_match_std],color='C1')
        plt.xticks(range(3),('Ind','Tied','Full'))
        plt.title('Average Time for Moment Matching')
        plt.ylabel('Seconds')

        fig2=plt.figure()
        for i in range(int(tied_avg_num)):
            plt.bar(1,tied_mean,color='C0',edgecolor='black',bottom=i*tied_mean)
        plt.bar(1,(tied_avg_num%1*tied_mean),color='C0',edgecolor='black',
                bottom=int(tied_avg_num)*tied_mean,label='sampling')
        plt.bar(1,tied_match,color='C1',edgecolor='black',yerr=tied_match_std+tied_avg_num*tied_std,
                bottom=tied_avg_num*tied_mean,label='moment match')

        for i in range(int(ind_avg_num)):
            plt.bar(0,ind_mean,color='C0',edgecolor='black',bottom=i*ind_mean)
        for i in range(int(full_avg_num)):
            plt.bar(2,full_mean,color='C0',edgecolor='black',bottom=i*full_mean)
        plt.bar(2,(full_avg_num%1*full_mean),color='C0',edgecolor='black',
                bottom=int(full_avg_num)*full_mean)
        plt.bar(2,full_match,color='C1',edgecolor='black',yerr=full_match_std+full_avg_num*full_std,
                bottom=full_avg_num*full_mean)
        plt.bar(0,(ind_avg_num%1*ind_mean),color='C0',edgecolor='black',
                bottom=int(ind_avg_num)*ind_mean)
        plt.bar(0,ind_match,color='C1',edgecolor='black',yerr=ind_match_std+ind_avg_num*ind_std,
                bottom=ind_avg_num*ind_mean)
        plt.xticks(range(3),('Ind','Tied','Full'))

        plt.title('Average Total Time for Classification')
        plt.ylabel('Seconds')
        plt.legend()

        fig1.savefig('figures/timing_parts.png',bbox_inches='tight',pad_inches=0)
        fig2.savefig('figures/timing_total.png',bbox_inches='tight',pad_inches=0)

    def precision_recall_graph(self,correct_tied,pred_percent_tied,correct_full,pred_percent_full,
            correct_ind,pred_percent_ind,correct_ml,pred_percent_ml,correct_ml_alone,
            pred_percent_ml_alone):
        fig=plt.figure(figsize=((15,10)),tight_layout=True)
        # Tied
        plt.subplot(231)
        precision, recall, _ =precision_recall_curve(correct_tied,pred_percent_tied)
        plt.step(recall,precision,where='post')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precission')
        plt.title('Tied Sim')

        # Full
        plt.subplot(232)
        precision, recall, _ =precision_recall_curve(correct_full,pred_percent_full)
        plt.step(recall,precision,where='post')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precission')
        plt.title('Full Sim')

        # Ind
        plt.subplot(233)
        precision, recall, _ =precision_recall_curve(correct_ind,pred_percent_ind)
        plt.step(recall,precision,where='post')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precission')
        plt.title('Ind Sim')

        # ML
        plt.subplot(234)
        precision, recall, _ =precision_recall_curve(correct_ml,pred_percent_ml)
        plt.step(recall,precision,where='post')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precission')
        plt.title('HMM Sim')

        # ML no pass
        plt.subplot(235)
        precision, recall, _ =precision_recall_curve(correct_ml_alone,pred_percent_ml_alone)
        plt.step(recall,precision,where='post')
        plt.xlim([0.0,1.0])
        plt.ylim([0.0,1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precission')
        plt.title('HMM No Human Sim')

        fig.savefig('figures/precision_recall.png',bbox_inches='tight',pad_inches=0)

    def data_graph(self,num_tar,data_dic):
        fig=plt.figure()
        for i in range(num_tar):
            data=np.zeros((len(data_dic[i]),100))
            for trial in range(len(data_dic[i])):
                data[trial,:]=data_dic[i][trial]
            mean_data=np.mean(data,axis=0)
            std_data=np.std(data,axis=0)
            plt.plot(range(len(mean_data)),mean_data,label=i)
            plt.fill_between(range(len(mean_data)),mean_data+std_data,mean_data-std_data,alpha=0.5)
        plt.xlabel('Time (Frames)')
        plt.ylabel('Intensity (Units)')
        plt.title('Average Data Seen')
        plt.legend()

        fig.savefig('figures/data.png',bbox_inches='tight',pad_inches=0)

    def pass_off_graph(self,num_events,pass_off_average,pass_off):
        fig=plt.figure()
        plt.plot([n+5 for n in range(num_events-5)],pass_off_average[5:])
        plt.scatter(range(len(pass_off)),pass_off)
        plt.xlabel('Number of Targets')
        plt.ylabel('Average Pass Off Frame')
        plt.title('Average Pass Off Frame Over Time')

        fig.savefig('figures/pass_off.png',bbox_inches='tight',pad_inches=0)

    # --------------Not using these plots------------
    def experimental_results(self,true_tar,pred_tar,real_obs,pred_obs,num_events,correct_percent,
            correct_percent_ml,correct,pred_percent,style='(Tied)'):
        fig=plt.figure(figsize=(12,9))
        fig.suptitle('Expereimental Results '+style)
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
        plt.show()

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

        fig.savefig('figures/experimental_results.png',bbox_inches='tight',pad_inches=0)

    def human_validation(self):
        total_difference_tied=np.empty([self.num_events,2*num_tar*num_tar,2*num_tar])
        total_difference_full=np.empty([self.num_events,2*num_tar*num_tar,2*num_tar])
        avg_difference_tied=np.empty((self.num_events))
        avg_difference_full=np.empty((self.num_events))
        for n in tqdm(range(self.num_events),ncols=100):
            theta2_tied=self.build_theta2(self.num_tar,self.all_theta2_tied[n,:,:])

            theta_tied_mean=np.empty((2*num_tar*num_tar,2*num_tar))
            theta_full_mean=np.empty((2*num_tar*num_tar,2*num_tar))
            for X in range(num_tar):
                for prev_obs in range(2*num_tar):
                    theta_tied_mean[X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=theta2_tied[X*2*num_tar+prev_obs,:])
                    theta_full_mean[X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=self.all_theta2_full[n,X*2*num_tar+prev_obs,:])
            for i in range(total_difference_tied.shape[1]):
                for j in range(total_difference_tied.shape[2]):
                    #  total_difference_tied[n,i,j]=self.KLD(theta_real_mean[i,j],theta_tied_mean[i,j],theta_real_var[i,j],theta_tied_var[i,j])
                    #  total_difference_full[n,i,j]=self.KLD(theta_real_mean[i,j],theta_full_mean[i,j],theta_real_var[i,j],theta_full_var[i,j])
                    #TODO sizing is wrong and need real alphas table
                    total_difference_tied[n,i,j]=self.KLD_dirichlet(theta2_tied,self.table_compare)
                    total_difference_full[n,i,j]=self.KLD_dirichlet(self.all_theta2_full,self.table_compare)
            one_dim_tied=np.reshape(theta_tied_mean,(1,4*num_tar**3))
            one_dim_full=np.reshape(theta_full_mean,(1,4*num_tar**3))
            one_dim_correct=np.reshape(self.theta2_correct,(1,4*num_tar**3))
            avg_difference_tied[n]=scipy.stats.entropy(one_dim_tied[0],one_dim_correct[0])
            avg_difference_full[n]=scipy.stats.entropy(one_dim_full[0],one_dim_correct[0])

        d=np.abs(total_difference_tied[1:,:,:]-np.median(total_difference_tied[1:,:,:]))
        mdev=np.median(d)
        vmax=2*mdev+np.median(total_difference_tied[1:,:,:])
        fig=plt.figure(figsize=(10,15),tight_layout=True)
        for frame in tqdm(range(self.num_events),ncols=100):
            imgplot0=plt.subplot2grid((5,2),(0,0),rowspan=4)
            im=imgplot0.imshow(total_difference_tied[frame],cmap='hot',vmin=np.min(total_difference_tied),vmax=vmax)
            fig.colorbar(im,ax=imgplot0)
            imgplot0.set_xticks([])
            imgplot0.set_yticks([])
            imgplot0.set_title('KLD all params (tied)')
            imgplot1=plt.subplot2grid((5,2),(0,1),rowspan=4)
            im2=imgplot1.imshow(total_difference_full[frame],cmap='hot',vmin=np.min(total_difference_full),vmax=vmax)
            fig.colorbar(im2,ax=imgplot1)
            imgplot1.set_xticks([])
            imgplot1.set_yticks([])
            imgplot1.set_title('KLD all params (full)')
            imgplot2=plt.subplot2grid((5,2),(4,0))
            imgplot2.plot(np.linspace(0,frame,frame),avg_difference_tied[0:frame],'c')
            imgplot2.set_xlabel('Number of Events')
            imgplot2.set_ylabel(r'KLD of all $\theta$')
            imgplot2.set_xlim(0,self.num_events)
            imgplot2.set_ylim(0,max(avg_difference_tied))
            imgplot3=plt.subplot2grid((5,2),(4,1))
            imgplot3.plot(np.linspace(0,frame,frame),avg_difference_full[0:frame],'c')
            imgplot3.set_xlabel('Number of Events')
            imgplot3.set_ylabel(r'KLD of all $\theta$')
            imgplot3.set_xlim(0,self.num_events)
            imgplot3.set_ylim(0,max(avg_difference_full))

            #  fig.savefig('figures/tmp/human'+str(frame)+'.png',bbox_inches='tight',pad_inches=0,dpi=400)
            fig.savefig('figures/tmp/human'+str(frame)+'.png',bbox_inches='tight',pad_inches=0)
            #  plt.pause(0.2)
        fig.clear()
        plt.close()
        fig,ax=plt.subplots(figsize=(10,15),tight_layout=True)
        images=[]
        for k in tqdm(range(1,self.num_events),ncols=100):
            fname='figures/tmp/human%d.png' % k
            img=mgimg.imread(fname)
            imgplot=plt.imshow(img)
            plt.axis('off')
            images.append([imgplot])
        ani=animation.ArtistAnimation(fig,images)
        ani.save("figures/human_validation.gif",fps=self.num_events/self.gif_time)
        fig.clear()
        plt.close()
            #  plt.xlabel('%d Targets' % (int(num_tar/10)*i))
        #  cax=plt.axes([0.93,0.25,0.025,0.5])
        #  plt.colorbar(cax=cax)

    def convergence_validation(self):
        total_difference_tied=np.empty([self.num_events-1,4,4])
        total_difference_full=np.empty([self.num_events-1,2*num_tar*num_tar,2*num_tar])
        avg_difference_tied=np.empty((self.num_events-1))
        avg_difference_full=np.empty((self.num_events-1))
        theta_tied_mean=np.empty((self.num_events,4,4))
        theta_tied_var=np.empty((self.num_events,4,4))
        theta_full_mean=np.empty((self.num_events,2*num_tar*num_tar,2*num_tar))
        theta_full_var=np.empty((self.num_events,2*num_tar*num_tar,2*num_tar))
        for n in tqdm(range(self.num_events),ncols=100):
            for X in range(num_tar):
                for prev_obs in range(2*num_tar):
                    theta_full_mean[n,X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=self.all_theta2_full[n,X*2*num_tar+prev_obs,:])
            for i in range(4):
                theta_tied_mean[n,i,:]=scipy.stats.dirichlet.mean(alpha=self.all_theta2_tied[n,i,:])
        for n in range(self.num_events)[1:]:
            for i in range(total_difference_tied.shape[1]):
                for j in range(total_difference_tied.shape[2]):
                    #TODO (sizing is wrong
                    total_difference_tied[n-1,i,j]=self.KLD_dirichlet(self.all_theta2_tied[n,i,:],self.all_theta2_tied[n-1,i,:])
                    total_difference_tied[n-1,i,j]=self.KLD_dirichlet(self.all_theta2_full[n,i,:],self.all_theta2_full[n-1,i,:])
            for i in range(total_difference_full.shape[1]):
                for j in range(total_difference_full.shape[2]):
                    total_difference_full[n-1,i,j]=self.KLD(theta_full_mean[n,i,j],theta_full_mean[n-1,i,j],theta_full_var[n,i,j],theta_full_var[n-1,i,j])

            one_dim_tied_old=np.reshape(theta_tied_mean[n-1],(1,16))
            one_dim_tied=np.reshape(theta_tied_mean[n],(1,16))
            one_dim_full_old=np.reshape(theta_full_mean[n-1],(1,4*num_tar**3))
            one_dim_full=np.reshape(theta_full_mean[n],(1,4*num_tar**3))
            avg_difference_tied[n-1]=scipy.stats.entropy(one_dim_tied_old[0],one_dim_tied[0])
            avg_difference_full[n-1]=scipy.stats.entropy(one_dim_full_old[0],one_dim_full[0])

        #  d=np.abs(total_difference_tied[1:,:,:]-np.median(total_difference_tied[1:,:,:]))
        #  mdev=np.median(d)
        #  vmax=2*mdev+np.median(total_difference_tied[1:,:,:])
        fig=plt.figure(figsize=(10,15),tight_layout=True)
        for frame in tqdm(range(self.num_events-1),ncols=100):
            imgplot0=plt.subplot2grid((5,2),(0,0),rowspan=4)
            im=imgplot0.imshow(total_difference_tied[frame],cmap='hot',vmin=np.min(total_difference_tied),vmax=np.max(total_difference_tied))
            fig.colorbar(im,ax=imgplot0)
            imgplot0.set_xticks([])
            imgplot0.set_yticks([])
            imgplot0.set_title('KLD between events (tied)')
            imgplot1=plt.subplot2grid((5,2),(0,1),rowspan=4)
            im2=imgplot1.imshow(total_difference_full[frame],cmap='hot',vmin=np.min(total_difference_full),vmax=np.max(total_difference_full))
            fig.colorbar(im2,ax=imgplot1)
            imgplot1.set_xticks([])
            imgplot1.set_yticks([])
            imgplot1.set_title('KLD between events (full)')
            imgplot2=plt.subplot2grid((5,2),(4,0))
            imgplot2.plot(np.linspace(0,frame,frame),avg_difference_tied[0:frame],'c')
            imgplot2.set_xlabel('Number of Events')
            imgplot2.set_ylabel(r'KLD of all $\theta$')
            imgplot2.set_xlim(0,self.num_events)
            imgplot2.set_ylim(0,max(avg_difference_tied))
            imgplot3=plt.subplot2grid((5,2),(4,1))
            imgplot3.plot(np.linspace(0,frame,frame),avg_difference_full[0:frame],'c')
            imgplot3.set_xlabel('Number of Events')
            imgplot3.set_ylabel(r'KLD of all $\theta$')
            imgplot3.set_xlim(0,self.num_events)
            imgplot3.set_ylim(0,max(avg_difference_full))

            #  fig.savefig('figures/tmp/converge'+str(frame)+'.png',bbox_inches='tight',pad_inches=0,dpi=400)
            fig.savefig('figures/tmp/converge'+str(frame)+'.png',bbox_inches='tight',pad_inches=0)
            #  plt.pause(0.2)
        fig.clear()
        plt.close()
        fig,ax=plt.subplots(figsize=(10,15),tight_layout=True)
        images=[]
        for k in tqdm(range(1,self.num_events-1),ncols=100):
            fname='figures/tmp/converge%d.png' % k
            img=mgimg.imread(fname)
            imgplot=plt.imshow(img)
            plt.axis('off')
            images.append([imgplot])
        ani=animation.ArtistAnimation(fig,images)
        ani.save("figures/convergence_validation.gif",fps=self.num_events/self.gif_time)
        fig.clear()
        plt.close()

    def KLD(self,mean_i,mean_j,var_i,var_j):
        dist=.5*((var_i**2/var_j**2)+var_j**2*(mean_j-mean_i)**2-1+np.log(var_j**2/var_i**2))
        return np.absolute(dist)

    def KLD_dirichlet(self,alphas1,alphas2):
        "http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/"
        a10=sum(alphas1)
        a20=sum(alphas2)
        final_term=0
        for k in range(len(alphas1)):
            final_term+=(alphas1[k]-alphas2[k])*(psi(alphas1[k])-psi(a10))
        return np.log(gamma(a10))-sum(np.log(gamma(alphas1)))-np.log(gamma(a20))+sum(np.log(gamma(alphas2)))+final_term

    def build_theta2(self,num_tar,alphas):
        # make full theta2 table from param tied table
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


if __name__ == '__main__':
    cfg=load_config('config.yaml')
    num_sims=cfg['num_sims']
    num_tar=cfg['num_tar']
    graph_params=cfg['graphs']
    graph_dic=pickle.load(open('graphing_data0.p','rb'))
    for j in range(1,num_sims):
        filename='graphing_data'+str(i)+'.p'
        new_dic=pickle.load(open(filename,'rb'))

        if graph_params['percent_correct']:
            graph_dic['correct_percent_tied'].append(new_dic['correct_percent_tied'])
            graph_dic['correct_percent_full'].append(new_dic['correct_percent_full'])
            graph_dic['correct_percent_ind'].append(new_dic['correct_percent_ind'])
            graph_dic['correct_percent_ml'].append(new_dic['correct_percent_ml'])
            graph_dic['correct_percent_ml_alone'].append(new_dic['correct_percent_ml_alone'])
            #TODO: need to change graph fcn

        if graph_params['precision_recall']:
            graph_dic['pred_percent_tied'].extend(new_dic['pred_percent_tied'])
            graph_dic['correct_tied'].extend(new_dic['correct_tied'])
            graph_dic['pred_percent_full'].extend(new_dic['pred_percent_full'])
            graph_dic['correct_full'].extend(new_dic['correct_full'])
            graph_dic['pred_percent_ind'].extend(new_dic['pred_percent_ind'])
            graph_dic['correct_ind'].extend(new_dic['correct_ind'])
            graph_dic['pred_percent_ml'].extend(new_dic['pred_percent_ml'])
            graph_dic['correct_ml'].extend(new_dic['correct_ml'])
            graph_dic['pred_percent_ml_alone'].extend(new_dic['pred_percent_ml_alone'])
            graph_dic['correct_ml_alone'].extend(new_dic['correct_ml_alone'])

        if graph_params['confusion']:
            graph_dic['true_tar_tied'].extend(new_dic['true_tar_tied'])
            graph_dic['pred_tar_tied'].extend(new_dic['pred_tar_tied'])
            graph_dic['pred_tar_full'].extend(new_dic['pred_tar_full'])
            graph_dic['pred_tar_ind'].extend(new_dic['pred_tar_ind'])
            graph_dic['pred_tar_ml'].extend(new_dic['pred_tar_ml'])
            graph_dic['pred_tar_ml_alone'].extend(new_dic['pred_tar_ml_alone'])
            graph_dic['real_obs'].extend(new_dic['real_obs'])
            graph_dic['pred_obs'].extend(new_dic['pred_obs'])

        if graph_params['data']:
            for i in range(num_tar):
                graph_dic['data'][i].append(new_dic['data'][i])

        #pass off, stack and get averages and std
        if graph_params['pass_off']:
            graph_dic['pass_off_average'].append(new_dic['pass_off_average'])
            #TODO: need to change graph fcn

        if graph_params['timing']:
            graph_dic['tied_times'].extend(new_dic['tied_times'])
            graph_dic['tied_number'].extend(new_dic['tied_number'])
            graph_dic['tied_match_times'].extend(new_dic['tied_match_times'])
            graph_dic['full_times'].extend(new_dic['full_times'])
            graph_dic['full_number'].extend(new_dic['full_number'])
            graph_dic['full_match_times'].extend(new_dic['full_match_times'])
            graph_dic['ind_times'].extend(new_dic['ind_times'])
            graph_dic['ind_number'].extend(new_dic['ind_number'])
            graph_dic['ind_match_times'].extend(new_dic['ind_match_times'])

        #theta val, stack and take averages
        if graph_params['theta_val']:
            graph_dic['theta1'].append(new_dic['theta1'])
            #array
            if j==1:
                graph_dic['theta1_correct']=np.append(graph_dic['theta1_correct'][:,np.newaxis], \
                        new_dic['theta1_correct'][:,np.newaxis],axis=1)
                graph_dic['theta2']=np.append(graph_dic['theta2'][:,:,np.newaxis], \
                        new_dic['theta2'][:,:,np.newaxis],axis=2)
                graph_dic['theta2_correct']=np.append(graph_dic['theta2_correct'][:,:,np.newaxis], \
                        new_dic['theta2_correct'][:,:,np.newaxis],axis=2)
            else:
                graph_dic['theta1_correct']=np.append(graph_dic['theta1_correct'], \
                        new_dic['theta1_correct'][:,np.newaxis],axis=1)
                graph_dic['theta2']=np.append(graph_dic['theta2'], \
                        new_dic['theta2'][:,:,np.newaxis],axis=2)
                graph_dic['theta2_correct']=np.append(graph_dic['theta2_correct'], \
                        new_dic['theta2_correct'][:,:,np.newaxis],axis=2)

            #  graph_dic['alphas_start'].append(new_dic['alphas_start'])
            #  graph_dic['alphas1_start'].append(new_dic['alphas1_start'])

        #avg pass off, take avg and std
        for i in range(num_tar):
            graph_dic['avg_pass_off'+str(i)].extend(new_dic['avg_pass_off'+str(i)])

        #human correct, append and and take avg
        graph_dic['human_correct_overall'].extend(new_dic['human_correct_overall'])

    
    graph_dic['theta1']=np.mean(graph_dic['theta1'],axis=0)
    graph_dic['theta1_correct']=np.mean(graph_dic['theta1_correct'],axis=1)
    graph_dic['theta2']=np.mean(graph_dic['theta2'],axis=2)
    graph_dic['theta2_correct']=np.mean(graph_dic['theta2_correct'],axis=2)

    Graphing(graph_dic)
    print 'Human Percent of Correct Observations:',sum(graph_dic['human_correct_overall'])/len(graph_dic['human_correct_overall'])
    for i in range(num_tar):
        print "Avg, std pass off frame target ",i,":",np.mean(graph_dic['avg_pass_off'+str(i)]), \
                np.std(graph_dic['avg_pass_off'+str(i)])


