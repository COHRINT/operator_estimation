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
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
np.set_printoptions(precision=2)


class Graphing():
    def __init__(self,num_events,num_tar,alphas_start=None,theta2=None,true_tar_full=None,
            pred_tar_full=None,real_obs=None,pred_obs=None,pred_tar_ml=None,correct_percent_full=None,
            correct_percent_ml_full=None,correct_full=None,pred_percent_full=None,true_tar_tied=None,
            pred_tar_tied=None,correct_percent_tied=None,correct_percent_ml_tied=None,correct_tied=None,
            pred_percent_tied=None,theta2_correct=None,theta2_samples=None,X_samples=None):

        self.gif_time=10 #seconds

        if (theta2_correct is not None) and (alphas_start is not None) and (theta2 is not None):
            print "Making Theta Validation Plots"
            self.theta_validation(num_tar,theta2_correct,alphas_start,theta2)
        #  if (theta2_samples is not None) and (X_samples is not None):
        if theta2_samples is not None:
            print "Making Gibbs Validation Plots"
            #  self.gibbs_validation(num_tar,theta2_samples,X_samples)
            self.gibbs_validation(num_tar,theta2_samples)
        #  # TODO: chang ethese conditions, only care about percent correct for full
        #  condition_full=((true_tar_full is not None) and (pred_tar_full is not None) and
        #          (real_obs is not None) and (pred_obs is not None) and
        #          (correct_percent_full is not None) and (correct_percent_ml_full is not None) and
        #          (correct_full is not None) and (pred_percent_full is not None))
        #  if condition_full:
        #      print "Making Experiment Results Plot"
        #      self.experimental_results(true_tar_full,pred_tar_full,real_obs,
        #              pred_obs,num_events,correct_percent_full,
        #          correct_percent_ml_full,correct_full,pred_percent_full,style='(Full)')
        #  condition_tied=((true_tar_tied is not None) and (pred_tar_tied is not None) and
        #          (real_obs is not None) and (pred_obs is not None) and
        #          (correct_percent_tied is not None) and (correct_percent_ml_tied is not None) and
        #          (correct_tied is not None) and (pred_percent_tied is not None))
        #  if condition_tied:
        #      print "Making Experiment Results Plot"
        #      self.experimental_results(true_tar_tied,pred_tar_tied,real_obs,
        #              pred_obs,num_events,correct_percent_tied,
        #          correct_percent_ml_tied,correct_tied,pred_percent_tied,style='(Tied)')
        if (true_tar_full is not None) and (pred_tar_full is not None):
            print "Making Full Sim Confusion Matrix"
            self.confusion(true_tar_full,pred_tar_full,style='target',
                    title='Full Sim Target Confusion Matrix',filename='full_tar_confusion')
        if (true_tar_tied is not None) and (pred_tar_tied is not None):
            print "Making Tied Sim Confusion Matrix"
            self.confusion(true_tar_tied,pred_tar_tied,style='target',
                    title='Tied Sim Target Confusion Matrix',filename='tied_tar_confusion')
        if (true_tar_tied is not None) and (pred_tar_ml is not None):
            print "Making ML Confusion Matrix"
            self.confusion(true_tar_tied,pred_tar_ml,style='target',
                    title='ML Target Confusion Matrix',filename='ml_tar_confusion')
        if (real_obs is not None) and (pred_obs is not None):
            print "Making Operator Confusion Matrix"
            self.confusion(real_obs,pred_obs,style='obs',
                    title='Operator Confusion Matrix',filename='operator_confusion')
        if (correct_percent_tied is not None) and (correct_percent_ml_tied is not None) and (correct_percent_full is not None):
            print "Making Percent Correct Plot"
            self.percent_correct(num_events,correct_percent_tied,correct_percent_ml_tied,correct_percent_full)
        elif (correct_percent_tied is not None) and (correct_percent_ml_tied is not None):
            print "Making Percent Correct Plot"
            self.percent_correct(num_events,correct_percent_tied,correct_percent_ml_tied)
        elif (correct_percent_full is not None) and (correct_percent_ml_full is not None):
            print "Making Percent Correct Plot"
            self.percent_correct(num_events,correct_percent_full,correct_percent_ml_full)
        #  print "Making Theta2 Validation GIF"
        #  self.human_validation()
        #  print "Making Convergence GIF"
        #  self.convergence_validation()


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

    #  def gibbs_validation(self,num_tar,theta2_samples,X_samples):
    def gibbs_validation(self,num_tar,theta2_samples):
        for i in range(4):
            if len(theta2_samples[i])==0:
                print "At least one case produced no samples, must have samples for gibbs graph"
                return
        strings=['TP','FP','FN','TN']
        #  colors=['g','y','r','c']
        #  fig1=plt.figure(figsize=(18,12),tight_layout=True)
        #  fig1.suptitle(r'Signal, Histogram, and Autocorrelation of Gibbs Samples ($\theta_2$)',fontweight='bold')
        #  for i in tqdm(range(4),ncols=100):
        #      for j in range(4):
        #          ax0=plt.subplot2grid((9,12),(2*i+1,3*j),colspan=2)
        #          ax0.plot(range(len(theta2_samples[4*i+j])),theta2_samples[4*i+j],color=colors[i])
        #          ax0.set_ylim((0,1))
        #          ax0.set_title(strings[i]+', '+strings[j],fontweight='bold',loc='right')
        #          ax0.set_xlabel('Sample #')
        #          ax0.set_ylabel(r'p($\theta_2$)')
        #          ax1=plt.subplot2grid((9,12),(2*i+1,3*j+2))
        #          ax1.hist(theta2_samples[4*i+j],bins=20,range=(0,1),color=colors[i])
        #          ax1.set_xlabel(r'p($\theta_2$)')
        #          ax1.set_ylabel('Frequency')
        #          ax2=plt.subplot2grid((9,12),(2*i+2,3*j),colspan=3)
        #          corr=self.lagk_correlation(theta2_samples[4*i+j])
        #          ax2.plot(range(len(corr)),corr,color=colors[i])
        #          ax2.set_xlabel('Lag')
        #          ax2.set_ylabel(r'$\rho$')
        #  fig2=plt.figure(figsize=(16,12),tight_layout=True)
        #  fig2.suptitle('Signal, Histogram, and Autocorrelation of Gibbs Samples (X)',fontweight='bold')
        #  for i in tqdm(range(num_tar),ncols=100):
        #      ax0=plt.subplot2grid((2*(int(num_tar/2)+num_tar%2)+1,6),(2*int(i/2)+1,3*(i%2)),colspan=2)
        #      ax0.plot(range(len(X_samples[:,:,i])),X_samples[:,:,i])
        #      ax0.set_ylim((0,1))
        #      ax0.set_title('X='+str(i),fontweight='bold',loc='right')
        #      ax0.set_xlabel('Sample #')
        #      ax0.set_ylabel('p(X)')
        #      ax1=plt.subplot2grid((2*(int(num_tar/2)+num_tar%2)+1,6),(2*int(i/2)+1,3*(i%2)+2))
        #      ax1.hist(X_samples[:,:,i],bins=20,range=(0,1))
        #      ax1.set_xlabel('p(X)')
        #      ax1.set_ylabel('Frequency')
        #      ax2=plt.subplot2grid((2*(int(num_tar/2)+num_tar%2)+1,6),(2*int(i/2)+2,3*(i%2)),colspan=3)
        #      corr=self.lagk_correlation(X_samples[:,:,i])
        #      ax2.plot(range(len(corr)),corr)
        #      ax2.set_xlabel('Lag')
        #      ax2.set_ylabel(r'$\rho$')
        #  fig1.savefig('figures/gibbs_validation_theta.png',bbox_inches='tight',pad_inches=0)
        #  fig2.savefig('figures/gibbs_validation_X.png',bbox_inches='tight',pad_inches=0)

        fig1=plt.figure(figsize=(9,4),tight_layout=True)
        fig1.suptitle(r'Signal, Histogram, and Autocorrelation of Gibbs Samples ($\theta_2$)',fontweight='bold')
        for j in range(2):
            ax0=plt.subplot2grid((3,6),(1,3*j),colspan=2)
            ax0.plot(range(len(theta2_samples[j])),theta2_samples[j])
            ax0.set_ylim((0,1))
            ax0.set_title('TP, '+strings[j],fontweight='bold',loc='right')
            ax0.set_xlabel('Sample #')
            ax0.set_ylabel(r'p($\theta_2$)')
            ax1=plt.subplot2grid((3,6),(1,3*j+2))
            ax1.hist(theta2_samples[j],bins=20,range=(0,1))
            ax1.set_xlabel(r'p($\theta_2$)')
            ax1.set_ylabel('Frequency')
            ax2=plt.subplot2grid((3,6),(2,3*j),colspan=3)
            corr=self.lagk_correlation(theta2_samples[j])
            ax2.plot(range(len(corr)),corr)
            ax2.set_xlabel('Lag')
            ax2.set_ylabel(r'$\rho$')

        fig1.savefig('figures/gibbs_validation_theta.png',bbox_inches='tight',pad_inches=0)

    def confusion(self,true,pred,style,title,filename):
        fig=plt.figure()
        cm=confusion_matrix(true,pred)
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        plt.imshow(cm,cmap='Blues',vmin=0.0,vmax=1.0)
        if style=='target':
            plt.ylabel('True Label')
            plt.xlabel('Given Label')
        elif style=='obs':
            plt.ylabel('True Value')
            plt.xlabel('Given Obs')
            plt.xticks([0,1],['pos','neg'])
            plt.yticks([0,1],['pos','neg'])
        plt.title(title)
        for i, j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
            plt.text(j,i,format(100*cm[i,j],'.1f')+'%',horizontalalignment="center",color="white" if cm[i,j]>cm.max()/2 else "black")

        fig.savefig('figures/'+filename+'.png',bbox_inches='tight',pad_inches=0)

    def percent_correct(self,num_events,correct_percent,correct_percent_ml,correct_percent_full=None):
        fig=plt.figure()
        if correct_percent_full is not None:
            plt.plot([n+5 for n in range(num_events-5)],correct_percent[5:], label="w/Human Total Correct (Tied)",linewidth=4)
            plt.plot([n+5 for n in range(num_events-5)],correct_percent_full[5:], label="w/Human Total Correct (Full)")
        else:
            plt.plot([n+5 for n in range(num_events-5)],correct_percent[5:], label="w/Human Total Correct")
        plt.plot([n+5 for n in range(num_events-5)],correct_percent_ml[5:], label="wo/Human Total Correct")
        plt.legend()
        plt.xlabel('Number of Targets')
        plt.ylabel('Percent Correct')
        plt.ylim(0,1.1)
        plt.title('Correct Classification')

        fig.savefig('figures/percent_correct.png',bbox_inches='tight',pad_inches=0)

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

    def dependent_independent_compare(self):
        pass

    def accuracy_comparison(self):
        # compare between independent and dependent
        pass
