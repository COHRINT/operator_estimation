from PyQt5 import QtGui,QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import os
import numpy as np
import scipy.stats
from scipy import spatial
from scipy.special import gamma, psi, polygamma
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import copy

sys.path.append('data_sim')
sys.path.append('HMM')
from regions import Region
from DynamicsProfiles import *
from hmm_classification import HMM_Classification

from matplotlib.backends.backend_qt4agg import FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class InterfaceWindow(QWidget):
    def __init__(self):
            super(InterfaceWindow,self).__init__()
            self.setGeometry(1,1,1350,800)
            self.layout=QGridLayout()
            self.layout.setColumnStretch(0,2)
            self.layout.setRowStretch(0,2)
            self.layout.setRowStretch(1,2)
            self.layout.setColumnStretch(1,2)
            self.layout.setRowStretch(0,2)
            self.layout.setRowStretch(1,2)
            self.setLayout(self.layout)

            self.initialize()
            self.updateTimer=QTimer(self)
            self.interval=1000
            self.updateTimer.setInterval(self.interval)
            self.updateTimer.start()
            self.updateTimer.timeout.connect(self.updateSat)
            self.updateTimer.timeout.connect(self.updateIntensity)
            self.updateTimer.timeout.connect(lambda: self.updateProbs(updateType='machine'))
            
            self.show()

    def initialize(self):

        self.frame=0
        self.obs=[]
        self.hmm=HMM_Classification()
        self.num_samples=5000
        self.burn_in=1000
        self.threshold=.90
        modelFileName = 'HMM/hmm_train.npy'
        self.hmm_models = np.load(modelFileName).item()
        self.names = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']
        self.alphas={}


        self.obs_group=QWidget(self)
        self.obs_group_layout=QVBoxLayout()
        self.obs_group.setLayout(self.obs_group_layout)
        obs_label=QLabel("Human Observations")
        obs_label.setAlignment(Qt.AlignCenter)
        self.obs_group_layout.addWidget(obs_label)

        self.layoutStratiform = QHBoxLayout()
        self.widgetStratiform = QWidget(self)
        self.widgetStratiform.setLayout(self.layoutStratiform)
        self.obs_group_layout.addWidget(self.widgetStratiform)
        #self.lStratiform = QLabel('Stratiform            ')
        self.layoutStratiform.addWidget(QWidget())
        self.lStratiform = QLabel('Genus 0:')
        self.layoutStratiform.addWidget(self.lStratiform)
        self.stratiformGroup = QButtonGroup(self.widgetStratiform)
        self.rStratiformYes = QRadioButton('Yes')
        self.stratiformGroup.addButton(self.rStratiformYes)
        self.rStratiformNo = QRadioButton('No')
        self.stratiformGroup.addButton(self.rStratiformNo)
        self.layoutStratiform.addWidget(self.rStratiformYes)
        self.layoutStratiform.addWidget(self.rStratiformNo)
        self.layoutStratiform.addWidget(QWidget())

        self.layoutCirriform = QHBoxLayout()
        self.widgetCirriform = QWidget(self)
        self.widgetCirriform.setLayout(self.layoutCirriform)
        self.obs_group_layout.addWidget(self.widgetCirriform)
        #self.lCirriform = QLabel('Cirriform               ')
        self.layoutCirriform.addWidget(QWidget())
        self.lCirriform = QLabel('Genus 1:')
        self.layoutCirriform.addWidget(self.lCirriform)
        self.CirriformGroup = QButtonGroup(self.widgetCirriform)
        self.rCirriformYes = QRadioButton('Yes')
        self.CirriformGroup.addButton(self.rCirriformYes)
        self.rCirriformNo = QRadioButton('No')
        self.CirriformGroup.addButton(self.rCirriformNo)
        self.layoutCirriform.addWidget(self.rCirriformYes)
        self.layoutCirriform.addWidget(self.rCirriformNo)
        self.layoutCirriform.addWidget(QWidget())

        self.layoutStratoCumuliform = QHBoxLayout()
        self.widgetStratoCumuliform = QWidget(self)
        self.widgetStratoCumuliform.setLayout(self.layoutStratoCumuliform)
        self.obs_group_layout.addWidget(self.widgetStratoCumuliform)
        #self.lStratoCumuliform = QLabel('StratoCumuliform ')
        self.layoutStratoCumuliform.addWidget(QWidget())
        self.lStratoCumuliform = QLabel('Genus 2:')
        self.layoutStratoCumuliform.addWidget(self.lStratoCumuliform)
        self.StratoCumuliformGroup = QButtonGroup(self.widgetStratoCumuliform)
        self.rStratoCumuliformYes = QRadioButton('Yes')
        self.StratoCumuliformGroup.addButton(self.rStratoCumuliformYes)
        self.rStratoCumuliformNo = QRadioButton('No')
        self.StratoCumuliformGroup.addButton(self.rStratoCumuliformNo)
        self.layoutStratoCumuliform.addWidget(self.rStratoCumuliformYes)
        self.layoutStratoCumuliform.addWidget(self.rStratoCumuliformNo)
        self.layoutStratoCumuliform.addWidget(QWidget())


        self.layoutCumuliform = QHBoxLayout()
        self.widgetCumuliform = QWidget(self)
        self.widgetCumuliform.setLayout(self.layoutCumuliform)
        self.obs_group_layout.addWidget(self.widgetCumuliform)
        #self.lCumuliform = QLabel('Cumuliform          ')
        self.layoutCumuliform.addWidget(QWidget())
        self.lCumuliform = QLabel('Genus 3:')
        self.layoutCumuliform.addWidget(self.lCumuliform)
        self.CumuliformGroup = QButtonGroup(self.widgetCumuliform)
        self.rCumuliformYes = QRadioButton('Yes')
        self.CumuliformGroup.addButton(self.rCumuliformYes)
        self.rCumuliformNo = QRadioButton('No')
        self.CumuliformGroup.addButton(self.rCumuliformNo)
        self.layoutCumuliform.addWidget(self.rCumuliformYes)
        self.layoutCumuliform.addWidget(self.rCumuliformNo)
        self.layoutCumuliform.addWidget(QWidget())


        self.layoutCumulonibiform = QHBoxLayout()
        self.widgetCumulonibiform = QWidget(self)
        self.widgetCumulonibiform.setLayout(self.layoutCumulonibiform)
        self.obs_group_layout.addWidget(self.widgetCumulonibiform)
        self.layoutCumulonibiform.addWidget(QWidget())
        # self.lCumulonibiform = QLabel('Cumulonibiform   ')
        self.lCumulonibiform = QLabel('Genus 4:')
        self.layoutCumulonibiform.addWidget(self.lCumulonibiform)
        self.CumulonibiformGroup = QButtonGroup(self.widgetCumulonibiform)
        self.rCumulonibiformYes = QRadioButton('Yes')
        self.CumulonibiformGroup.addButton(self.rCumulonibiformYes)
        self.rCumulonibiformNo = QRadioButton('No')
        self.CumulonibiformGroup.addButton(self.rCumulonibiformNo)
        self.layoutCumulonibiform.addWidget(self.rCumulonibiformYes)
        self.layoutCumulonibiform.addWidget(self.rCumulonibiformNo)
        self.layoutCumulonibiform.addWidget(QWidget())


        self.layoutspacing = QHBoxLayout()
        self.updateContainer=QWidget()
        self.updateContainer.setLayout(self.layoutspacing)
        self.layoutspacing.addWidget(QWidget())

        self.updatebutton=QPushButton("Update",self)
        #  self.updatebutton.setFixedWidth(100)
        self.layoutspacing.addWidget(self.updatebutton)
        self.layoutspacing.addWidget(QWidget())
        self.obs_group_layout.addWidget(self.updateContainer)
        self.obs_group_layout.addWidget(QWidget())
        self.obs_group_layout.addWidget(QWidget())
        self.obs_group_layout.addWidget(QWidget())
        self.layout.addWidget(self.obs_group,0,1)


        # Probability graph
        self.figure_prob=Figure()
        self.probability=FigureCanvas(self.figure_prob)
        self.prob_ax=self.probability.figure.subplots()
        names = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']
        graph_names=['0','Genus0','Genus1','Genus2','Genus3','Genus4',]
        self.alphas={}
        self.probs={}
        for i in names:
            self.alphas[i]=[-1,-1]
            self.probs[i]=.2
            #  self.probs[i]=np.random.uniform()
        for i in names:
            self.probs[i]/=sum(self.probs.values())
        self.prob_ax.bar(range(5),self.probs.values())
        #DEBUG
        #  self.prob_ax.bar(range(2),self.probs.values())
        self.prob_ax.set_ylim(0,1)
        self.prob_ax.set_ylabel('Probability')
        self.prob_ax.set_xticklabels(graph_names)
        self.prob_ax.figure.canvas.draw()
        self.layout.addWidget(self.probability,0,0)
        
        # Intensity and Data on same tab
        self.tabs_top=QTabWidget(self)

        # Intensity graph
        self.figure_int=Figure()
        self.intensity=FigureCanvas(self.figure_int)
        self.int_ax=self.intensity.figure.subplots()
        self.intensity_data=self.get_intensity()
        self.int_ax.plot([0],self.intensity_data[0])
        self.int_ax.set_ylim(0,np.max(self.intensity_data)+1)
        self.int_ax.set_xlabel('Time Steps')
        self.figure_int.tight_layout()
        self.int_ax.figure.canvas.draw()
        self.tabs_top.addTab(self.intensity,'Intensity')

        # Satellite image
        self.figure_sat=Figure()
        self.satellite=FigureCanvas(self.figure_sat)
        self.sat_ax=self.satellite.figure.subplots()
        self.satellite_data=self.make_some_data()
        self.maxsig=np.amax(self.satellite_data)
        self.sat_ax.imshow(self.satellite_data[0],vmax=self.maxsig,cmap='Greys_r')
        self.sat_ax.axis('off')
        self.sat_ax.figure.canvas.draw()
        self.tabs_top.addTab(self.satellite,'Satellite Image')

        self.layout.addWidget(self.tabs_top,1,0)


        #TODO
        self.table=self.DirPrior(5)


        self.reference=QLabel(self)
        self.reference.setPixmap(QPixmap('data_clean.png'))
        self.layout.addWidget(self.reference,1,1)

        self.updatebutton.clicked.connect(lambda: self.updateProbs(updateType='human'))
        self.updatebutton.clicked.connect(lambda: self.moment_matching())

    def advice(self):
        QMessageBox.about(self,'Heads Up!','Targets in this region are known to be of type 3')

    def reset(self):

        self.frame=0
        self.obs=[]
        self.alphas={}

        self.int_ax.clear()
        self.intensity_data=self.get_intensity()
        self.int_ax.plot([0],self.intensity_data[0])
        self.int_ax.set_ylim(0,np.max(self.intensity_data)+1)
        self.int_ax.set_xlabel('Time Steps')
        self.figure_int.tight_layout()
        self.int_ax.figure.canvas.draw()

        self.sat_ax.clear()
        self.sat_ax.imshow(self.satellite_data[0],vmax=self.maxsig,cmap='Greys_r')
        self.sat_ax.axis('off')
        self.sat_ax.figure.canvas.draw()

        graph_names=['0','Genus0','Genus1','Genus2','Genus3','Genus4',]
        self.prob_ax.clear()
        self.probs={}
        for i in self.names:
            self.alphas[i]=[-1,-1]
            self.probs[i]=.2
            #  self.probs[i]=np.random.uniform()
        for i in self.names:
            self.probs[i]/=sum(self.probs.values())
        self.prob_ax.bar(range(5),self.probs.values())
        #DEBUG
        #  self.prob_ax.bar(range(2),self.probs.values())
        self.prob_ax.set_ylim(0,1)
        self.prob_ax.set_ylabel('Probability')
        self.prob_ax.set_xticklabels(graph_names)
        self.prob_ax.figure.canvas.draw()

    def make_some_data(self):
        img_path="data_sim/boulder.png"
        region_coordinates={'latmin':0,'latmax':0,'lonmin':0,'lonmax':0}
        Boulder=Region('Boulder',img_path,region_coordinates)
        Boulder.initPointTargets()
        Boulder.generateLayers()
        total_targets=np.zeros((100,100,100))
        for i, (gt_layer,sb_layer,pb_layer) in enumerate(zip(Boulder.ground_truth_layers,Boulder.shake_base_layers,Boulder.pixel_bleed_layers)):
            total_targets=total_targets+gt_layer+sb_layer+pb_layer
        total=total_targets+Boulder.noise_layer+Boulder.structured_noise_layer+Boulder.shotgun_noise_layer
        return total

    def get_intensity(self):
        genus=np.random.randint(5)
        print genus
        model=Cumuliform(genus=genus,weather=False)
        intensity_data=model.intensityModel+np.random.normal(0,2,(len(model.intensityModel)))
        for j in range(len(intensity_data)):
            intensity_data[j]=max(intensity_data[j],1e-5)
        return intensity_data

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
        #  self.table_compare=np.zeros((2*num_tar*num_tar,2*num_tar))
        for X in range(num_tar):
            for prev_obs in range(2*num_tar):
                self.theta2_correct[X*2*num_tar+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=table_real[X,prev_obs,:])
                #  self.table_compare[X*2*num_tar+prev_obs,:]=table_real[X,prev_obs,:]

    def updateSat(self):
        self.sat_ax.clear()
        self.sat_ax.imshow(self.satellite_data[self.frame],vmax=self.maxsig,cmap='Greys_r')
        self.sat_ax.axis('off')
        self.sat_ax.figure.canvas.draw()
        self.frame+=1
        if self.frame%10==0:
            self.advice()

    def updateIntensity(self):
        self.int_ax.clear()
        self.int_ax.plot(range(self.frame),self.intensity_data[:self.frame])
        self.int_ax.set_ylim(0,np.max(self.intensity_data)+1)
        self.int_ax.set_xlabel('Time Steps')
        self.int_ax.figure.canvas.draw()
        
        #  if self.frame==11:
        #      self.intel.document().setPlainText("Heads up: This region in known to have target of type 3")


    def updateProbs(self,updateType=None):
        #  modelFileName = 'HMM/hmm_train.npy'
        #  models = np.load(modelFileName).item()
        #  hmm=HMM_Classification()
        #  names = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']
        graph_names=['0','Genus0','Genus1','Genus2','Genus3','Genus4',]
        if updateType=='machine':
            data=self.intensity_data[self.frame]
            #forward algorithm
            for i in self.names:
                self.alphas[i]=self.hmm.continueForward(data,self.hmm_models[i],self.alphas[i])
                self.probs[i]=self.probs[i]*sum(self.alphas[i])
            #noramlize
            suma=sum(self.probs.values())
            for i in self.names:
                self.probs[i]/=suma
            post_probs=self.probs.values()
        elif updateType=='human':
            if(self.rStratiformYes.isChecked()):
                self.obs.append(0)
            elif(self.rStratiformNo.isChecked()):
                self.obs.append(1)

            if(self.rCirriformYes.isChecked()):
                self.obs.append(2)
            elif(self.rCirriformNo.isChecked()):
                self.obs.append(3)

            if(self.rStratoCumuliformYes.isChecked()):
                self.obs.append(4)
            elif(self.rStratoCumuliformNo.isChecked()):
                self.obs.append(5)

            if(self.rCumuliformYes.isChecked()):
                self.obs.append(6)
            elif(self.rCumuliformNo.isChecked()):
                self.obs.append(7)

            if(self.rCumulonibiformYes.isChecked()):
                self.obs.append(8)
            elif(self.rCumulonibiformNo.isChecked()):
                self.obs.append(9)

            # reset all buttons
            self.stratiformGroup.setExclusive(False)
            self.CirriformGroup.setExclusive(False)
            self.StratoCumuliformGroup.setExclusive(False)
            self.CumuliformGroup.setExclusive(False)
            self.CumulonibiformGroup.setExclusive(False)
            self.rStratiformYes.setChecked(False)
            self.rStratiformNo.setChecked(False)
            self.rCirriformYes.setChecked(False)
            self.rCirriformNo.setChecked(False)
            self.rStratoCumuliformYes.setChecked(False)
            self.rStratoCumuliformNo.setChecked(False)
            self.rCumuliformYes.setChecked(False)
            self.rCumuliformNo.setChecked(False)
            self.rCumulonibiformYes.setChecked(False)
            self.rCumulonibiformNo.setChecked(False)
            
            postX=copy.deepcopy(self.probs)
            num_tar=5
            # only learning theta2 on 2+ observations
            if len(self.obs)>1:
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
                        index=self.select_param(self.names.index(i),self.obs[0])
                        if index%2==0:
                            likelihood=self.theta1[index]
                        else:
                            likelihood=self.theta1[index]/(num_tar-1)
                        # likelihood from theta2
                        count=0
                        for value in self.obs[1:]:
                            indicies=self.select_param(self.names.index(i),value,self.obs[count])
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
                    for i in range(len(self.obs)-1):
                        indicies=self.select_param(X,self.obs[i+1],self.obs[i])
                        alphas[indicies[0],indicies[1]]+=1
                    for j in range(4):
                        theta2[j,:]=np.random.dirichlet(alphas[j,:])
                    if n%5==0:
                        self.theta2_samples[int((n-self.burn_in)/5),:,:]=theta2

                # take max likelihood of X for next obs
                post_probs=np.mean(all_post,axis=0)[0]
            # using only theat1 on first observation
            else:
                for i in self.names:
                    # likelihood from theta1 (not full dist, assuming we know theta1)
                    index=self.select_param(self.names.index(i),self.obs[0])
                    likelihood=self.theta1[index]
                    postX[i]=self.probs[i]*likelihood
                # normalize and set final values
                suma=sum(postX.values())
                for i in self.names:
                    postX[i]=np.log(postX[i])-np.log(suma) 
                    postX[i]=np.exp(postX[i])
                post_probs=postX.values()

        self.prob_ax.clear()
        self.prob_ax.set_ylim(0,1)
        self.prob_ax.bar(range(5),post_probs)
        #  print self.probs.values()
        self.prob_ax.set_xticklabels(graph_names)
        self.prob_ax.set_ylabel('Probability')
        self.prob_ax.figure.canvas.draw()

        self.post_probs=post_probs

    def moment_matching(self,graph=False):
        if (len(self.obs)>1) and (max(self.post_probs)>self.threshold):
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
                            if (n==0) and (k==0):
                                plt.figure()
                                plt.hist(samples,bins=20,density=True)
                                x=np.linspace(0,1)
                                plt.plot(x,scipy.stats.beta.pdf(x,alphak,sum(self.theta2[n,:])-alphak))
                                plt.xlabel(r'$\theta_2$')
                                plt.ylabel(r'$p(\theta_2)$')
                                plt.title("Moment Matching for TP,TP")
                                plt.show()
                                sys.exit()
            self.reset()
        else:
            pass

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


if __name__ == '__main__':
    app=QApplication(sys.argv)
    ex=InterfaceWindow()
    sys.exit(app.exec_())
