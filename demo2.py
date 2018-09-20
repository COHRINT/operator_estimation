from PyQt5 import QtGui,QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import os
import numpy as np
import scipy.stats
from scipy import spatial
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import copy

sys.path.append('../../../scenario_simulator/modules')
sys.path.append('../../HMM/modules')
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
            #  self.updateTimer.timeout.connect(lambda: self.updateProbs(updateType='machine'))
            
            self.show()

    def initialize(self):

        self.frame=0
        self.obs=[]


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
        self.rStratiformNull = QRadioButton('IDK')
        self.stratiformGroup.addButton(self.rStratiformNull)
        self.rStratiformNo = QRadioButton('No')
        self.stratiformGroup.addButton(self.rStratiformNo)
        self.layoutStratiform.addWidget(self.rStratiformYes)
        self.layoutStratiform.addWidget(self.rStratiformNo)
        #  self.layoutStratiform.addWidget(self.rStratiformNull)
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
        self.rCirriformNull = QRadioButton('IDK')
        self.CirriformGroup.addButton(self.rCirriformNull)
        self.rCirriformNo = QRadioButton('No')
        self.CirriformGroup.addButton(self.rCirriformNo)
        self.layoutCirriform.addWidget(self.rCirriformYes)
        self.layoutCirriform.addWidget(self.rCirriformNo)
        #  self.layoutCirriform.addWidget(self.rCirriformNull)
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
        self.rStratoCumuliformNull = QRadioButton('IDK')
        self.StratoCumuliformGroup.addButton(self.rStratoCumuliformNull)
        self.rStratoCumuliformNo = QRadioButton('No')
        self.StratoCumuliformGroup.addButton(self.rStratoCumuliformNo)
        self.layoutStratoCumuliform.addWidget(self.rStratoCumuliformYes)
        self.layoutStratoCumuliform.addWidget(self.rStratoCumuliformNo)
        #  self.layoutStratoCumuliform.addWidget(self.rStratoCumuliformNull)
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
        self.rCumuliformNull = QRadioButton('IDK')
        self.CumuliformGroup.addButton(self.rCumuliformNull)
        self.rCumuliformNo = QRadioButton('No')
        self.CumuliformGroup.addButton(self.rCumuliformNo)
        self.layoutCumuliform.addWidget(self.rCumuliformYes)
        self.layoutCumuliform.addWidget(self.rCumuliformNo)
        #  self.layoutCumuliform.addWidget(self.rCumuliformNull)
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
        self.rCumulonibiformNull = QRadioButton('IDK')
        self.CumulonibiformGroup.addButton(self.rCumulonibiformNull)
        self.rCumulonibiformNo = QRadioButton('No')
        self.CumulonibiformGroup.addButton(self.rCumulonibiformNo)
        self.layoutCumulonibiform.addWidget(self.rCumulonibiformYes)
        self.layoutCumulonibiform.addWidget(self.rCumulonibiformNo)
        #  self.layoutCumulonibiform.addWidget(self.rCumulonibiformNull)
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

        # Genus dirichlet distributions
        self.tabs_bot=QTabWidget(self)

        self.table=self.DirPrior()
        self.theta1=np.zeros((5,10))
        for X in range(5):
            self.theta1[X,:]=scipy.stats.dirichlet.mean(alpha=np.mean(self.table,axis=1)[X,:])

        self.figure_gen0=Figure()
        self.genus0=FigureCanvas(self.figure_gen0)
        self.gen0_ax=self.genus0.figure.add_subplot(111,projection='3d')
        self.gen0_ax.set_xlabel('Previous Obs')
        self.gen0_ax.set_ylabel('Next Obs')
        self.updateDir(0)
        self.tabs_bot.addTab(self.genus0,'Genus 0')

        self.figure_gen1=Figure()
        self.genus1=FigureCanvas(self.figure_gen1)
        self.gen1_ax=self.genus1.figure.add_subplot(111,projection='3d')
        self.gen1_ax.set_xlabel('Previous Obs')
        self.gen1_ax.set_ylabel('Next Obs')
        self.updateDir(1)
        self.tabs_bot.addTab(self.genus1,'Genus 1')

        self.figure_gen2=Figure()
        self.genus2=FigureCanvas(self.figure_gen2)
        self.gen2_ax=self.genus2.figure.add_subplot(111,projection='3d')
        self.gen2_ax.set_xlabel('Previous Obs')
        self.gen2_ax.set_ylabel('Next Obs')
        self.updateDir(2)
        self.tabs_bot.addTab(self.genus2,'Genus 2')

        self.figure_gen3=Figure()
        self.genus3=FigureCanvas(self.figure_gen3)
        self.gen3_ax=self.genus3.figure.add_subplot(111,projection='3d')
        self.gen3_ax.set_xlabel('Previous Obs')
        self.gen3_ax.set_ylabel('Next Obs')
        self.updateDir(3)
        self.tabs_bot.addTab(self.genus3,'Genus 3')

        self.figure_gen4=Figure()
        self.genus4=FigureCanvas(self.figure_gen4)
        self.gen4_ax=self.genus4.figure.add_subplot(111,projection='3d')
        self.gen4_ax.set_xlabel('Previous Obs')
        self.gen4_ax.set_ylabel('Next Obs')
        self.updateDir(4)
        self.tabs_bot.addTab(self.genus4,'Genus 4')

        self.layout.addWidget(self.tabs_bot,1,1)

        self.updatebutton.clicked.connect(lambda: self.updateProbs(updateType='human'))
        self.updatebutton.clicked.connect(lambda: self.updateDir(0))
        self.updatebutton.clicked.connect(lambda: self.updateDir(1))
        self.updatebutton.clicked.connect(lambda: self.updateDir(2))
        self.updatebutton.clicked.connect(lambda: self.updateDir(3))
        self.updatebutton.clicked.connect(lambda: self.updateDir(4))

    def make_some_data(self):
        img_path="../imgs/boulder.png"
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
        #  genus=4
        target=Cumuliform(genus=genus,weather=True)
        true_data=[]
        for frame in target.intensityModel:
            frame+=max(np.random.normal()*2,0)
            true_data.append(frame)
        return true_data

    def DirPrior(self):
        table=np.zeros((5,10,10))
        base_table=np.ones((5,10))
        #  base_table*=0.1
        for i in range(5):
            base_table[i,2*i]*=2
            for j in range(5):
                if i==j:
                    base_table[i,2*j+1]*=0.5
                else:
                    base_table[i,2*j+1]*=2
                    base_table[i,2*j]*=0.5
        for i in range(10):
            table[:,:,i]=base_table
        for i in range(10):
            table[:,i,i]*=3
        table=np.swapaxes(table,1,2)

        table*=5
        #  print table
        return table

    def updateSat(self):
        self.sat_ax.clear()
        self.sat_ax.imshow(self.satellite_data[self.frame],vmax=self.maxsig,cmap='Greys_r')
        self.sat_ax.axis('off')
        self.sat_ax.figure.canvas.draw()
        self.frame+=1

    def updateIntensity(self):
        self.int_ax.clear()
        self.int_ax.plot(range(self.frame),self.intensity_data[:self.frame])
        self.int_ax.set_ylim(0,np.max(self.intensity_data)+1)
        self.int_ax.set_xlabel('Time Steps')
        self.int_ax.figure.canvas.draw()

    def updateDir(self,GenType):
        options=['0-Y','0-N','1-Y','1-N','2-Y','2-N','3-Y','3-N','4-Y','4-N']
        means=scipy.stats.dirichlet.mean(alpha=np.reshape(self.table,(1,self.table.shape[0]*self.table.shape[1]*self.table.shape[2]))[0])
        new_means=np.reshape(np.array(means),(5,10,10))

        _x=np.arange(10)
        _y=np.arange(10)
        _xx,_yy=np.meshgrid(_x,_y)
        x,y=_xx.ravel(),_yy.ravel()
        top=new_means[GenType,:,:]
        top=np.reshape(top,(1,100))[0]
        bottom=np.zeros_like(top)
        width=depth=1

        if GenType==0:
            self.gen0_ax.clear()
            self.gen0_ax.bar3d(x,y,bottom,width,depth,top,shade=True,color='c')
            self.gen0_ax.set_xticklabels(options)
            self.gen0_ax.set_yticklabels(options)
            self.gen0_ax.figure.canvas.draw()
        elif GenType==1:
            self.gen1_ax.clear()
            self.gen1_ax.bar3d(x,y,bottom,width,depth,top,shade=True,color='b')
            self.gen1_ax.set_xticklabels(options)
            self.gen1_ax.set_yticklabels(options)
            self.gen1_ax.figure.canvas.draw()
        elif GenType==2:
            self.gen2_ax.clear()
            self.gen2_ax.bar3d(x,y,bottom,width,depth,top,shade=True,color='r')
            self.gen2_ax.set_xticklabels(options)
            self.gen2_ax.set_yticklabels(options)
            self.gen2_ax.figure.canvas.draw()
        elif GenType==3:
            self.gen3_ax.clear()
            self.gen3_ax.bar3d(x,y,bottom,width,depth,top,shade=True,color='y')
            self.gen3_ax.set_xticklabels(options)
            self.gen3_ax.set_yticklabels(options)
            self.gen3_ax.figure.canvas.draw()
        elif GenType==4:
            self.gen4_ax.clear()
            self.gen4_ax.bar3d(x,y,bottom,width,depth,top,shade=True,color='g')
            self.gen4_ax.set_xticklabels(options)
            self.gen4_ax.set_yticklabels(options)
            self.gen4_ax.figure.canvas.draw()


    def updateProbs(self,updateType=None):
        modelFileName = '../../HMM/data/histModels_final.npy'
        models = np.load(modelFileName).item()
        hmm=HMM_Classification()
        names = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']
        graph_names=['0','Genus0','Genus1','Genus2','Genus3','Genus4',]
        if updateType=='machine':
            data=self.intensity_data[self.frame]
            #forward algorithm
            for i in names:
                self.alphas[i]=hmm.continueForward(data,models[i],self.alphas[i])
                self.probs[i]=self.probs[i]*sum(self.alphas[i])
            #noramlize
            suma=sum(self.probs.values())
            for i in names:
                self.probs[i]/=suma
        elif updateType=='human':
            if(self.rStratiformYes.isChecked()):
                self.obs.append(0)
                #  self.rStratiformYes.setChecked(False)
            elif(self.rStratiformNo.isChecked()):
                self.obs.append(1)
                #  self.rStratiformNo.setChecked(False)

            if(self.rCirriformYes.isChecked()):
                self.obs.append(2)
                #  self.rCirriformYes.setChecked(False)
            elif(self.rCirriformNo.isChecked()):
                self.obs.append(3)
                #  self.rCirriformNo.setChecked(False)

            if(self.rStratoCumuliformYes.isChecked()):
                self.obs.append(4)
                #  self.rStratoCumliformYes.setChecked(False)
            elif(self.rStratoCumuliformNo.isChecked()):
                self.obs.append(5)
                #  self.rStratoCumliformNo.setChecked(False)

            if(self.rCumuliformYes.isChecked()):
                self.obs.append(6)
                #  self.rCumuliformYes.setChecked(False)
            elif(self.rCumuliformNo.isChecked()):
                self.obs.append(7)
                #  self.rCumuliformNo.setChecked(False)

            if(self.rCumulonibiformYes.isChecked()):
                self.obs.append(8)
                #  self.rCumulonibiformYes.setChecked(False)
            elif(self.rCumulonibiformNo.isChecked()):
                self.obs.append(9)
                #  self.rCumulonibiformNo.setChecked(False)
            
            # initialize Dir sample
            theta2_static=np.zeros((50,10))
            postX=copy.copy(self.probs)
            all_post=np.zeros((1,5))
            all_theta2=np.zeros((1,50,10))
            all_theta2[:]=np.nan
            for X in range(5):
                for prev_obs in range(10):
                    #  print self.table[X,prev_obs,:]
                    theta2_static[X*10+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=self.table[X,prev_obs,:])
                    #  print theta2_static[X*10+prev_obs,:]
            tic=time.time()
            theta2=copy.copy(theta2_static)
            for n in range(2000):
                for i in names:
                    likelihood=self.theta1[names.index(i),self.obs[0]]
                    # sample from theta2
                    if len(self.obs)>1:
                        for value in self.obs[1:]:
                            likelihood*=theta2[names.index(i)*10+self.obs[self.obs.index(value)-1],value]
                            #  print self.table[names.index(i),self.obs[self.obs.index(value)-1],value]
                            #  print theta2[names.index(i)*10+self.obs[self.obs.index(value)-1],value]
                    #  print likelihood
                    postX[i]=self.probs[i]*likelihood
                suma=sum(postX.values())
                # normalize
                for i in names:
                    postX[i]=np.log(postX[i])-np.log(suma) 
                    postX[i]=np.exp(postX[i])
                if n%5==0:
                    all_post=np.append(all_post,[postX.values()],axis=0)
                # sample from X
                X=np.random.choice(range(5),p=postX.values())
                alphas=copy.copy(self.table)
                theta2=copy.copy(theta2_static)
                theta2_to_store=np.zeros((50,10))
                theta2_to_store[:]=np.nan
                if len(self.obs)>1:
                    for value in self.obs[1:]:
                        alphas[X,self.obs[self.obs.index(value)-1],value]+=1
                        theta2[X*10+self.obs[self.obs.index(value)-1],:]=np.random.dirichlet(alphas[X,self.obs[self.obs.index(value)-1],:])
                        if n%5==0:
                            theta2_to_store[X*10+self.obs[self.obs.index(value)-1],:]=theta2[X*10+self.obs[self.obs.index(value)-1],:]
                            all_theta2=np.append(all_theta2,[theta2_to_store],axis=0)
                    #  print theta2
            theta2_change=(theta2_static-np.nan_to_num(np.nanmean(all_theta2,axis=0)))/theta2_static
            theta2_change[theta2_change==1]=0
            if np.amin(theta2_change)<0:
                rows=list(set(np.where(theta2_change<0)[0]))
                for i in rows:
                    smallest=min(theta2_change[i,:])
                    theta2_change[i,:]+=smallest
            theta2_change=theta2_change.reshape((5,10,10))
            #  print theta2_change
            post_probs=np.mean(all_post[1:],axis=0)
            for i in names:
                self.probs[i]=post_probs[names.index(i)]
            if len(self.obs)>1:
                self.table+=((theta2_change)/np.sum(theta2_change))*len(self.obs)
            #  print self.table

            print time.time()-tic

        self.prob_ax.clear()
        self.prob_ax.set_ylim(0,1)
        self.prob_ax.bar(range(5),self.probs.values())
        #  print self.probs.values()
        self.prob_ax.set_xticklabels(graph_names)
        self.prob_ax.set_ylabel('Probability')
        self.prob_ax.figure.canvas.draw()




if __name__ == '__main__':
    app=QApplication(sys.argv)
    ex=InterfaceWindow()
    sys.exit(app.exec_())
