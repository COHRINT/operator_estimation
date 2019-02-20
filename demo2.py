from PyQt5 import QtGui,QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import copy

sys.path.append('data_sim')
sys.path.append('HMM')
from regions import Region
from DynamicsProfiles import *
from hmm_classification import HMM_Classification
from fusion import Fusion

from matplotlib.backends.backend_qt4agg import FigureCanvas
from matplotlib.figure import Figure


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
            self.updateTimer.timeout.connect(self.HMM_update)

            self.show()

    def initialize(self):
        self.fusion=Fusion()
        self.frame=0
        self.obs=[]
        self.hmm=HMM_Classification()
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
        self.layout.addWidget(self.obs_group,1,1)


        # Probability graph
        self.figure_prob=Figure()
        self.probability=FigureCanvas(self.figure_prob)
        self.prob_ax=self.probability.figure.subplots()
        names = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']
        self.graph_names=['0','Genus0','Genus1','Genus2','Genus3','Genus4',]
        self.fusion.probs={}
        for i in names:
            self.alphas[i]=[-1,-1]
            self.fusion.probs[i]=.2
            #  self.probs[i]=np.random.uniform()
        for i in names:
            self.fusion.probs[i]/=sum(self.fusion.probs.values())
        self.prob_ax.bar(range(5),self.fusion.probs.values())
        self.prob_ax.set_ylim(0,1)
        self.prob_ax.set_ylabel('Probability')
        self.prob_ax.set_xticklabels(self.graph_names)
        self.prob_ax.figure.canvas.draw()
        self.layout.addWidget(self.probability,1,0)
        
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

        self.reference=QLabel(self)
        self.reference.setPixmap(QPixmap('data_clean.png'))
        self.tabs_top.addTab(self.reference,'Reference')
        self.layout.addWidget(self.tabs_top,0,1)

        # Satellite image
        self.figure_sat=Figure()
        self.satellite=FigureCanvas(self.figure_sat)
        self.sat_ax=self.satellite.figure.subplots()
        self.satellite_data=self.make_some_data()
        self.maxsig=np.amax(self.satellite_data)
        self.sat_ax.imshow(self.satellite_data[0],vmax=self.maxsig,cmap='Greys_r')
        self.sat_ax.axis('off')
        self.sat_ax.figure.canvas.draw()
        self.layout.addWidget(self.satellite,0,0)


        self.table=self.fusion.DirPrior(5)
        self.give_advice=np.random.choice([1,0],p=[0.95,0.05])
        self.help=False

        self.updatebutton.clicked.connect(self.getObs)
        self.updatebutton.clicked.connect(lambda: self.fusion.moment_matching(self.obs))
        self.updatebutton.clicked.connect(self.reset)

    def HMM_update(self):
        if not self.help:
            data=self.intensity_data[self.frame]
            #forward algorithm
            for i in self.names:
                self.alphas[i]=self.hmm.continueForward(data,self.hmm_models[i],self.alphas[i])
                self.fusion.probs[i]=self.fusion.probs[i]*sum(self.alphas[i])
            #noramlize
            suma=sum(self.fusion.probs.values())
            for i in self.names:
                self.fusion.probs[i]/=suma
            post_probs=self.fusion.probs.values()

            self.updateProbGraph(post_probs)

    def getObs(self):
        self.help=True

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
        
        post_probs=self.fusion.updateProbs(self.obs)
        self.updateProbGraph(post_probs)

    def reset(self):
        if self.fusion.reset:
            self.give_advice=np.random.choice([1,0],p=[0.95,0.05])
            self.help=False
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

            self.prob_ax.clear()
            self.fusion.probs={}
            for i in self.names:
                self.alphas[i]=[-1,-1]
                self.fusion.probs[i]=.2
                #  self.probs[i]=np.random.uniform()
            for i in self.names:
                self.fusion.probs[i]/=sum(self.fusion.probs.values())
            self.prob_ax.bar(range(5),self.fusion.probs.values())
            #DEBUG
            #  self.prob_ax.bar(range(2),self.probs.values())
            self.prob_ax.set_ylim(0,1)
            self.prob_ax.set_ylabel('Probability')
            self.prob_ax.set_xticklabels(self.graph_names)
            self.prob_ax.figure.canvas.draw()
        else:
            pass

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
        self.genus=genus
        model=Cumuliform(genus=genus,weather=False)
        intensity_data=model.intensityModel+np.random.normal(0,2,(len(model.intensityModel)))
        for j in range(len(intensity_data)):
            intensity_data[j]=max(intensity_data[j],1e-5)
        return intensity_data

    def updateSat(self):
        self.sat_ax.clear()
        self.sat_ax.imshow(self.satellite_data[self.frame],vmax=self.maxsig,cmap='Greys_r')
        self.sat_ax.axis('off')
        self.sat_ax.figure.canvas.draw()
        self.frame+=1
        if (self.frame==10) and (self.give_advice):
            self.advice()

    def updateIntensity(self):
        self.int_ax.clear()
        self.int_ax.plot(range(self.frame),self.intensity_data[:self.frame])
        self.int_ax.set_ylim(0,np.max(self.intensity_data)+1)
        self.int_ax.set_xlabel('Time Steps')
        self.int_ax.figure.canvas.draw()

    def updateProbGraph(self,post_probs):
        self.prob_ax.clear()
        self.prob_ax.set_ylim(0,1)
        self.prob_ax.bar(range(5),post_probs)
        #  print self.probs.values()
        self.prob_ax.set_xticklabels(self.graph_names)
        self.prob_ax.set_ylabel('Probability')
        self.prob_ax.figure.canvas.draw()
        
    def advice(self):
        message="Targets in this region are known to be of type %d" % self.genus
        QMessageBox.about(self,'Heads Up!',message)

if __name__ == '__main__':
    app=QApplication(sys.argv)
    ex=InterfaceWindow()
    sys.exit(app.exec_())
