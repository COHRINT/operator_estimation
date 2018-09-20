from PyQt4 import QtGui, QtCore
from PyQt4.QtGui import *
import sys
import numpy as np
import scipy.stats
from scipy import spatial
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

sys.path.append('../../../scenario_simulator/modules')
sys.path.append('../../HMM/modules')
from regions import Region
from DynamicsProfiles import *
from hmm_classification import HMM_Classification

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class Window(QtGui.QWidget):
    def __init__(self,total_data,intensity,prior):
        super(Window,self).__init__()
        self.frame=0
        self.all_data=total_data
        self.intensity=intensity
        self.table=prior
        #  self.prev_obs=range(15)
        self.prev_obs=[1,4,7,10,13]
        #  self.total_obs=0
        #  self.matplotlibWidget=MatplotlibWidget(self)
        self.matplotlibWidget2=MatplotlibWidget2(self)
        self.matplotlibWidget3=MatplotlibWidget3(self)
        self.matplotlibWidget4=MatplotlibWidget4(self)
        #  self.matplotlibWidget.move(25,25)
        self.matplotlibWidget2.move(550,300)
        self.matplotlibWidget3.move(25,450)
        self.matplotlibWidget4.move(25,25)
        self.updateProbs()
        means=scipy.stats.dirichlet.mean(alpha=np.reshape(self.table,(1,self.table.shape[0]*self.table.shape[1]*self.table.shape[2]))[0])
        new_means=np.reshape(np.array(means),(5,15,15))
        self.matplotlibWidget2.axis2.clear()

        _x=np.arange(15)
        _y=np.arange(15)
        _xx,_yy=np.meshgrid(_x,_y)
        x,y=_xx.ravel(),_yy.ravel()
        top=new_means[0,:,:]
        top=np.reshape(top,(1,225))[0]
        bottom=np.zeros_like(top)
        width=depth=1
        self.matplotlibWidget2.axis2.set_xlabel('Previous Obs')
        self.matplotlibWidget2.axis2.set_ylabel('Next Obs')
        self.matplotlibWidget2.axis2.set_title('Likelihoods of Genus 0')
        self.matplotlibWidget2.axis2.bar3d(x,y,bottom,width,depth,top,shade=True)
        self.matplotlibWidget2.canvas2.draw()

        self.updateTimer=QtCore.QTimer(self)
        self.interval = 1000
        self.updateTimer.setInterval(self.interval)

        self.updateTimer.start()
        #  self.updateTimer.timeout.connect(self.updateGraph)
        self.updateTimer.timeout.connect(self.updateIntensity)
        #  self.updateTimer.timeout.connect(lambda: self.updateProbs(updateType='machine'))

        self.setGeometry(350,350,1250,900)



        self.updatebutton=QtGui.QPushButton("Update",self)
        self.updatebutton.move(775,200)
        self.updatebutton.clicked.connect(lambda: self.updateProbs(updateType='human'))
        self.updatebutton.clicked.connect(self.updateDir)
        self.layoutStratiform = QtGui.QHBoxLayout()
        self.widgetStratiform = QtGui.QWidget(self)
        self.widgetStratiform.setLayout(self.layoutStratiform)
        self.widgetStratiform.move(700,50)
        #self.lStratiform = QLabel('Stratiform            ')
        self.lStratiform = QLabel('Genus 0:')
        self.layoutStratiform.addWidget(self.lStratiform)
        self.layoutStratiform.addSpacing(10)
        self.stratiformGroup = QtGui.QButtonGroup(self.widgetStratiform)
        self.rStratiformYes = QtGui.QRadioButton('Yes')
        self.stratiformGroup.addButton(self.rStratiformYes)
        self.rStratiformNull = QtGui.QRadioButton('IDK')
        self.stratiformGroup.addButton(self.rStratiformNull)
        self.rStratiformNo = QtGui.QRadioButton('No')
        self.stratiformGroup.addButton(self.rStratiformNo)
        self.layoutStratiform.addWidget(self.rStratiformYes)
        self.layoutStratiform.addWidget(self.rStratiformNo)
        self.layoutStratiform.addWidget(self.rStratiformNull)

        self.layoutCirriform = QtGui.QHBoxLayout()
        self.widgetCirriform = QtGui.QWidget(self)
        self.widgetCirriform.setLayout(self.layoutCirriform)
        self.widgetCirriform.move(700,75)
        #self.lCirriform = QLabel('Cirriform               ')
        self.lCirriform = QLabel('Genus 1:')
        self.layoutCirriform.addWidget(self.lCirriform)
        self.layoutCirriform.addSpacing(10)
        self.CirriformGroup = QtGui.QButtonGroup(self.widgetCirriform)
        self.rCirriformYes = QtGui.QRadioButton('Yes')
        self.CirriformGroup.addButton(self.rCirriformYes)
        self.rCirriformNull = QtGui.QRadioButton('IDK')
        self.CirriformGroup.addButton(self.rCirriformNull)
        self.rCirriformNo = QtGui.QRadioButton('No')
        self.CirriformGroup.addButton(self.rCirriformNo)
        self.layoutCirriform.addWidget(self.rCirriformYes)
        self.layoutCirriform.addWidget(self.rCirriformNo)
        self.layoutCirriform.addWidget(self.rCirriformNull)

        self.layoutStratoCumuliform = QtGui.QHBoxLayout()
        self.widgetStratoCumuliform = QtGui.QWidget(self)
        self.widgetStratoCumuliform.setLayout(self.layoutStratoCumuliform)
        self.widgetStratoCumuliform.move(700,100)
        #self.lStratoCumuliform = QLabel('StratoCumuliform ')
        self.lStratoCumuliform = QLabel('Genus 2:')
        self.layoutStratoCumuliform.addWidget(self.lStratoCumuliform)
        self.layoutStratoCumuliform.addSpacing(10)
        self.StratoCumuliformGroup = QtGui.QButtonGroup(self.widgetStratoCumuliform)
        self.rStratoCumuliformYes = QtGui.QRadioButton('Yes')
        self.StratoCumuliformGroup.addButton(self.rStratoCumuliformYes)
        self.rStratoCumuliformNull = QtGui.QRadioButton('IDK')
        self.StratoCumuliformGroup.addButton(self.rStratoCumuliformNull)
        self.rStratoCumuliformNo = QtGui.QRadioButton('No')
        self.StratoCumuliformGroup.addButton(self.rStratoCumuliformNo)
        self.layoutStratoCumuliform.addWidget(self.rStratoCumuliformYes)
        self.layoutStratoCumuliform.addWidget(self.rStratoCumuliformNo)
        self.layoutStratoCumuliform.addWidget(self.rStratoCumuliformNull)


        self.layoutCumuliform = QtGui.QHBoxLayout()
        self.widgetCumuliform = QtGui.QWidget(self)
        self.widgetCumuliform.setLayout(self.layoutCumuliform)
        self.widgetCumuliform.move(700,125)
        #self.lCumuliform = QLabel('Cumuliform          ')
        self.lCumuliform = QLabel('Genus 3:')
        self.layoutCumuliform.addWidget(self.lCumuliform)
        self.layoutCumuliform.addSpacing(10)
        self.CumuliformGroup = QtGui.QButtonGroup(self.widgetCumuliform)
        self.rCumuliformYes = QtGui.QRadioButton('Yes')
        self.CumuliformGroup.addButton(self.rCumuliformYes)
        self.rCumuliformNull = QtGui.QRadioButton('IDK')
        self.CumuliformGroup.addButton(self.rCumuliformNull)
        self.rCumuliformNo = QtGui.QRadioButton('No')
        self.CumuliformGroup.addButton(self.rCumuliformNo)
        self.layoutCumuliform.addWidget(self.rCumuliformYes)
        self.layoutCumuliform.addWidget(self.rCumuliformNo)
        self.layoutCumuliform.addWidget(self.rCumuliformNull)


        self.layoutCumulonibiform = QtGui.QHBoxLayout()
        self.widgetCumulonibiform = QtGui.QWidget(self)
        self.widgetCumulonibiform.setLayout(self.layoutCumulonibiform)
        self.widgetCumulonibiform.move(700,150)
        # self.lCumulonibiform = QLabel('Cumulonibiform   ')
        self.lCumulonibiform = QLabel('Genus 4:')
        self.layoutCumulonibiform.addWidget(self.lCumulonibiform)
        self.layoutCumulonibiform.addSpacing(10)
        self.CumulonibiformGroup = QtGui.QButtonGroup(self.widgetCumulonibiform)
        self.rCumulonibiformYes = QtGui.QRadioButton('Yes')
        self.CumulonibiformGroup.addButton(self.rCumulonibiformYes)
        self.rCumulonibiformNull = QtGui.QRadioButton('IDK')
        self.CumulonibiformGroup.addButton(self.rCumulonibiformNull)
        self.rCumulonibiformNo = QtGui.QRadioButton('No')
        self.CumulonibiformGroup.addButton(self.rCumulonibiformNo)
        self.layoutCumulonibiform.addWidget(self.rCumulonibiformYes)
        self.layoutCumulonibiform.addWidget(self.rCumulonibiformNo)
        self.layoutCumulonibiform.addWidget(self.rCumulonibiformNull)



        self.show()

    def updateGraph(self):
        maxsig=np.amax(self.all_data)
        self.matplotlibWidget.axis.imshow(self.all_data[self.frame],vmax=maxsig,cmap='Greys_r')
        self.matplotlibWidget.canvas.draw()
        #  self.frame+=1

    def updateIntensity(self):
        self.matplotlibWidget3.axis3.clear()
        self.matplotlibWidget3.axis3.plot(range(self.frame),self.intensity[:self.frame])
        self.matplotlibWidget3.axis3.set_ylim(0,max(self.intensity)+1)
        self.matplotlibWidget3.canvas3.draw()
        self.frame+=1

    def updateDir(self):
        #  self.table[0,self.prev_obs,obs]+=.01
        means=scipy.stats.dirichlet.mean(alpha=np.reshape(self.table,(1,self.table.shape[0]*self.table.shape[1]*self.table.shape[2]))[0])
        new_means=np.reshape(np.array(means),(5,15,15))
        self.matplotlibWidget2.axis2.clear()

        _x=np.arange(15)
        _y=np.arange(15)
        _xx,_yy=np.meshgrid(_x,_y)
        x,y=_xx.ravel(),_yy.ravel()
        top=new_means[2,:,:]
        top=np.reshape(top,(1,225))[0]
        bottom=np.zeros_like(top)
        width=depth=1
        self.matplotlibWidget2.axis2.set_xlabel('Previous Obs')
        self.matplotlibWidget2.axis2.set_ylabel('Next Obs')
        self.matplotlibWidget2.axis2.set_title('Likelihoods of Genus 0')
        self.matplotlibWidget2.axis2.bar3d(x,y,bottom,width,depth,top,shade=True)
        self.matplotlibWidget2.canvas2.draw()
        #  ax1.bar3d(x,y,bottom,width,depth,top,shade=True)
        #  plt.show()

        #  self.prev_obs=random.choice(obs)

    def updateProbs(self,updateType=None):
        def logSumExp(xn):
            maximum=np.max(xn)
            dn=xn-maximum
            sumOfExp=np.sum(np.exp(dn))
            return maximum+np.log(sumOfExp)

        modelFileName = '../../HMM/data/histModels_final.npy'
        models = np.load(modelFileName).item()
        hmm=HMM_Classification()
        names = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']
        if not updateType:
            self.alphas={}
            self.probs={}
            for i in names:
                self.alphas[i]=[-1,-1]
                self.probs[i]=.2
                #  self.probs[i]=np.random.uniform()
            for i in names:
                self.probs[i]/=sum(self.probs.values())
        elif updateType=='machine':
            data=self.intensity[self.frame]
            #forward algorithm
            for i in names:
                self.alphas[i]=hmm.continueForward(data,models[i],self.alphas[i])
                self.probs[i]=self.probs[i]*sum(self.alphas[i])
            #noramlize
            suma=sum(self.probs.values())
            for i in names:
                self.probs[i]/=suma
        elif updateType=='human':
            obs=[]
            if(self.rStratiformYes.isChecked()):
                obs.append(0)
            elif(self.rStratiformNull.isChecked()):
                obs.append(1)
            elif(self.rStratiformNo.isChecked()):
                obs.append(2)

            if(self.rCirriformYes.isChecked()):
                obs.append(3)
            elif(self.rCirriformNull.isChecked()):
                obs.append(4)
            elif(self.rCirriformNo.isChecked()):
                obs.append(5)

            if(self.rStratoCumuliformYes.isChecked()):
                obs.append(6)
            elif(self.rStratoCumuliformNull.isChecked()):
                obs.append(7)
            elif(self.rStratoCumuliformNo.isChecked()):
                obs.append(8)

            if(self.rCumuliformYes.isChecked()):
                obs.append(9)
            elif(self.rCumuliformNull.isChecked()):
                obs.append(10)
            elif(self.rCumuliformNo.isChecked()):
                obs.append(11)

            if(self.rCumulonibiformYes.isChecked()):
                obs.append(12)
            elif(self.rCumulonibiformNull.isChecked()):
                obs.append(13)
            elif(self.rCumulonibiformNo.isChecked()):
                obs.append(14)
            
            #  prev_probs=np.ones(5)
            # initialize Dir sample
            theta1=np.zeros((5,15))
            theta2=np.zeros((75,15))
            for X in range(5):
                #  theta1[X,:]=scipy.stats.dirichlet.mean(alpha=np.mean(self.table,axis=2)[X,:])
                theta1[X,:]=np.random.dirichlet(np.mean(self.table,axis=2)[X,:])
                for prev_obs in range(15):
                    #  theta2[X*15+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=self.table[X,prev_obs,:])
                    theta2[X*15+prev_obs,:]=np.random.dirichlet(self.table[X,prev_obs,:])
            tic=time.time()
            for k in range(250):
                # step 1
                for i in names:
                    likelihoods=[]
                    for prev_value in self.prev_obs:
                        for value in obs:
                            #  likelihoods.append(theta2[names.index(i)*15+prev_value,value]*theta1[names.index(i),prev_value])
                            self.probs[i]*=theta2[names.index(i)*15+prev_value,value]*theta1[names.index(i),prev_value]
                suma=sum(self.probs.values())
                for i in names:
                   self.probs[i]=np.log(self.probs[i])-np.log(suma) 
                   #  print self.probs[i]
                   self.probs[i]=np.exp(self.probs[i])
                # normalize
                suma=sum(self.probs.values())
                #  print self.probs.values()
                for i in names:
                    self.probs[i]/=suma
                #  print self.probs.values()
                # sample X
                X=np.random.choice(range(5),p=self.probs.values())
                # step 2
                for prev_value in self.prev_obs:
                    for value in obs:
                        self.table[X,prev_value,value]+=0.001
                    #  theta2[X*15+prev_value,:]=scipy.stats.dirichlet.mean(alpha=self.table[X,prev_value,:])
                    theta2[X*15+prev_value,:]=np.random.dirichlet(self.table[X,prev_value,:])
            self.prev_obs=obs
            print time.time()-tic

#              while spatial.distance.cosine(prev_probs,self.probs.values())!=0.0:

#              means=scipy.stats.dirichlet.mean(alpha=np.reshape(self.table,(1,self.table.shape[0]*self.table.shape[1]*self.table.shape[2]))[0])
#              new_means=np.reshape(np.array(means),(5,15,15))
#              for i in names:
#                  for prev_value in self.prev_obs:
#                      for value in obs:
#                          likelihood=(new_means[names.index(i),prev_value,value]/np.sum(new_means[names.index(i),prev_value,:]))*(np.sum(new_means[names.index(i),prev_value,:])/np.sum(new_means[names.index(i),:,:]))
#                          self.probs[i]*=likelihood
#              #normalize
#              suma=sum(self.probs.values())
#              for i in names:
#                  self.probs[i]/=suma


#              for prev_value in self.prev_obs:
#                  for value in obs:
#                      self.table[:,prev_value,value]+=0.1
#              self.prev_obs=obs

        self.matplotlibWidget4.axis4.clear()
        self.matplotlibWidget4.axis4.bar(range(5),self.probs.values())
        self.matplotlibWidget4.axis4.set_ylim(0,1)
        self.matplotlibWidget4.canvas4.draw()

# display probs
class MatplotlibWidget4(QtGui.QWidget):
    def __init__(self,parent=None):
        super(MatplotlibWidget4,self).__init__(parent)
        self.figure4=Figure()
        self.canvas4=FigureCanvasQTAgg(self.figure4)
        self.axis4=self.figure4.add_subplot(111)
        self.layoutVertical4=QtGui.QVBoxLayout(self)
        self.layoutVertical4.addWidget(self.canvas4)

# display intensity
class MatplotlibWidget3(QtGui.QWidget):
    def __init__(self,parent=None):
        super(MatplotlibWidget3,self).__init__(parent)
        self.figure3=Figure()
        self.canvas3=FigureCanvasQTAgg(self.figure3)
        self.axis3=self.figure3.add_subplot(111)
        self.layoutVertical3=QtGui.QVBoxLayout(self)
        self.layoutVertical3.addWidget(self.canvas3)

# display alphas
class MatplotlibWidget2(QtGui.QWidget):
    def __init__(self,parent=None):
        super(MatplotlibWidget2,self).__init__(parent)
        self.figure2=Figure()
        self.canvas2=FigureCanvasQTAgg(self.figure2)
        self.axis2=self.figure2.add_subplot(111,projection='3d')
        self.layoutVertical2=QtGui.QVBoxLayout(self)
        self.layoutVertical2.addWidget(self.canvas2)

# map data
class MatplotlibWidget(QtGui.QWidget):
    def __init__(self,parent=None):
        super(MatplotlibWidget,self).__init__(parent)
        self.figure=Figure()
        self.canvas=FigureCanvasQTAgg(self.figure)
        self.axis=self.figure.add_subplot(111)
        self.layoutVertical=QtGui.QVBoxLayout(self)
        self.layoutVertical.addWidget(self.canvas)

def make_some_data():
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

def get_intensity():
    genus=np.random.randint(5)
    print genus
    genus=4
    target=Cumuliform(genus=genus,weather=True)
    true_data=[]
    for frame in target.intensityModel:
        frame+=max(np.random.normal()*2,0)
        true_data.append(frame)
    return true_data

def DirPrior():
    table=np.zeros((5,15,15))
    #  base_table=np.array([[0.0817438692,0.1634877384,0.0136239782,0.0544959128,0.1634877384,0.0272479564,0.0272479564,0.0953678474,0.0544959128,0.0408719346,0.068119891,0.0544959128,0.0326975477,0.0544959128,0.068119891],
    #          [0.0476190476,0.1428571429,0.0238095238,0.0714285714,0.1428571429,0.0119047619,0.0357142857,0.0833333333,0.0714285714,0.0476190476,0.119047619,0.0238095238,0.0238095238,0.0476190476,0.1071428571],
    #          [0.047318612,0.1261829653,0.0630914826,0.0378548896,0.0630914826,0.094637224,0.1261829653,0.047318612,0.0157728707,0.0630914826,0.141955836,0.0315457413,0.0315457413,0.047318612,0.0630914826],
    #          [0.0338983051,0.0847457627,0.0508474576,0.0508474576,0.2033898305,0.0338983051,0.0169491525,0.0338983051,0.1355932203,0.0508474576,0.1186440678,0.0169491525,0.0338983051,0.1016949153,0.0338983051],
    #          [0.0282258065,0.0483870968,0.0967741935,0.0201612903,0.0483870968,0.1612903226,0.0403225806,0.060483871,0.060483871,0.0201612903,0.0403225806,0.1814516129,0.1008064516,0.0806451613,0.0120967742]])
    base_table=np.ones((5,15))
    base_table*=0.1
    for i in range(5):
        base_table[i,3*i]*=2
        for j in range(5):
            if i==j:
                base_table[i,3*j+2]*=0.5
            else:
                base_table[i,3*j+2]*=2
                base_table[i,3*j]*=0.5
    for i in range(15):
        table[:,:,i]=base_table
    for i in range(15):
        table[:,i,i]*=3
    #  print table
    return table


if __name__ == '__main__':
    #  total=make_some_data()
    total=[]
    prior=DirPrior()
    intensity=get_intensity()
    app=QtGui.QApplication(sys.argv)
    ex=Window(total,intensity,prior)
    sys.exit(app.exec_())
