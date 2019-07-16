from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy 
import time
import random


class AbstractProfile():
	def __init__(self,startPose=[0,0],genus = 0,weather = True, startTime=0):
		self.time = startTime; 
		state = startPose; 
		self.weather = weather;
                self.genus = genus;

                if -100<=startPose[0]<=50:
                    velX=random.random()*2
                elif 50<startPose[0]<=200:
                    velX=random.random()*-2
                if -100<=startPose[1]<=50:
                    velY=random.random()*2
                elif 50<startPose[1]<=200:
                    velY=random.random()*-2

		#  self.initVel = [random.random()*6-3,random.random()*6-3];
		self.initVel = [velX,velY]

		state.append(self.initVel[0]); 
		state.append(self.initVel[1]); 
		self.state = np.matrix(state).T; 
		self.period = 100; 

		self.intensityModel = self.buildIntensity(); 
		self.dynamicsModel = self.buildDynamics(); 

		

	def getShakes(self,BlairWitchFactor = 0.4):
		#Set up camera shaking
		#BlairWitchFactor = 0.4
		shakes = [0, 0]
		shakes[0] = np.random.randint(-BlairWitchFactor, BlairWitchFactor+1)
		shakes[1] = np.random.randint(-BlairWitchFactor, BlairWitchFactor+1)
		return shakes; 

	def update(self,shake=False):
		
		self.state = self.dynamicsModel[self.time]; 
		intense = self.intensityModel[self.time%self.period]; 
		self.time +=1; 
		
		if(shake):
			shakes = self.getShakes(); 
			return self.state,intense,self.genus,shakes; 
		else:
			return self.state,intense,self.genus


class Cirrus(AbstractProfile):
	 
	def buildIntensity(self):
		intense = np.zeros(shape=(100)); 
		for i in range(0,50):
			intense[i] = (i/20)*(5+2*np.sin(1*i))*10/16;
		for i in range(50,100):
			intense[i] = ((100-i)/20)*(5+2*np.sin(1*i))*10/16; 

		return intense;   

	def buildDynamics(self):

		dmod = []; 
		curState = self.state.T.tolist()[0]; 
		
		for i in range(0,500):
			xdot = 5*np.sin(.1*i)/4+ self.initVel[0]; 
			ydot = 5*np.cos(.1*i)/4+ self.initVel[1]; 
			x = curState[0]+xdot; 
			y = curState[1]+ydot; 
			curState = [x,y,xdot,ydot]; 
			dmod.append(curState); 

		return dmod; 


class Stratus(AbstractProfile):

	def buildIntensity(self):
		intense = np.zeros(shape=(100)); 
		for i in range(0,50):
			intense[i] = ((50-i)/20)*(5+2*np.sin(.5*i))*10/16;
		for i in range(50,100):
			intense[i] = ((i-50)/20)*(5+2*np.sin(.5*i))*10/16; 

		return intense;   

	def buildDynamics(self):
		dmod = []; 
		curState = self.state.T.tolist()[0]; 
		for k in range(0,5):
			for i in range(k*100,k*100+50):
				xdot = 5*np.sin(.1*i)/4+ self.initVel[0]; 
				ydot = (i%100)/5/4+ self.initVel[1]
				x = curState[0]+xdot; 
				y = curState[1]+ydot; 
				curState = [x,y,xdot,ydot]; 
				dmod.append(curState); 
			for i in range(k*100+50,k*100+100):
				xdot = 5*np.sin(.1*i)/4 + self.initVel[0]; 
				ydot = (100-(i%100))/5/4 + self.initVel[1]
				x = curState[0]+xdot; 
				y = curState[1]+ydot; 
				curState = [x,y,xdot,ydot]; 
				dmod.append(curState); 


		return dmod; 

class Cumulus(AbstractProfile):

	def buildIntensity(self):
		intense = np.zeros(shape=(100)); 
		for i in range(0,10):
			for j in range(0,10):
				if(i%2 == 1 and j < 6):
					intense[i*10+j] = 5; 
				else:
					intense[i*10+j] = 1; 

		return intense;   

	def buildDynamics(self):
		dmod = []; 
		curState = self.state.T.tolist()[0]; 
		
		for i in range(0,500):
			if((i//10)%2 == 1 and i%10 < 5):
				ydot = 4/4 + self.initVel[1]
				xdot = 5.8/4 + self.initVel[0]
			else:
				ydot = 2/4+ self.initVel[1]; 
				xdot = 1.4/4+ self.initVel[0]; 
			x = curState[0]+xdot; 
			y = curState[1]+ydot; 
			curState = [x,y,xdot,ydot]; 
			dmod.append(curState); 
		

		return dmod; 


class Nimbostratus(AbstractProfile):

	def buildIntensity(self):
		intense = np.zeros(shape=(100)); 
		for i in range(0,10):
			for j in range(0,10):
				if(i%2 == 1):
					intense[i*10+j] = 5 + 5*np.sin(j); 
				else:
					intense[i*10+j] = 6; 

		return intense;

	def buildDynamics(self):
		dmod = []; 
		curState = self.state.T.tolist()[0]; 
		
		for k in range(0,5):
			for i in range(k*100,k*100+50):
				ydot = 5*np.sin(.1*i)/4+ self.initVel[1]; 
				xdot = (i%100)/5/4+ self.initVel[0]
				x = curState[0]+xdot; 
				y = curState[1]+ydot; 
				curState = [x,y,xdot,ydot]; 
				dmod.append(curState); 
			for i in range(k*100+50,k*100+100):
				ydot = 5*np.sin(.1*i)/4+ self.initVel[1]; 
				xdot = (100-(i%100))/5/4+ self.initVel[0]
				x = curState[0]+xdot; 
				y = curState[1]+ydot; 
				curState = [x,y,xdot,ydot]; 
				dmod.append(curState); 


		return dmod; 


class Cumulonimbus(AbstractProfile):

	def buildIntensity(self):
		intense = np.zeros(shape=(100)); 
		for i in range(0,30):
			intense[i] = (i/3)*10/11; 
		for i in range(30,70):
			intense[i] = (10+np.sin(.5*i))*10/11; 
		for i in range(70,100):
			intense[i] = ((100-i)/3)*10/11; 

		return intense;   

	def buildDynamics(self):
		dmod = []; 
		curState = self.state.T.tolist()[0]; 
		
		for k in range(0,5):
			for i in range(k*100,k*100+50):
				xdot = (i%100)/10/4+ self.initVel[0]
				ydot = (50-(i%100))/10/4+ self.initVel[1]
				x = curState[0]+xdot; 
				y = curState[1]+ydot; 
				curState = [x,y,xdot,ydot]; 
				dmod.append(curState); 
			for i in range(k*100+50,k*100+100):
				xdot = 5/4 + self.initVel[0]
				ydot = 0/4 + self.initVel[1]
				x = curState[0]+xdot; 
				y = curState[1]+ydot; 
				curState = [x,y,xdot,ydot]; 
				dmod.append(curState); 


		return dmod; 

class Altostratus(AbstractProfile):

	def buildIntensity(self):
		intense = np.zeros(shape=(100)); 
		for i in range(0,100):
			intense[i] = (30 + np.sin(.5*i))*10/31; 
		return intense; 

	def buildDynamics(self):
		dmod = []; 
		curState = self.state.T.tolist()[0];

		for k in range(0,5):
			for i in range(k*100,k*100+100):
				xdot = 0+self.initVel[0]; 
				ydot = 0+self.initVel[1]; 
				x = curState[0]+xdot; 
				y = curState[1]+ydot; 
				curState = [x,y,xdot,ydot]; 
				dmod.append(curState); 
		return dmod; 


class FamilyContainer(AbstractProfile):

	def buildDynamics(self):
		dmod = []; 
		curState = self.state.T.tolist()[0];
		
		for i in range(0,100):
			xdot = 0+self.initVel[0]; 
			ydot = 0+self.initVel[1]; 
			x = curState[0]+xdot; 
			y = curState[1]+ydot; 
			curState = [x,y,xdot,ydot]; 
			dmod.append(curState); 
		return dmod; 


class Stratiform(FamilyContainer):
	#Three stage cloud, decreasing intensity
	#with take off

	def buildIntensity(self):
		intense = np.zeros(shape=(100)); 
		
		if(self.genus==0):
			stageSets = [0,5,40,70,100]; 
			vals = [10,9.5,7,4];
		elif(self.genus==1):
			stageSets = [0,10,40,70,100]; 
			vals = [9.8,9.5,7,4];
		elif(self.genus==2):
			stageSets = [0,5,50,70,100]; 
			vals = [10,9,7,4];
		elif(self.genus==3):
			stageSets = [0,5,40,80,100]; 
			vals = [10,9.5,6.6,4];
		elif(self.genus==4):
			stageSets = [0,5,35,65,100]; 
			vals = [10,9.5,7,4.5];

		for i in range(1,len(stageSets)):
			for j in range(stageSets[i-1],stageSets[i]):
				intense[j] = vals[i-1]; 
		
		if(self.weather):
			attenuation_from_weather = attenuation_rocket(len(intense))
			for k in range(len(intense)):
				intense[k]=intense[k]*attenuation_from_weather[k]

		return intense; 


class Cirriform(FamilyContainer):
	#Two stage cloud, decreasing intensity
	#with takeoff

	def buildIntensity(self):
		intense = np.zeros(shape=(100)); 
		
		if(self.genus==0):
			stageSets = [0,5,60,100]; 
			vals = [9.5,8,5]; 
		elif(self.genus == 1):
			stageSets = [0,10,60,100]; 
			vals = [9.1,8,5]; 
		elif(self.genus == 2):
			stageSets = [0,5,50,100]; 
			vals = [9.5,9.1,5]; 
		elif(self.genus == 3):
			stageSets = [0,5,70,100]; 
			vals = [9.5,8,5.4]; 
		elif(self.genus == 4):
			stageSets = [0,15,40,100]; 
			vals = [8.8,8,5]; 	

		for i in range(1,len(stageSets)):
			for j in range(stageSets[i-1],stageSets[i]):
				intense[j] = vals[i-1]; 
		
		if(self.weather):
			attenuation_from_weather = attenuation_rocket(len(intense))
			for k in range(len(intense)):
				intense[k]=intense[k]*attenuation_from_weather[k]

		return intense; 


class Stratocumuliform(FamilyContainer):
	#One stage cloud, mostly constant intensity
	#with takeoff

	def buildIntensity(self):
		intense = np.zeros(shape=(100)); 

		if(self.genus==0):
			stageSets = [0,10,100]; 
			vals = [9.8,7]; 
		elif(self.genus == 1):
			stageSets = [0,15,100]; 
			vals = [9.7,7]; 
		elif(self.genus == 2):
			stageSets = [0,5,100]; 
			vals = [10,7.5]; 
		elif(self.genus == 3):
			stageSets = [0,20,100]; 
			vals = [9.6,6.9]; 
		elif(self.genus == 4):
			stageSets = [0,5,100]; 
			vals = [10,6.8]; 

		for i in range(1,len(stageSets)):
			for j in range(stageSets[i-1],stageSets[i]):
				intense[j] = vals[i-1]; 
		
		if(self.weather):
			attenuation_from_weather = attenuation_rocket(len(intense))
			for k in range(len(intense)):
				intense[k]=intense[k]*attenuation_from_weather[k]

		return intense; 


class Cumuliform(FamilyContainer):
	#Cloud that drops and reignites
	#with takeoff

	def buildIntensity(self):
		intense = np.zeros(shape=(100)); 
		

                if(self.genus==0):
                        stageSets = [0,15,50,70,100];
                        vals = [4,6,1,8];
                elif(self.genus == 1):
                        stageSets = [0,10,50,65,100];
                        vals = [8.5,5,2.5,5];
                elif(self.genus == 2):
                        stageSets = [0,5,50,60,100];
                        vals = [6,3,4.5,6];
                elif(self.genus == 3):
                        stageSets = [0,20,40,75,100];
                        vals = [7,9,3,5.5];
                elif(self.genus == 4):
                        stageSets = [0,5,45,60,100];
                        vals = [5,4.5,0.5,3];
                elif(self.genus==5):
                        stageSets = [0,5,35,70,100];
                        vals = [3,7,5.5,11];
                elif(self.genus == 6):
                        stageSets = [0,10,60,80,100];
                        vals = [8,1,9,7];
                elif(self.genus == 7):
                        stageSets = [0,5,50,90,100];
                        vals = [3.5,7,1.5,4.5];
                elif(self.genus == 8):
                        stageSets = [0,8,32,45,100];
                        vals = [5,10,2,7.5];
                elif(self.genus == 9):
                        stageSets = [0,5,70,80,100];
                        vals = [6,9,0.5,5.5];

                #  if(self.genus==0):
                #          stageSets = [0,15,50,70,100];
                #          vals = [40,60,10,80];
                #  elif(self.genus == 1):
                #          stageSets = [0,10,50,65,100];
                #          vals = [85,50,25,50];
                #  elif(self.genus == 2):
                #          stageSets = [0,5,50,60,100];
                #          vals = [60,30,45,60];
                #  elif(self.genus == 3):
                #          stageSets = [0,20,40,75,100];
                #          vals = [70,90,30,55];
                #  elif(self.genus == 4):
                #          stageSets = [0,5,45,60,100];
                #          vals = [50,45,5,30];
                #  elif(self.genus==5):
                #          stageSets = [0,5,35,70,100];
                #          vals = [30,70,55,110];
                #  elif(self.genus == 6):
                #          stageSets = [0,10,60,80,100];
                #          vals = [80,10,90,70];
                #  elif(self.genus == 7):
                #          stageSets = [0,5,50,90,100];
                #          vals = [35,70,15,45];
                #  elif(self.genus == 8):
                #          stageSets = [0,8,32,45,100];
                #          vals = [50,100,20,75];
                #  elif(self.genus == 9):
                #          stageSets = [0,5,70,80,100];
                #          vals = [60,90,5,55];


		for i in range(1,len(stageSets)):
			for j in range(stageSets[i-1],stageSets[i]):
				intense[j] = vals[i-1]; 
		
		if(self.weather):
			attenuation_from_weather = attenuation_rocket(len(intense))
			for k in range(len(intense)):
				intense[k]=intense[k]*attenuation_from_weather[k]

		return intense; 

class Cumulonibiform(FamilyContainer):
	#Cloud with increasing intensity that explodes at the end
	#without takeoff

	def buildIntensity(self):
		intense = np.zeros(shape=(100)); 
		
		stageSets = [i*5 for i in range(0,21)]; 
		vals = [i/5 for i in range(0,20)]; 
		vals[-1] = 10; 

		if(self.genus==0):
			stageSets = [i*5 for i in range(0,21)]; 
			vals = [i/5 for i in range(0,20)]; 
			vals[-1] = 10; 
		elif(self.genus == 1):
			stageSets = [i*10 for i in range(0,11)]; 
			vals = [i/10 for i in range(0,10)]; 
			vals[-1] = 10; 
		elif(self.genus == 2):
			stageSets = [i*5 for i in range(0,21)]; 
			vals = [i/5 + 0.5 for i in range(0,20)]; 
			vals[-1] = 10;
		elif(self.genus == 3):
			stageSets = [i*2 for i in range(0,51)]; 
			vals = [min(i/2,9) for i in range(0,50)]; 
			vals[-1] = 10;
		elif(self.genus == 4):
			stageSets = [i*4 for i in range(0,26)]; 
			vals = [i/4 for i in range(0,25)]; 
			vals[-1] = 10;

		for i in range(1,len(stageSets)):
			for j in range(stageSets[i-1],stageSets[i]):
				intense[j] = vals[i-1]; 
		
		if(self.weather):
			attenuation_from_weather = attenuation_rocket(len(intense))
			for k in range(len(intense)):
				intense[k]=intense[k]*attenuation_from_weather[k]

		return intense; 

def attenuation_rocket(length):
	# all attenuation data from clouds found here: https://www.rand.org/content/dam/rand/pubs/reports/2006/R1694.pdf
	
	def intensity_decrease(db_expo):
		return 10**(-(10**(db_expo))/10)


	rocket_type = np.random.randint(0,5)
	if rocket_type==0:
		db_expo = 2.5 # AtlasV
		height = np.random.normal(1.5,.5) #km
		length_of_attunation = int(np.random.normal(40,15))
	if rocket_type==1:
		db_expo = 2.3 # DeltaIV
		height = np.random.normal(.5,.1) #km
		length_of_attunation = int(np.random.normal(60,20))
	if rocket_type==2:
		db_expo = 2.2 # Falcon9
		height = np.random.normal(1,.25) #km
		length_of_attunation = int(np.random.normal(20,10))
	if rocket_type==3:
		db_expo = 2.1 # SLS
		height = np.random.normal(3,1) #km
		length_of_attunation = int(np.random.normal(35,15))
	if rocket_type==4:
		db_expo = 2 # Vulcan
		height = np.random.normal(0.75,.5) #km
		length_of_attunation = int(np.random.normal(10,3))

	attenuation_array=np.ones(length)
	attenuation=intensity_decrease(db_expo)*height
	starting_point = np.random.randint(0,length)
	if length_of_attunation+starting_point>length:
		attenuation_array[starting_point:]=attenuation
	else:
		attenuation_array[starting_point:starting_point+length_of_attunation]=attenuation
	return attenuation_array
	

def testProfile(prof,duration = 100,animate = False):
	allInts = []; 	
	allPose = prof.state.tolist(); 

	for i in range(0,duration):
		[s,I] = prof.update(); 
		allInts.append(I); 
		for j in range(0,len(s)):
			allPose[j].append(s[j]); 

	for i in range(0,len(allPose)):
		allPose[i].remove(allPose[i][0]); 
	allInts.remove(allInts[0]); 

	allVel = []; 
	allAngle = []; 
	for i in range(0,len(allPose[2])):
		allVel.append(np.sqrt(allPose[2][i]**2 + allPose[3][i]**2)); 
		allAngle.append(np.arctan2(allPose[3][i],allPose[2][i]) + np.pi); 

	#plt.xkcd(); 

	fig,axarr = plt.subplots(2,2); 
	if(not animate):
		axarr[0][0].plot(allInts,c='b'); 
		axarr[0][0].set_title('Intensity')
		axarr[0][0].set_ylim([0,10.5]); 

		axarr[0][1].plot(allPose[0],allPose[1],c='r'); 
		axarr[0][1].set_title('Position'); 

		axarr[1][0].plot(allVel,c='k'); 
		axarr[1][0].set_title('|V|')

		axarr[1][1].plot(allAngle,c='m'); 
		axarr[1][1].set_title('Angle'); 

		plt.suptitle("Profile of: " + prof.__class__.__name__); 

	else:
		for i in range(0,len(allPose[0])-1):

			axarr[0][0].scatter(i,allInts[i],c='b'); 
			axarr[0][0].set_title('Intensity')
			axarr[0][0].set_ylim([min(allInts)-5,max(allInts)+5]); 


			colG = (255 - 255*i/len(allPose[0]))/255;
			colR = (255*i/len(allPose[0]))/255; 

			axarr[0][1].scatter(allPose[0][i],allPose[1][i],c=(colR,colG,0)); 
			axarr[0][1].set_title('Position'); 
			axarr[0][1].set_xlim([min(allPose[0])-5,max(allPose[0])+5]); 
			axarr[0][1].set_ylim([min(allPose[1])-5,max(allPose[1])+5]); 

			axarr[1][0].scatter(i,allVel[i],c='k'); 
			axarr[1][0].set_ylim([min(allVel) - 3,max(allVel) + 3]); 
			axarr[1][0].set_title('|V|')

			axarr[1][1].scatter(i,allAngle[i],c='m'); 
			axarr[1][1].set_ylim([-np.pi/2,np.pi*2 + np.pi/2]); 
			axarr[1][1].set_title('Angle');

			plt.suptitle("Profile of: " + prof.__class__.__name__); 

			plt.pause(0.001); 
	plt.show(); 

def testFamily(profName,makeFigs = False):

	fig = plt.figure(); 

	ints = []; 
	x = [i for i in range(0,100)]; 
	for i in range(0,5):
		c = profName(genus = i); 
		ints.append(c.buildIntensity()+np.random.normal(0,0.1,100)); 
		plt.plot(x,ints[i],label=str(i),linewidth = 3); 

	plt.title('Family: {}'.format(profName.__name__),fontsize=25)
	plt.ylim([0,10]);
	plt.xlabel("Time (frames)",fontsize=15); 
	plt.ylabel("Intensity (Units)",fontsize=15); 
	plt.legend()

	if(makeFigs):
		plt.savefig('../imgs/Family_{}.png'.format(profName.__name__)); 
	else:
		plt.show(); 


if __name__ == '__main__':
   	#plt.xkcd();
   	# allProf = []; 

   	# allProf.append(Cirrus(startPose=[40,40])); 
   	# allProf.append(Stratus(startPose=[40,40]));
   	# allProf.append(Cumulus(startPose=[40,40]));
   	# allProf.append(Nimbostratus(startPose=[40,40]));
   	# allProf.append(Cumulonimbus(startPose=[40,40]));
   	# allProf.append(Altostratus(startPose=[40,40])); 

   	#testProfile(Cumulonibiform(genus = 0),duration = 100,animate = False)
   	figs = False; 
   	# testFamily(Stratiform,figs);
   	# testFamily(Cirriform,figs); 
   	# testFamily(Stratocumuliform,figs); 
   	testFamily(Cumuliform,figs); 
   	# testFamily(Cumulonibiform,figs); 



 #   	allInts = []; 
 #   	for i in range(0,len(allProf)):
 #  		allInts.append(allProf[i].intensityModel); 
 
 #  	allCor = np.zeros(shape=(len(allInts),len(allInts))); 
 #   	for i in range(0,len(allInts)):
 #   		for j in range(0,len(allInts)):
 #   			#allCor[i][j] = np.correlate(allInts[i],allInts[j],normed=True); 
 #   			[lags,c,line,b] = plt.xcorr(allInts[i],allInts[j],normed=True); 
 #   			allCor[i][j] = max(abs(c));
 #   	plt.close(); 

 #   	#row_sums = allCor.sum(axis=1);
	# #row_sums[row_sums==0] = 0.0000001; 
	# #allCor = allCor/row_sums[:,np.newaxis]; 

	# #plt.figure(); 
	# #print(allCor); 

 #   	plt.matshow(allCor); 
 #   	plt.colorbar(); 
 #   	plt.title('Maximum Cross Correlation')
 #   	plt.show(); 
