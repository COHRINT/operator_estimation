from __future__ import division

import numpy as np
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
import cPickle as pickle
np.set_printoptions(precision=2)

from data_sim.DynamicsProfiles import *
from HMM.hmm_classification import HMM_Classification
from gaussianMixtures import GM
from graphing import Graphing
from validation import *

warnings.filterwarnings("ignore",category=RuntimeWarning)
warnings.filterwarnings("ignore",category=UserWarning)

__author__ = "Jeremy Muesing"
__version__ = "2.4.0"
__maintainer__ = "Jeremy Muesing"
__email__ = "jeremy.muesing@colorado.edu"
__status__ = "maintained"

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
    
    # initializing variables
    num_sims=cfg['num_sims']
    num_tar=cfg['num_tar']
    threshold=cfg['threshold']
    human_type=cfg['human']
    tar_seq=np.load(str(num_tar)+'_tar.npy')

    for sim_num in range(num_sims):
        #data
        data_dic={}
        for i in range(num_tar):
            data_dic[i]=[]
        # target confusion matrix
        pred_tar_full=[]
        # precision recall
        pred_percent_full=[]
        correct_full=[0]*num_events
        # running average
        correct_percent_full=[]
        # target confusion matrix
        true_tar_tied=[]
        pred_tar_tied=[]
        pred_tar_ml=[]
        pred_tar_ml_alone=[]
        # precision recall
        pred_percent_tied=[]
        correct_tied=[0]*num_events
        pred_percent_ml=[]
        pred_percent_ml_alone=[]
        correct_ml=[0]*num_events
        correct_ml_alone=[0]*num_events
        # running average
        correct_percent_tied=[]
        correct_ml=[0]*num_events
        correct_percent_ml=[]
        correct_ml_alone=[0]*num_events
        correct_percent_ml_alone=[]
        pass_off_tracker=[0]*num_events
        pass_off_average=[]
        # target confusion matrix
        pred_tar_ind=[]
        pred_tar_ind_ml=[]
        # precision recall
        pred_percent_ind=[]
        correct_ind=[0]*num_events
        # running average
        correct_percent_ind=[]

        # timing
        full_times=[]
        full_number=[]
        full_match_times=[]
        tied_times=[]
        tied_number=[]
        tied_match_times=[]
        ind_times=[]
        ind_number=[]
        ind_match_times=[]

        #  # human validation
        #  all_theta2_tied=np.empty((num_events,4,4))
        #  all_theta2_full=np.empty((num_events,2*num_tar*num_tar,2*num_tar))

        # gibbs validations
        theta2_samples=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]

        if num_tar==5:
            pass_off_targets=[[],[],[],[],[]]
        elif num_tar==10:
            pass_off_targets=[[],[],[],[],[],[],[],[],[],[]]

        # start sim
        full_sim=DataFusion(num_tar)
        full_sim.DirPrior(num_tar,human_type)
        param_tied_sim=DataFusion(num_tar)
        param_tied_sim.DirPrior(num_tar,human_type)
        ind_sim=DataFusion(num_tar)
        ind_sim.DirPrior(num_tar,human_type)
        alphas_start=copy.deepcopy(param_tied_sim.theta2)
        alphas1_start=copy.deepcopy(param_tied_sim.theta1)

        #running sim
        for n in tqdm(range(num_events),ncols=100):
            # initialize target type
            #  genus=np.random.randint(num_tar)
            genus=tar_seq[sim_num,n]
            #  genus=3
            # getting a prior from ML
            if cfg['starting_dist']=='assist':
                #  ml_threshold=0.25
                pass_off=0
                param_tied_sim.make_data(genus)
                data_dic[genus].append(param_tied_sim.intensity_data)
                param_tied_sim.frame=0
                param_tied_sim.alphas={}
                param_tied_sim.probs={}
                for i in param_tied_sim.names:
                    param_tied_sim.alphas[i]=[-1,-1]
                    param_tied_sim.probs[i]=1/num_tar
                #  while max(param_tied_sim.probs.values())<ml_threshold:
                while (pass_off==0) and (param_tied_sim.frame<100):
                    param_tied_sim.probs=param_tied_sim.updateProbsML()
                    param_tied_sim.frame+=1
                    pass_off=param_tied_sim.VOI2(num_tar,threshold,param_tied_sim.probs)
                #  print param_tied_sim.frame
                pass_off_tracker[n]=param_tied_sim.frame
                pass_off_average.append(np.mean(pass_off_tracker[:n+1]))
                pass_off_targets[genus].append(param_tied_sim.frame)
                full_sim.probs=copy.copy(param_tied_sim.probs)
                ind_sim.probs=copy.copy(param_tied_sim.probs)
                chosen=np.argmax(param_tied_sim.probs.values())
                if genus==chosen:
                    correct_ml[n]=1
                pred_percent_ml.append(max(param_tied_sim.probs.values()))
                correct_percent_ml.append(sum(correct_ml)/(n+1))
                pred_tar_ml.append(np.argmax(param_tied_sim.probs.values()))
                #continue even after passing off for comparison
                continue_probs=copy.copy(param_tied_sim.probs)
                while param_tied_sim.frame<100:
                    continue_probs=param_tied_sim.updateProbsML()
                    param_tied_sim.frame+=1
                if genus==np.argmax(continue_probs.values()):
                    correct_ml_alone[n]=1
                pred_percent_ml_alone.append(max(continue_probs.values()))
                correct_percent_ml_alone.append(sum(correct_ml_alone)/(n+1))
                pred_tar_ml_alone.append(np.argmax(continue_probs.values()))

            elif cfg['starting_dist']=='uniform':
                choose_from=[1]
                for i in range(num_tar-1):
                    choose_from.append(0)
                full_sim.probs={}
                param_tied_sim.probs={}
                ind_sim.probs={}
                for i in param_tied_sim.names:
                    full_sim.probs[i]=1/num_tar
                    ind_sim.probs[i]=1/num_tar
                    param_tied_sim.probs[i]=1/num_tar
                chosen=np.random.choice(choose_from)
                pred_percent_ml.append(1/num_tar)
                correct_ml[n]=chosen
                correct_percent_ml.append(sum(correct_ml)/(n+1))
                pred_tar_tied_ml.append(np.random.randint(num_tar))

                correct_ml_alone[n]=chosen
                pred_percent_ml_alone=copy.copy(pred_percent_ml)
                correct_percent_ml_alone=copy.copy(correct_percent_ml)
                pred_tar_ml_alone=copy.copy(pred_tar_ml)

            obs=[]
            start_tar=time.time()
            full_sim_probs=param_tied_sim.probs.values()
            ind_sim_probs=param_tied_sim.probs.values()
            param_tied_sim_probs=param_tied_sim.probs.values()
            count_full=0
            count_tied=0
            count_ind=0
            while (max(full_sim_probs)<threshold) or (max(param_tied_sim_probs)<threshold) or (max(ind_sim_probs)<threshold):
                if time.time()-start_tar>30:
                    break
                #  tar_to_ask=None
                #  if count_tied>0:
                #      tar_to_ask=param_tied_sim.VOI(num_tar,obs,threshold)
                #  if tar_to_ask is not None:
                #      obs=param_tied_sim.HumanAnswer(num_tar,tar_to_ask,genus,obs)
                #  else:
                obs=param_tied_sim.HumanObservations(num_tar,genus,obs)
                if max(full_sim_probs)<threshold:
                    start=time.time()
                    full_sim_probs=full_sim.sampling_full(num_tar,obs)
                    full_times.append(time.time()-start)
                    count_full+=1
                if max(param_tied_sim_probs)<threshold:
                    start=time.time()
                    param_tied_sim_probs=param_tied_sim.sampling_param_tied(num_tar,obs)
                    tied_times.append(time.time()-start)
                    count_tied+=1
                if max(ind_sim_probs)<threshold:
                    start=time.time()
                    ind_sim_probs=ind_sim.sampling_ind(num_tar,obs)
                    ind_times.append(time.time()-start)
                    count_ind+=1
            full_number.append(count_full)
            tied_number.append(count_tied)
            ind_number.append(count_ind)
            if count_full>1:
                start=time.time()
                full_sim.moment_matching_full(num_tar)
                full_sim.moment_matching_full_small(num_tar)
                full_match_times.append(time.time()-start)
            if count_tied>1:
                start=time.time()
                if (n==num_events-1) and (graph_params['moment_match']):
                    param_tied_sim.moment_matching(graph=True)
                else:
                    param_tied_sim.moment_matching()
                param_tied_sim.moment_matching_small()
                tied_match_times.append(time.time()-start)
            if count_ind>0:
                start=time.time()
                ind_sim.moment_matching_ind(num_tar)
                ind_match_times.append(time.time()-start)


            if graph_params['gibbs_val']:
                # need a run where all 16 have been sampled, keep storing until a run produces that
                if count_tied>1:
                    for i in range(4):
                        for j in range(4):
                            samples=param_tied_sim.theta2_samples[np.nonzero(param_tied_sim.theta2_samples[:,i,j]),i,j]
                            if len(samples[0])>len(theta2_samples[i*4+j]):
                                theta2_samples[i*4+j]=samples[0]

            # building graphing parameters
            chosen=max(full_sim_probs)
            pred_percent_full.append(chosen)
            pred_tar_full.append(np.argmax(full_sim_probs))
            if genus==np.argmax(full_sim_probs):
                correct_full[n]=1
            correct_percent_full.append(sum(correct_full)/(n+1))
            # building graphing parameters
            chosen=max(param_tied_sim_probs)
            pred_percent_tied.append(chosen)
            true_tar_tied.append(genus)
            pred_tar_tied.append(np.argmax(param_tied_sim_probs))
            if genus==np.argmax(param_tied_sim_probs):
                correct_tied[n]=1
            correct_percent_tied.append(sum(correct_tied)/(n+1))
            # building graphing parameters
            chosen=max(ind_sim_probs)
            pred_percent_ind.append(chosen)
            pred_tar_ind.append(np.argmax(ind_sim_probs))
            if genus==np.argmax(ind_sim_probs):
                correct_ind[n]=1
            correct_percent_ind.append(sum(correct_ind)/(n+1))
            #  all_theta2_tied[n,:,:]=param_tied_sim.theta2
            #  theta2_full_alphas=np.empty((2*num_tar*num_tar,2*num_tar))
            #  for X in range(num_tar):
            #      for prev_obs in range(2*num_tar):
            #          theta2_full_alphas[X*2*num_tar+prev_obs,:]=full_sim.theta2_full[X,prev_obs,:]
            #  all_theta2_full[n,:,:]=theta2_full_alphas
        graph_dic={'num_events':num_events,'num_tar':num_tar}
        if graph_params['percent_correct']:
            graph_dic['correct_percent_tied']=correct_percent_tied
            graph_dic['correct_percent_full']=correct_percent_full
            graph_dic['correct_percent_ind']=correct_percent_ind
            graph_dic['correct_percent_ml']=correct_percent_ml
            graph_dic['correct_percent_ml_alone']=correct_percent_ml_alone
        else:
            graph_dic['correct_percent_tied']=None
            graph_dic['correct_percent_full']=None
            graph_dic['correct_percent_ind']=None
            graph_dic['correct_precent_ml']=None

        if graph_params['precision_recall']:
            graph_dic['pred_percent_tied']=pred_percent_tied
            graph_dic['correct_tied']=correct_tied
            graph_dic['pred_percent_full']=pred_percent_full
            graph_dic['correct_full']=correct_full
            graph_dic['pred_percent_ind']=pred_percent_ind
            graph_dic['correct_ind']=correct_ind
            graph_dic['pred_percent_ml']=pred_percent_ml
            graph_dic['correct_ml']=correct_ml
            graph_dic['pred_percent_ml_alone']=pred_percent_ml_alone
            graph_dic['correct_ml_alone']=correct_ml_alone
        else:
            graph_dic['pred_percent_tied']=None
            graph_dic['correct_tied']=None
            graph_dic['pred_percent_full']=None
            graph_dic['correct_full']=None
            graph_dic['pred_percent_ind']=None
            graph_dic['correct_ind']=None

        if graph_params['confusion']:
            graph_dic['true_tar_tied']=true_tar_tied
            graph_dic['pred_tar_tied']=pred_tar_tied
            graph_dic['pred_tar_full']=pred_tar_full
            graph_dic['pred_tar_ind']=pred_tar_ind
            graph_dic['pred_tar_ml']=pred_tar_ml
            graph_dic['pred_tar_ml_alone']=pred_tar_ml_alone
            graph_dic['real_obs']=param_tied_sim.real_obs
            graph_dic['pred_obs']=param_tied_sim.pred_obs
        else:
            graph_dic['true_tar_tied']=None
            graph_dic['pred_tar_tied']=None
            graph_dic['pred_tar_full']=None
            graph_dic['pred_tar_ind']=None
            graph_dic['pred_tar_ml']=None
            graph_dic['pred_tar_ml_alone']=None
            graph_dic['real_obs']=None
            graph_dic['pred_obs']=None

        if graph_params['data']:
            graph_dic['data']=data_dic
        else:
            graph_dic['data']=None

        if graph_params['pass_off']:
            graph_dic['pass_off_average']=pass_off_average
            graph_dic['pass_off']=pass_off_tracker
        else:
            graph_dic['pass_off_average']=None
            graph_dic['pass_off']=None

        if graph_params['timing']:
            graph_dic['tied_times']=tied_times
            graph_dic['tied_number']=tied_number
            graph_dic['tied_match_times']=tied_match_times
            graph_dic['full_times']=full_times
            graph_dic['full_number']=full_number
            graph_dic['full_match_times']=full_match_times
            graph_dic['ind_times']=ind_times
            graph_dic['ind_number']=ind_number
            graph_dic['ind_match_times']=ind_match_times
        else:
            graph_dic['tied_times']=None
            graph_dic['tied_number']=None
            graph_dic['tied_match_times']=None
            graph_dic['full_times']=None
            graph_dic['full_number']=None
            graph_dic['full_match_times']=None
            graph_dic['ind_times']=None
            graph_dic['ind_number']=None
            graph_dic['ind_match_times']=None

        if graph_params['theta_val']:
            graph_dic['theta1']=param_tied_sim.theta1
            graph_dic['theta1_correct']=param_tied_sim.theta1_correct
            graph_dic['theta2']=param_tied_sim.theta2
            graph_dic['theta2_correct']=param_tied_sim.theta2_correct
            graph_dic['alphas_start']=alphas_start
            graph_dic['alphas1_start']=alphas1_start
        else:
            graph_dic['theta1']=None
            graph_dic['theta1_correct']=None
            graph_dic['theta2']=None
            graph_dic['theta2_correct']=None
            graph_dic['alphas_start']=None
            graph_dic['alphas1_start']=None

        if graph_params['gibbs_val']:
            graph_dic['X_samples']=param_tied_sim.X_samples
            graph_dic['theta2_samples']=theta2_samples
        else:
            graph_dic['X_samples']=None
            graph_dic['theta2_samples']=None

        for i in range(num_tar):
            graph_dic['avg_pass_off_'+str(i)]=pass_off_targets[i]

        graph_dic['human_correct_overall']=param_tied_sim.human_correct

        filename='figures/graphing_data/graphing_data'+str(sim_num)+'.p'
        pickle.dump(graph_dic,open(filename,'wb'))


        #  print "Ind percent,num:",correct_percent_ind[-1],np.mean(ind_number)
        #  print "Tied percent,num:",correct_percent_tied[-1],np.mean(tied_number)
        #  print "Full percent,num:",correct_percent_full[-1],np.mean(full_number)
        #  print "Human Percent of correct observations:",sum(param_tied_sim.human_correct)/len(param_tied_sim.human_correct)
        #  for i in range(num_tar):
        #      print "Avg pass off frame target ",i,":",np.mean(pass_off_targets[i])
        #  Graphing(graph_dic)
        #  plt.show()
