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
np.set_printoptions(precision=2)

from data_sim.DynamicsProfiles import *
from HMM.hmm_classification import HMM_Classification
from gaussianMixtures import GM
from graphing import Graphing
from validation import *

warnings.filterwarnings("ignore",category=RuntimeWarning)
warnings.filterwarnings("ignore",category=UserWarning)

__author__ = "Jeremy Muesing"
__version__ = "2.3.0"
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
    # checking the config choices
    if not cfg['sim_types']['full_dir']:
        if graph_params['sim_results_full']:
            print "can't make the full sim results"
            raise
    if not cfg['sim_types']['param_tied_dir']:
        if graph_params['sim_results_tied']:
            print "can't make the tied sim results"
            raise
    if not cfg['sim_types']['param_tied_dir']:
        if graph_params['gibbs_val']:
            print "can't make gibbs graph"
            raise
    
    # initializing variables
    num_tar=cfg['num_tar']
    threshold=cfg['threshold']
    human_type=cfg['human']

    if graph_params['sim_results_full']:
        # target confusion matrix
        true_tar_full=[]
        pred_tar_full=[]
        pred_tar_full_ml=[]
        # precision recall
        pred_percent_full=[]
        correct_full=[0]*num_events
        # running average
        correct_percent_full=[]
        correct_ml_full=[0]*num_events
        correct_percent_ml_full=[]
    else:
        # target confusion matrix
        true_tar_full=None
        pred_tar_full=None
        pred_tar_full_ml=None
        # precision recall
        pred_percent_full=None
        correct_full=None
        # running average
        correct_percent_full=None
        correct_ml_full=None
        correct_percent_ml_full=None
    if graph_params['sim_results_tied']:
        # target confusion matrix
        true_tar_tied=[]
        pred_tar_tied=[]
        pred_tar_tied_ml=[]
        # precision recall
        pred_percent_tied=[]
        correct_tied=[0]*num_events
        # running average
        correct_percent_tied=[]
        correct_ml_tied=[0]*num_events
        correct_percent_ml_tied=[]
    else:
        # target confusion matrix
        true_tar_tied=None
        pred_tar_tied=None
        pred_tar_tied_ml=None
        # precision recall
        pred_percent_tied=None
        correct_tied=None
        # running average
        correct_percent_tied=None
        correct_ml_tied=None
        correct_percent_ml_tied=None
    if graph_params['sim_results_ind']:
        # target confusion matrix
        true_tar_ind=[]
        pred_tar_ind=[]
        pred_tar_ind_ml=[]
        # precision recall
        pred_percent_ind=[]
        correct_ind=[0]*num_events
        # running average
        correct_percent_ind=[]
        correct_ml_ind=[0]*num_events
        correct_percent_ml_ind=[]
    else:
        # target confusion matrix
        true_tar_ind=None
        pred_tar_ind=None
        pred_tar_ind_ml=None
        # precision recall
        pred_percent_ind=None
        correct_ind=None
        # running average
        correct_percent_ind=None
        correct_ml_ind=None
        correct_percent_ml_ind=None

    if (cfg['sim_types']['full_dir']) and (cfg['sim_types']['param_tied_dir']):
        # timing
        full_times=[]
        full_number=[]
        full_match_times=[]
        tied_times=[]
        tied_number=[]
        tied_match_times=[]
    else:
        full_times=None
        full_number=None
        full_match_times=None
        tied_times=None
        tied_number=None
        tied_match_times=None
    if cfg['sim_types']['ind_dir']:
        ind_times=[]
        ind_number=[]
        ind_match_times=[]
    else:
        ind_times=None
        ind_number=None
        ind_match_times=None


    #  # human validation
    #  all_theta2_tied=np.empty((num_events,4,4))
    #  all_theta2_full=np.empty((num_events,2*num_tar*num_tar,2*num_tar))
    if graph_params['gibbs_val']:
        # gibbs validations
        theta2_samples=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    else:
        theta2_samples=None

    # start sim
    if cfg['sim_types']['full_dir']:
        full_sim=DataFusion(num_tar)
        full_sim.DirPrior(num_tar,human_type)
    if cfg['sim_types']['param_tied_dir']:
        param_tied_sim=DataFusion(num_tar)
        param_tied_sim.DirPrior(num_tar,human_type)
    if cfg['sim_types']['ind_dir']:
        ind_sim=DataFusion(num_tar)
        ind_sim.DirPrior(num_tar,human_type)
    if graph_params['theta_val']:
        alphas_start=copy.deepcopy(param_tied_sim.theta2)
        alphas1_start=copy.deepcopy(param_tied_sim.theta1)
    else:
        alphas_start=None
        alphas1_start=None

    #running sim
    for n in tqdm(range(num_events),ncols=100):
        # initialize target type
        genus=np.random.randint(num_tar)
        #  genus=3
        # getting a prior from ML
        if cfg['starting_dist']=='assist':
            #  ml_threshold=0.25
            pass_off=0
            if cfg['sim_types']['param_tied_dir'] and cfg['sim_types']['full_dir'] and cfg['sim_types']['ind_dir']:
                param_tied_sim.make_data(genus)
                param_tied_sim.frame=0
                param_tied_sim.alphas={}
                param_tied_sim.probs={}
                for i in param_tied_sim.names:
                    param_tied_sim.alphas[i]=[-1,-1]
                    param_tied_sim.probs[i]=1/num_tar
                #  while max(param_tied_sim.probs.values())<ml_threshold:
                while (pass_off==0) and (param_tied_sim.frame<100):
                    param_tied_sim.updateProbsML()
                    param_tied_sim.frame+=1
                    pass_off=param_tied_sim.VOI2(num_tar,threshold,param_tied_sim.probs)
                full_sim.probs=param_tied_sim.probs
                ind_sim.probs=param_tied_sim.probs
                chosen=np.argmax(param_tied_sim.probs.values())
                if graph_params['sim_results_full']:
                    if genus==chosen:
                        correct_ml_full[n]=1
                    correct_percent_ml_full.append(sum(correct_ml_full)/(n+1))
                    pred_tar_full_ml.append(np.argmax(param_tied_sim.probs.values()))
                if graph_params['sim_results_tied']:
                    if genus==chosen:
                        correct_ml_tied[n]=1
                    correct_percent_ml_tied.append(sum(correct_ml_tied)/(n+1))
                    pred_tar_tied_ml.append(np.argmax(param_tied_sim.probs.values()))
                if graph_params['sim_results_ind']:
                    if genus==chosen:
                        correct_ml_ind[n]=1
                    correct_percent_ml_ind.append(sum(correct_ml_ind)/(n+1))
                    pred_tar_ind_ml.append(np.argmax(param_tied_sim.probs.values()))

            elif cfg['sim_types']['param_tied_dir'] and cfg['sim_types']['full_dir']:
                param_tied_sim.make_data(genus)
                param_tied_sim.frame=0
                param_tied_sim.alphas={}
                param_tied_sim.probs={}
                for i in param_tied_sim.names:
                    param_tied_sim.alphas[i]=[-1,-1]
                    param_tied_sim.probs[i]=1/num_tar
                #  while max(param_tied_sim.probs.values())<ml_threshold:
                while (pass_off==0) and (param_tied_sim.frame<100):
                    param_tied_sim.updateProbsML()
                    param_tied_sim.frame+=1
                    pass_off=param_tied_sim.VOI2(num_tar,threshold,param_tied_sim.probs)
                full_sim.probs=param_tied_sim.probs
                chosen=np.argmax(param_tied_sim.probs.values())
                if graph_params['sim_results_full']:
                    if genus==chosen:
                        correct_ml_full[n]=1
                    correct_percent_ml_full.append(sum(correct_ml_full)/(n+1))
                    pred_tar_full_ml.append(np.argmax(param_tied_sim.probs.values()))
                if graph_params['sim_results_tied']:
                    if genus==chosen:
                        correct_ml_tied[n]=1
                    correct_percent_ml_tied.append(sum(correct_ml_tied)/(n+1))
                    pred_tar_tied_ml.append(np.argmax(param_tied_sim.probs.values()))

            elif cfg['sim_types']['param_tied_dir'] and cfg['sim_types']['ind_dir']:
                param_tied_sim.make_data(genus)
                param_tied_sim.frame=0
                param_tied_sim.alphas={}
                param_tied_sim.probs={}
                for i in param_tied_sim.names:
                    param_tied_sim.alphas[i]=[-1,-1]
                    param_tied_sim.probs[i]=1/num_tar
                #  while max(param_tied_sim.probs.values())<ml_threshold:
                while (pass_off==0) and (param_tied_sim.frame<100):
                    param_tied_sim.updateProbsML()
                    param_tied_sim.frame+=1
                    pass_off=param_tied_sim.VOI2(num_tar,threshold,param_tied_sim.probs)
                ind_sim.probs=param_tied_sim.probs
                chosen=np.argmax(param_tied_sim.probs.values())
                if graph_params['sim_results_ind']:
                    if genus==chosen:
                        correct_ml_ind[n]=1
                    correct_percent_ml_ind.append(sum(correct_ml_ind)/(n+1))
                    pred_tar_ind_ml.append(np.argmax(param_tied_sim.probs.values()))
                if graph_params['sim_results_tied']:
                    if genus==chosen:
                        correct_ml_tied[n]=1
                    correct_percent_ml_tied.append(sum(correct_ml_tied)/(n+1))
                    pred_tar_tied_ml.append(np.argmax(param_tied_sim.probs.values()))

            elif cfg['sim_types']['full_dir']:
                full_sim.make_data(genus)
                full_sim.frame=0
                full_sim.alphas={}
                full_sim.probs={}
                for i in full_sim.names:
                    full_sim.alphas[i]=[-1,-1]
                    full_sim.probs[i]=1/num_tar
                #  while max(full_sim.probs.values())<ml_threshold:
                while (pass_off==0) and (full_sim.frame<100):
                    full_sim.updateProbsML()
                    full_sim.frame+=1
                    pass_off=full_sim.VOI2(num_tar,threshold,full_sim.probs)
                chosen=np.argmax(full_sim.probs.values())
                if graph_params['sim_results_full']:
                    if genus==chosen:
                        correct_ml_full[n]=1
                    correct_percent_ml_full.append(sum(correct_ml_full)/(n+1))
                    pred_tar_full_ml.append(np.argmax(full_sim.probs.values()))

            elif cfg['sim_types']['param_tied_dir']:
                param_tied_sim.make_data(genus)
                param_tied_sim.frame=0
                param_tied_sim.alphas={}
                param_tied_sim.probs={}
                for i in param_tied_sim.names:
                    param_tied_sim.alphas[i]=[-1,-1]
                    param_tied_sim.probs[i]=1/num_tar
                #  while max(param_tied_sim.probs.values())<ml_threshold:
                while (pass_off==0) and (param_tied_sim.frame<100):
                    param_tied_sim.updateProbsML()
                    param_tied_sim.frame+=1
                    pass_off=param_tied_sim.VOI2(num_tar,threshold,param_tied_sim.probs)
                chosen=np.argmax(param_tied_sim.probs.values())
                if graph_params['sim_results_tied']:
                    if genus==chosen:
                        correct_ml_tied[n]=1
                    correct_percent_ml_tied.append(sum(correct_ml_tied)/(n+1))
                    pred_tar_tied_ml.append(np.argmax(param_tied_sim.probs.values()))

        elif cfg['starting_dist']=='uniform':
            choose_from=[1]
            for i in range(num_tar-1):
                choose_from.append(0)
            # TODO: include independent observations
            if cfg['sim_types']['param_tied_dir'] and cfg['sim_types']['full_dir'] and cfg['sim_types']['ind_dir']:
                full_sim.probs={}
                param_tied_sim.probs={}
                ind_sim.probs={}
                for i in param_tied_sim.names:
                    full_sim.probs[i]=1/num_tar
                    ind_sim.probs[i]=1/num_tar
                    param_tied_sim.probs[i]=1/num_tar
                chosen=np.random.choice(choose_from)
                if graph_params['sim_results_full']:
                    correct_ml_full[n]=chosen
                    correct_percent_ml_full.append(sum(correct_ml_full)/(n+1))
                    pred_tar_full_ml.append(np.random.randint(num_tar))
                if graph_params['sim_results_tied']:
                    correct_ml_tied[n]=chosen
                    correct_percent_ml_tied.append(sum(correct_ml_tied)/(n+1))
                    pred_tar_tied_ml.append(np.random.randint(num_tar))
                if graph_params['sim_results_ind']:
                    correct_ml_ind[n]=chosen
                    correct_percent_ml_ind.append(sum(correct_ml_ind)/(n+1))
                    pred_tar_ind_ml.append(np.random.randint(num_tar))
            elif (cfg['sim_types']['full_dir']) and (cfg['sim_types']['param_tied_dir']):
                full_sim.probs={}
                param_tied_sim.probs={}
                for i in param_tied_sim.names:
                    full_sim.probs[i]=1/num_tar
                    param_tied_sim.probs[i]=1/num_tar
                chosen=np.random.choice(choose_from)
                if graph_params['sim_results_full']:
                    correct_ml_full[n]=chosen
                    correct_percent_ml_full.append(sum(correct_ml_full)/(n+1))
                    pred_tar_full_ml.append(np.random.randint(num_tar))
                if graph_params['sim_results_tied']:
                    correct_ml_tied[n]=chosen
                    correct_percent_ml_tied.append(sum(correct_ml_tied)/(n+1))
                    pred_tar_tied_ml.append(np.random.randint(num_tar))
            elif (cfg['sim_types']['ind_dir']) and (cfg['sim_types']['param_tied_dir']):
                ind_sim_probs.probs={}
                param_tied_sim.probs={}
                for i in param_tied_sim.names:
                    ind_sim.probs[i]=1/num_tar
                    param_tied_sim.probs[i]=1/num_tar
                chosen=np.random.choice(choose_from)
                if graph_params['sim_results_ind']:
                    correct_ml_ind[n]=chosen
                    correct_percent_ml_ind.append(sum(correct_ml_ind)/(n+1))
                    pred_tar_ind.append(np.random.randint(num_tar))
                if graph_params['sim_results_tied']:
                    correct_ml_tied[n]=chosen
                    correct_percent_ml_tied.append(sum(correct_ml_tied)/(n+1))
                    pred_tar_tied_ml.append(np.random.randint(num_tar))
            elif (cfg['sim_types']['full_dir']) and not (cfg['sim_types']['param_tied_dir']):
                full_sim.probs={}
                for i in full_sim.names:
                    full_sim.probs[i]=1/num_tar
                if graph_params['sim_results_full']:
                    correct_ml_full[n]=np.random.choice(choose_from)
                    correct_percent_ml_full.append(sum(correct_ml_full)/(n+1))
                    pred_tar_full_ml.append(np.random.randint(num_tar))
            elif not (cfg['sim_types']['full_dir']) and (cfg['sim_types']['param_tied_dir']):
                param_tied_sim.probs={}
                for i in param_tied_sim.names:
                    param_tied_sim.probs[i]=1/num_tar
                if graph_params['sim_results_tied']:
                    correct_ml_tied[n]=np.random.choice(choose_from)
                    correct_percent_ml_tied.append(sum(correct_ml_tied)/(n+1))
                    pred_tar_tied_ml.append(np.random.randint(num_tar))

        obs=[]
        start_tar=time.time()
        if cfg['sim_types']['param_tied_dir'] and cfg['sim_types']['full_dir'] and cfg['sim_types']['ind_dir']:
            full_sim_probs=full_sim.probs.values()
            ind_sim_probs=full_sim.probs.values()
            param_tied_sim_probs=param_tied_sim.probs.values()
            count_full=0
            count_tied=0
            count_ind=0
            while (max(full_sim_probs)<threshold) or (max(param_tied_sim_probs)<threshold) or (max(ind_sim_probs)<threshold):
                if time.time()-start_tar>20:
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
                param_tied_sim.moment_matching()
                param_tied_sim.moment_matching_small()
                tied_match_times.append(time.time()-start)
            if count_ind>0:
                start=time.time()
                ind_sim.moment_matching_ind(num_tar)
                ind_match_times.append(time.time()-start)

        elif cfg['sim_types']['param_tied_dir'] and cfg['sim_types']['full_dir']:
            full_sim_probs=full_sim.probs.values()
            param_tied_sim_probs=param_tied_sim.probs.values()
            count_full=0
            count_tied=0
            while (max(full_sim_probs)<threshold) or (max(param_tied_sim_probs)<threshold):
                if time.time()-start_tar>20:
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
            full_number.append(count_full)
            tied_number.append(count_tied)
            if count_full>1:
                start=time.time()
                full_sim.moment_matching_full(num_tar)
                full_sim.moment_matching_full_small(num_tar)
                full_match_times.append(time.time()-start)
            if count_tied>1:
                start=time.time()
                param_tied_sim.moment_matching()
                param_tied_sim.moment_matching_small()
                tied_match_times.append(time.time()-start)

        elif cfg['sim_types']['param_tied_dir'] and cfg['sim_types']['ind_dir']:
            ind_sim_probs=ind_sim.probs.values()
            param_tied_sim_probs=param_tied_sim.probs.values()
            count_ind=0
            count_tied=0
            while (max(ind_sim_probs)<threshold) or (max(param_tied_sim_probs)<threshold):
                if time.time()-start_tar>20:
                    break
                obs=param_tied_sim.HumanObservations(num_tar,genus,obs)
                if max(ind_sim_probs)<threshold:
                    start=time.time()
                    ind_sim_probs=ind_sim.sampling_ind(num_tar,obs)
                    ind_times.append(time.time()-start)
                    count_ind+=1
                if max(param_tied_sim_probs)<threshold:
                    start=time.time()
                    param_tied_sim_probs=param_tied_sim.sampling_param_tied(num_tar,obs)
                    tied_times.append(time.time()-start)
                    count_tied+=1
            ind_number.append(count_ind)
            tied_number.append(count_tied)
            if count_ind>0:
                start=time.time()
                ind_sim.moment_matching_ind(num_tar)
                ind_match_times.append(time.time()-start)
            if count_tied>1:
                start=time.time()
                param_tied_sim.moment_matching()
                param_tied_sim.moment_matching_small()
                tied_match_times.append(time.time()-start)

        elif cfg['sim_types']['full_dir']:
            full_sim_probs=full_sim.probs.values()
            count_full=0
            while (max(full_sim_probs)<threshold):
                if time.time()-start>20:
                    break
                obs=full_sim.HumanObservations(num_tar,genus,obs)
                full_sim_probs=full_sim.sampling_full(num_tar,obs)
                count_full+=1

            if count_full>1:
                full_sim.moment_matching_full(num_tar)
                full_sim.moment_matching_full_small(num_tar)

        elif cfg['sim_types']['param_tied_dir']:
            param_tied_sim_probs=param_tied_sim.probs.values()
            count_tied=0
            while (max(param_tied_sim_probs)<threshold):
                if time.time()-start>20:
                    break
                obs=param_tied_sim.HumanObservations(num_tar,genus,obs)
                param_tied_sim_probs=param_tied_sim.sampling_full(num_tar,obs)
                count_tied+=1

            if count_tied>1:
                param_tied_sim.moment_matching()
                param_tied_sim.moment_matching_small()

            if graph_params['gibbs_val']:
                # need a run where all 16 have been sampled, keep storing until a run produces that
                if count_tied>1:
                    for i in range(4):
                        for j in range(4):
                            samples=param_tied_sim.theta2_samples[np.nonzero(param_tied_sim.theta2_samples[:,i,j]),i,j]
                            if len(samples[0])>len(theta2_samples[i*4+j]):
                                theta2_samples[i*4+j]=samples[0]

        if graph_params['sim_results_full']:
            # building graphing parameters
            chosen=max(full_sim_probs)
            pred_percent_full.append(chosen)
            true_tar_full.append(genus)
            pred_tar_full.append(np.argmax(full_sim_probs))
            if genus==np.argmax(full_sim_probs):
                correct_full[n]=1
            correct_percent_full.append(sum(correct_full)/(n+1))
        if graph_params['sim_results_tied']:
            # building graphing parameters
            chosen=max(param_tied_sim_probs)
            pred_percent_tied.append(chosen)
            true_tar_tied.append(genus)
            pred_tar_tied.append(np.argmax(param_tied_sim_probs))
            if genus==np.argmax(param_tied_sim_probs):
                correct_tied[n]=1
            correct_percent_tied.append(sum(correct_tied)/(n+1))
        if graph_params['sim_results_ind']:
            # building graphing parameters
            chosen=max(ind_sim_probs)
            pred_percent_ind.append(chosen)
            true_tar_ind.append(genus)
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
    if graph_params['theta_val']:
        theta1=param_tied_sim.theta1
        theta1_correct=param_tied_sim.theta1_correct
        theta2=param_tied_sim.theta2
        theta2_correct=param_tied_sim.theta2_correct
    else:
        theta1=None
        theta1_correct=None
        theta2=None
        theta2_correct=None
    if graph_params['gibbs_val']:
        #  X_samples=param_tied_sim.X_samples
        X_samples=None
    else:
        X_samples=None
    if graph_params['sim_results_full']:
        real_obs=full_sim.real_obs
        pred_obs=full_sim.pred_obs
        pred_tar_ml=pred_tar_full_ml
    if graph_params['sim_results_tied']:
        real_obs=param_tied_sim.real_obs
        pred_obs=param_tied_sim.pred_obs
        pred_tar_ml=pred_tar_tied_ml
    if not (graph_params['sim_results_tied']) and not (graph_params['sim_results_full']):
        real_obs=None
        pred_obs=None
        pred_tar_ml=None


    print "Ind percent,num:",correct_percent_ind[-1],np.mean(ind_number)
    print "Tied percent,num:",correct_percent_tied[-1],np.mean(tied_number)
    print "Full percent,num:",correct_percent_full[-1],np.mean(full_number)
    graphs=Graphing(num_events,num_tar,alphas_start,theta2,true_tar_full,
            pred_tar_full,real_obs,pred_obs,pred_tar_ml,correct_percent_full,
            correct_percent_ml_full,correct_full,pred_percent_full,true_tar_tied,
            pred_tar_tied,correct_percent_tied,correct_percent_ml_tied,correct_tied,
            pred_percent_tied,theta2_correct,theta2_samples,X_samples,full_times,
            tied_times,full_number,tied_number,full_match_times,tied_match_times,
            alphas1_start,theta1,theta1_correct,correct_percent_ind,true_tar_ind,
            pred_tar_ind,ind_times,ind_number,ind_match_times)
    plt.show()
