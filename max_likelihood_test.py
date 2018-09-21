from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats

human_observation_model = {}
human_observation_model['Cumuliform0'] = [0.0817438692,0.1634877384,0.0136239782,0.0544959128,0.1634877384,0.0272479564,0.0272479564,0.0953678474,0.0544959128,0.0408719346,0.068119891,0.0544959128,0.0326975477,0.0544959128,0.068119891]
human_observation_model['Cumuliform1'] = [0.0476190476,0.1428571429,0.0238095238,0.0714285714,0.1428571429,0.0119047619,0.0357142857,0.0833333333,0.0714285714,0.0476190476,0.119047619,0.0238095238,0.0238095238,0.0476190476,0.1071428571]
human_observation_model['Cumuliform2'] = [0.047318612,0.1261829653,0.0630914826,0.0378548896,0.0630914826,0.094637224,0.1261829653,0.047318612,0.0157728707,0.0630914826,0.141955836,0.0315457413,0.0315457413,0.047318612,0.0630914826]
human_observation_model['Cumuliform3'] = [0.0338983051,0.0847457627,0.0508474576,0.0508474576,0.2033898305,0.0338983051,0.0169491525,0.0338983051,0.1355932203,0.0508474576,0.1186440678,0.0169491525,0.0338983051,0.1016949153,0.0338983051]
human_observation_model['Cumuliform4'] = [0.0282258065,0.0483870968,0.0967741935,0.0201612903,0.0483870968,0.1612903226,0.0403225806,0.060483871,0.060483871,0.0201612903,0.0403225806,0.1814516129,0.1008064516,0.0806451613,0.0120967742]

truth_array=np.column_stack((human_observation_model['Cumuliform0'],human_observation_model['Cumuliform1'],
    human_observation_model['Cumuliform2'],human_observation_model['Cumuliform3'],human_observation_model['Cumuliform4']))

def max_likelihood_total_samples():
    error=[]
    variance=[]
    k=range(1,1000,10)
    for h in k:
        total_array=np.zeros([15,5])
        for i in range(h):
            j=0
            for key in human_observation_model.keys():
                probs=human_observation_model[key]
                sample=np.random.choice(15,p=probs)
                total_array[sample,j]=total_array[sample,j]+1
                j=j+1
        total_array=total_array/(i+1)
        #  error_total=100*abs(truth_array-total_array)/truth_array
        error_total=truth_array-total_array
        error_row=np.mean(error_total,axis=1)
        #  error.append(np.mean(error_total))
        error.append(error_row.tolist())
        variance_row=[2*math.sqrt(i) for i in abs(error_row)]
        #  variance.append(2*math.sqrt(np.mean(error_total)))
        variance.append(variance_row)
        #  variance.append(math.pow(np.mean(error_total),2)/(i+1))
    error_rows_plot=np.zeros((len(error),15))
    variance_rows_plot=np.zeros((len(error),15))
    for m in range(len(error)):
        for n in range(15):
            error_rows_plot[m,n]=error[m][n]
            variance_rows_plot[m,n]=variance[m][n]
    for n in range(15):
        plt.plot(k,error_rows_plot[:,n])
        plt.fill_between(k,error_rows_plot[:,n]+variance_rows_plot[:,n],error_rows_plot[:,n]-variance_rows_plot[:,n],alpha=0.5)
        #  plt.fill_between(k,error_rows_plot[:,n]+variance_rows_plot[:,n],alpha=0.5)

    plt.xlabel('Total number of samples')
    plt.ylabel('Absolute Error')
    plt.title('Error of Max Likelihood over total # of samples')
    plt.show()

def max_likelihood_ind_sample():
    probs=human_observation_model['Cumuliform0']
    error_list=[]
    variance=[]
    obs=[]
    total_array=np.zeros([15,1])
    for i in range(500):
        sample=np.random.choice(15,p=probs)
        total_array[sample]=total_array[sample]+1
        mus=total_array/(i+1)
        error=abs(mus-np.array([truth_array[:,0]]).transpose())
        error_list.append(error.tolist())
        variance.append([2*math.sqrt(i) for i in abs(error)])
        obs.append(total_array.tolist())
    #  print error_list
    #  print variance
    #  print obs
    error_rows_plot=np.zeros((len(error_list),15))
    variance_rows_plot=np.zeros((len(error_list),15))
    obs_rows_plot=np.zeros((len(error_list),15))
    for m in range(len(error_list)):
        for n in range(15):
            #  print error_list[m][n]
            error_rows_plot[m,n]=error_list[m][n][0]
            variance_rows_plot[m,n]=variance[m][n]
            obs_rows_plot[m,n]=obs[m][n][0]
    #  print error_rows_plot
    #  print variance_rows_plot
    #  print obs_rows_plot

    for n in range(15):
        plt.plot(obs_rows_plot[:,n],error_rows_plot[:,n])
        #  plt.fill_between(obs_rows_plot[:,n],error_rows_plot[:,n]+variance_rows_plot[:,n],error_rows_plot[:,n]-variance_rows_plot[:,n],alpha=0.5)
    plt.xlabel('Number of samples for observatoin type')
    plt.ylabel('Absolute Error')
    plt.title('Error of Max Likelihood over # of samples')

    plt.show()

def dirichlet(prior):
    probs=human_observation_model['Cumuliform0']
    if prior=='shit':
        alphas=np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    elif prior=='uniform':
        alphas=np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2])
    elif prior=='good':
        alphas=np.array([2,4,1,1,4,1,1,2,1,1,2,1,1,1,2])
    elif prior=='super':
        alphas=np.array([10,20,5,5,20,5,5,10,5,5,10,5,5,5,10])
    mu_list=[]
    var_list=[]
    for i in range(500):
        sample=np.random.choice(15,p=probs)
        alphas[sample]=alphas[sample]+1
        mu_list.append(scipy.stats.dirichlet.mean(alphas))
        variances=scipy.stats.dirichlet.var(alphas)
        var_list.append([math.sqrt(i) for i in variances])
    mu_rows_plot=np.zeros((len(mu_list),15))
    variance_rows_plot=np.zeros((len(mu_list),15))
    for m in range(len(mu_list)):
        for n in range(15):
            mu_rows_plot[m,n]=mu_list[m][n]
            variance_rows_plot[m,n]=var_list[m][n]
    for n in range(15):
        plt.plot(range(len(mu_list)),mu_rows_plot[:,n])
        plt.fill_between(range(len(mu_list)),mu_rows_plot[:,n]+variance_rows_plot[:,n],mu_rows_plot[:,n]-variance_rows_plot[:,n],alpha=0.5)

    plt.xlabel('Number of total observations')
    plt.ylabel('Means')
    plt.title('Dirichlet Distribution of Means')
    plt.show()
    #  print scipy.stats.dirichlet.mean(alphas)
    #  print scipy.stats.dirichlet.var(alphas)

if __name__ == '__main__':
    max_likelihood_total_samples()
    max_likelihood_ind_sample()
    dirichlet('shit')
    dirichlet('uniform')
    dirichlet('good')
    dirichlet('super')
