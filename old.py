
    def updateProbs(self,real_target):
        names = ['Cumuliform0','Cumuliform1','Cumuliform2','Cumuliform3','Cumuliform4']
        obs_names=['Yes-0','No-0','Yes-1','No-1','Yes-2','No-2','Yes-3','No-3','Yes-4','No-4']
        
        # initialize Dir sample
        num_samples=5000
        sample_check=[]
        theta2_static=np.empty((50,10))
        postX=copy.deepcopy(self.probs)
        all_post=np.zeros((int(num_samples/5),1,5))
        all_theta2=np.zeros((int(num_samples/5),50,10))
        for X in range(5):
            for prev_obs in range(10):
                theta2_static[X*10+prev_obs,:]=scipy.stats.dirichlet.mean(alpha=self.table[X,prev_obs,:])
        if len(self.obs)>0:
            prev_obs=self.obs[-1]
            self.obs.append(np.random.choice(range(10),p=self.theta2_correct[real_target*10+prev_obs,:]))
        else:
            self.obs.append(np.random.choice(range(10),p=self.theta1[real_target,:]))
        # confusion matrix for human
        if self.obs[-1]%2==0:
            self.pred_obs.append(0)
            if (self.obs[-1]/2)==real_target:
                self.real_obs.append(0)
            else:
                self.real_obs.append(1)
        else:
            self.pred_obs.append(1)
            if (int(self.obs[-1]/2))==real_target:
                self.real_obs.append(0)
            else:
                self.real_obs.append(1)

        theta2=copy.deepcopy(theta2_static)
        #  print "Observation: %s" % obs_names[self.obs[-1]]
        for n in range(num_samples):
            for i in names:
                likelihood=self.theta1[names.index(i),self.obs[0]]
                # sample from theta2
                if len(self.obs)>1:
                    for value in self.obs[1:]:
                        likelihood*=theta2[names.index(i)*10+self.obs[self.obs.index(value)-1],value]
                #  print likelihood
                postX[i]=self.probs[i]*likelihood
            suma=sum(postX.values())
            # normalize
            for i in names:
                postX[i]=np.log(postX[i])-np.log(suma) 
                postX[i]=np.exp(postX[i])
            if n%5==0:
                all_post[int(n/5),:,:]=postX.values()
            # sample from X
            X=np.random.choice(range(5),p=postX.values())
            alphas=copy.deepcopy(self.table)
            theta2=copy.deepcopy(theta2_static)
            if len(self.obs)>1:
                alphas[X,self.obs[-2],self.obs[-1]]+=1
                theta2[X*10+self.obs[-2],:]=np.random.dirichlet(alphas[X,self.obs[-2],:])
                if n%5==0:
                    all_theta2[int(n/5),X*10+self.obs[-2],:]=theta2[X*10+self.obs[-2],:]

        if len(self.obs)>1:
            sample_counts=np.zeros((50,10))
            # estimation of alphas from distributions
            for n in range(all_theta2.shape[1]):
                pk_top_list=[]
                sum_alpha=sum(self.table[int(n/10),n%10,:])
                for k in range(all_theta2.shape[2]):
                    samples=all_theta2[np.nonzero(all_theta2[:,n,k]),n,k]
                    if len(samples[0])==0:
                        pass
                    else:
                        sample_counts[n,k]=len(samples[0])
                        pk_top_list.append(np.mean(samples[0]))
                        current_alpha=self.table[int(n/10),n%10,k]
                        for x in range(5):
                            sum_alpha_old=sum_alpha-current_alpha+self.table[int(n/10),n%10,k]
                            logpk=np.sum(np.log(samples[0]))/len(samples[0])
                            y=psi(sum_alpha_old)+logpk
                            if y>=-2.22:
                                alphak=np.exp(y)+0.5
                            else:
                                alphak=-1/(y+psi(1))
                            #  print "start:",alphak
                            for w in range(5):
                                alphak-=((psi(alphak)-y)/polygamma(1,alphak))
                            self.table[int(n/10),n%10,k]=alphak

        post_probs=np.mean(all_post,axis=0)
        for i in names:
            self.probs[i]=post_probs[0][names.index(i)]
