from copy import copy
import numpy as np
from numpy.linalg import norm

class Kmeans:
    def __init__(self,
                num_cl, 
                tol,
                max_iters,
                normalize_data, 
                verbose):
        
        self.num_cl = num_cl
        self.tol = tol
        self.max_iters = max_iters
        self.normalize_data = normalize_data
        self.verbose = verbose
        
    def normalize(self, data):
        if self.normalize_data : 
            return (data - data.mean())/data.std()
        else:
            return data
        
    def cluster(self, observations):
        observations = self.normalize(observations)
        num_exmpls = observations.shape[0]
        
        # initialize random clusters
        random_index = np.random.choice(range(num_exmpls), self.num_cl)
        clusters = observations[random_index]
        cluster_labels = np.zeros(shape=num_exmpls)
        cluster_distances = np.zeros(shape=(num_exmpls, self.num_cl))
        
        
        while self.max_iters > 0:
        
            # find closest observations to clusters

            for cluster_index in range(self.num_cl):
                cluster_distances[:,cluster_index] = norm(observations - clusters[cluster_index], axis=1)
                
            # find closest points to clustersr
            cluster_labels = np.argmin(cluster_distances, axis=1)

            old_clusters = copy(clusters)

            well_done = False

            # find and update clusters
            for idx in range(self.num_cl):
                clusters[idx] = observations[cluster_labels==idx].mean(0)
                
            # check if clusters almost do not move
            
            dist_var =  norm(old_clusters - clusters)/norm(old_clusters)
            if dist_var < self.tol:
                well_done = True
                if self.verbose:
                    print(f'stop with dist variance : {dist_var} on iteration {self.max_iters}')
            else:
                if self.verbose:
                    print(f'current dist variance : {dist_var}  iteration num {self.max_iters}')

            if well_done:
                break
            
            self.max_iters -=1
        
        return cluster_labels, clusters