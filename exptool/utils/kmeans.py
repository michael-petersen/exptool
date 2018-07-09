#
# kmeans.py
#
# robust implementation of kmeans based on
# https://datasciencelab.wordpress.com

import random
import numpy as np

import matplotlib.pyplot as plt
import matplotlib


class KMeans():
    '''
    class to implement K-means in Python


    '''
    
    def __init__(self, K, X=None, N=0):
        '''
        initialize kmeans


        inputs
        --------------
        self      : KMeans class
        K         : number of clusters
        X         : array of observations
        N         : error guard


        returns
        -------------
        self      : Kmeans class

        
        '''
        
        self.K = K
        
        try:

            tmp = len(X)
            self.X = X
            self.N = len(X)
                
        except:

            if N == 0:
                raise Exception("kmeans.KMeans: If no data is provided, \
                                 a parameter N (number of points) is needed")
            else:
                self.N = N
                self.X = self._init_board_gauss(N, K)
            
        self.mu = None
        self.clusters = None
        self.method = None
 
    def _init_board_gauss(self, N, k):
        '''
        _init_board_gauss
           initialize the guess points


        inputs
        -------------------
        self
        N
        k


        returns
        ------------------
        self
        X        : randomly partitioned clusters


        '''

        # number of points to put in each cluster
        n = float(N)/k
        
        X = []

        # set up
        for i in range(k):
            
            c = (random.uniform(-1,1), random.uniform(-1,1))
            
            s = random.uniform(0.05,0.15)

            # just reflecting--but this means that the clusters are forced to ahve the same number of points
            x = []
            while len(x) < n:
                
                a,b = np.array([np.random.normal(c[0],s),np.random.normal(c[1],s)])
                
                # Continue drawing points from the distribution in the range [-1,1]
                if abs(a) and abs(b)<1:
                    x.append([a,b])
                    
            X.extend(x)
        X = np.array(X)[:N]
        
        return X
 
 
    def _cluster_points(self):
        mu = self.mu
        clusters  = {}
        for x in self.X:
            bestmukey = min([(i[0], np.linalg.norm(x-mu[i[0]])) \
                             for i in enumerate(mu)], key=lambda t:t[1])[0]
            try:
                clusters[bestmukey].append(x)
            except KeyError:
                clusters[bestmukey] = [x]
        self.clusters = clusters
 
    def _reevaluate_centers(self):
        '''
        _reevaluate_centers
            draw new centers based on which center they are closest to
            NOTE that this can create asymmetric cluster sizes

        '''
        clusters = self.clusters
        newmu = []
        keys = sorted(self.clusters.keys())
        for k in keys:
            newmu.append(np.mean(clusters[k], axis = 0))
        self.mu = newmu
 
    def _has_converged(self):
        '''
        _has_converged
            check to see whether clusters change from one step to the next. if not, declare convergence!


        '''
        K = len(self.oldmu)
        return(set([tuple(a) for a in self.mu]) == \
               set([tuple(a) for a in self.oldmu])\
               and len(set([tuple(a) for a in self.mu])) == K)
 
    def find_centers(self, method='random'):
        '''
        find_centers
           iteratively select new centers

        inputs
        ----------------


        returns
        ---------------


        '''
        
        self.method = method
        X = self.X
        K = self.K
        
        #self.oldmu = random.sample(X, K)

        # draw K samples from the array of clusters
        self.oldmu = random.sample(list(X), K)

        
        if method != '++':
            # Initialize to K random centers

            # this has a python2/3 compatibility issue
            #self.mu = random.sample(X, K)
            self.mu = random.sample(list(X), K)
            
        while not self._has_converged():
            self.oldmu = self.mu
            # Assign all points in X to clusters
            self._cluster_points()
            # Reevaluate centers
            self._reevaluate_centers()



# interesting second class option

class KPlusPlus(KMeans):
    def _dist_from_centers(self):
        cent = self.mu
        X = self.X
        D2 = np.array([min([np.linalg.norm(x-c)**2 for c in cent]) for x in X])
        self.D2 = D2
 
    def _choose_next_center(self):
        self.probs = self.D2/self.D2.sum()
        self.cumprobs = self.probs.cumsum()
        r = random.random()
        ind = np.where(self.cumprobs >= r)[0][0]
        return(self.X[ind])
 
    def init_centers(self):
        #self.mu = random.sample(self.X, 1)
        self.mu = random.sample(list(self.X), 1)
        while len(self.mu) < self.K:
            self._dist_from_centers()
            self.mu.append(self._choose_next_center())
 
