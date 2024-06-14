"""
kmeans.py

purpose-built, robust implementation of kmeans based on
https://datasciencelab.wordpress.com

MSP 19 May 2024 Improve documentation; code cleanup

"""

import numpy as np
import random

class KMeans:
    """
    A class to implement K-means clustering in Python.
    """

    def __init__(self, K, X=None, N=0):
        """
        Initialize KMeans.

        Parameters
        ----------
        K : int
            Number of clusters.
        X : array-like, optional
            Array of observations.
        N : int, optional
            Number of points (needed if X is not provided).

        Raises
        ------
        Exception
            If no data is provided and N is not specified.
        """
        self.K = K
        
        if X is not None:
            self.X = X
            self.N = len(X)
        else:
            if N == 0:
                raise Exception("kmeans.KMeans: If no data is provided, a parameter N (number of points) is needed")
            else:
                self.N = N
                self.X = self._init_board_gauss(N, K)
            
        self.mu = None
        self.oldmu = None
        self.clusters = None
        self.method = None
 
    def _init_board_gauss(self, N, K):
        """
        Initialize the guess points using a Gaussian distribution.

        Parameters
        ----------
        N : int
            Number of points.
        K : int
            Number of clusters.

        Returns
        -------
        np.array
            Randomly partitioned clusters.
        """
        n = float(N) / K
        X = []

        for i in range(K):
            c = (random.uniform(-1, 1), random.uniform(-1, 1))
            s = random.uniform(0.05, 0.15)

            cluster_points = []
            while len(cluster_points) < n:
                a, b = np.array([np.random.normal(c[0], s), np.random.normal(c[1], s)])
                if abs(a) < 1 and abs(b) < 1:
                    cluster_points.append([a, b])
            X.extend(cluster_points)
        X = np.array(X)[:N]
        
        return X
 
    def _cluster_points(self):
        """
        Assign each point to the nearest cluster center.
        """
        mu = self.mu
        clusters = {}
        for x in self.X:
            bestmukey = min([(i[0], np.linalg.norm(x - mu[i[0]])) for i in enumerate(mu)], key=lambda t: t[1])[0]
            try:
                clusters[bestmukey].append(x)
            except KeyError:
                clusters[bestmukey] = [x]
        self.clusters = clusters
 
    def _reevaluate_centers(self):
        """
        Compute new cluster centers based on the current cluster assignments.
        """
        clusters = self.clusters
        newmu = []
        keys = sorted(clusters.keys())
        for k in keys:
            newmu.append(np.mean(clusters[k], axis=0))
        self.mu = newmu
 
    def _has_converged(self):
        """
        Check if the algorithm has converged.

        Returns
        -------
        bool
            True if the cluster centers do not change, False otherwise.
        """
        K = len(self.oldmu)
        return (set([tuple(a) for a in self.mu]) == set([tuple(a) for a in self.oldmu]) and
                len(set([tuple(a) for a in self.mu])) == K)
 
    def find_centers(self, method='random', nitermax=1000):
        """
        Find the cluster centers using the specified method.

        Parameters
        ----------
        method : str, optional
            Method to initialize the cluster centers ('random' or '++', default is 'random').
        nitermax : int, optional
            Maximum number of iterations (default is 1000).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the specified method is not supported.
        """
        self.method = method
        X = self.X
        K = self.K

        if method == '++':
            # Initialize using K-means++
            self.mu = self._init_kmeans_plusplus()
        elif method == 'random':
            # Initialize to K random centers
            self.mu = random.sample(list(X), K)
        else:
            raise ValueError("Unsupported method: {}".format(method))

        iter = 0
        while not self._has_converged() and iter < nitermax:
            self.oldmu = self.mu
            self._cluster_points()
            self._reevaluate_centers()
            iter += 1

        if iter == nitermax:
            print('kmeans.KMeans.find_centers: Maximum iterations hit. Check conditions.')

    def _init_kmeans_plusplus(self):
        """
        Initialize cluster centers using the K-means++ algorithm.

        Returns
        -------
        list
            List of initial cluster centers.
        """
        centers = [random.choice(self.X)]
        while len(centers) < self.K:
            D2 = np.array([min([np.linalg.norm(x - c)**2 for c in centers]) for x in self.X])
            probabilities = D2 / D2.sum()
            cumulative_probabilities = probabilities.cumsum()
            r = random.random()
            for i, p in enumerate(cumulative_probabilities):
                if r < p:
                    centers.append(self.X[i])
                    break
        return centers


class KPlusPlus(KMeans):
    """
    A class to implement K-means++ initialization for K-means clustering.
    """

    def _dist_from_centers(self):
        """
        Compute the distance from each point to the nearest cluster center.

        Returns
        -------
        None
        """
        cent = self.mu
        X = self.X
        D2 = np.array([min([np.linalg.norm(x - c)**2 for c in cent]) for x in X])
        self.D2 = D2
 
    def _choose_next_center(self):
        """
        Choose the next cluster center probabilistically based on the distance squared.

        Returns
        -------
        np.array
            The next cluster center.
        """
        self.probs = self.D2 / self.D2.sum()
        self.cumprobs = self.probs.cumsum()
        r = random.random()
        ind = np.where(self.cumprobs >= r)[0][0]
        return self.X[ind]
 
    def init_centers(self):
        """
        Initialize the cluster centers using K-means++ initialization.

        Returns
        -------
        None
        """
        self.mu = random.sample(list(self.X), 1)
        while len(self.mu) < self.K:
            self._dist_from_centers()
            self.mu.append(self._choose_next_center())
