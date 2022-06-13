import numpy as np
import matplotlib.pyplot as plt
# import statistics

np.random.seed(42)

def EuclideanDistanceCal(xi, xj):
    return np.sqrt(np.sum((xi - xj)**2))

def squaredDeviation(xi, mean):
    return np.sum((xi - mean)**2)

class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        
        # List of samples for each cluster
        self.clusters = [[] for _ in range(self.K)] # Contain index of each data point.
        self.centroids = [] # Contain coordinate of centroid points
        
        # inertia
        self.inertia_ = 0
    
            
    def _initCenters(self):
        clusters = [[] for _ in range(self.K)]
        X = self.X
        
        # Calculate variances of each dimension 
        variance_x = np.var(X, axis=0)
        
        #loop find max variance
        if (variance_x[0]) :
            X.sort(key=lambda column: column[0]) 
        else :
            X.sort(key=lambda column: column[1])       
        
        # push X into cluster to divide to k subsets
        for idxCluster in range (self.K):           
            for idxSample, _ in enumerate(self.X):
                clusters[idxCluster].append(idxSample)

        # find median 
        
        for idxCluster in range (self.K):           
            clusters[idxCluster].append(idxSample)

        # 
        
        
        return self.centroids

    def _closestCentroid(self, sample, centroids):
        #  Calculate distance of the current sample to each centroid
        distances = []
        for centroid in centroids:
            distance = EuclideanDistanceCal(sample, centroid)
            distances.append(distance)
        #  Find the labels that make minimum distance
        return np.argmin(distances)

    def _createClusters(self, centroids):
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idxSample, sample in enumerate(self.X):
            idxCentroid = self._closestCentroid(sample, centroids)
            clusters[idxCentroid].append(idxSample)
        return clusters
        
    def _getClusterLabels(self, clusters):
        # Each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)        
        for idxCluster, cluster in enumerate(clusters):
            for idxSample in cluster:
                labels[idxSample] = idxCluster
        return labels
    
    def _getCentroids(self, clusters):
        # Assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for idxCluster, cluster in enumerate(clusters):
            meanClusterValue = np.mean(self.X[cluster], axis=0)
            centroids[idxCluster] = meanClusterValue
        return centroids
    
    def _calcInertia(self):
        wcss = []
        for idxCluster, cluster  in enumerate(self.clusters):
            centroid = self.centroids[idxCluster]
            subsetData  = self.clusters[idxCluster]
            for idxData in subsetData:
                point = self.X[idxData]
                wcss.append(squaredDeviation(point, centroid))
        return sum(wcss)    
    
    def _isConverged(self, prevCentroids, currCentroids):
        # Distances between each old and new centroids, fol all centroids
        distances = []
        for idx in range(self.K):
            distance = EuclideanDistanceCal(prevCentroids[idx], currCentroids[idx])
            distances.append(distance)
        return sum(distances) == 0
    
    def visualize(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)
        plt.show()
    
    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # STEP 1: Initialize
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]        
        # self._initCenters() # TODO Ở ĐÂY NÈ BẠN, XOÁ 2 DÒNG TRÊN ĐI
        
        # STEP 2: Clustering 
        for _ in range(self.max_iters):
            # Assigning samples to the closest centroids 
            self.clusters = self._createClusters(self.centroids)
            # if self.plot_steps:
            #     self.visualize()
            
            # Calculate new centroids from clusters
            prevCentroids = self.centroids
            self.centroids = self._getCentroids(self.clusters)
            
            # Update inertia_
            self.inertia_ = self._calcInertia()
            
            # If clusters have changed -> repeated, if not -> break
            if self._isConverged(prevCentroids=prevCentroids, currCentroids=self.centroids):
                break
            # if self.plot_steps:
            #     self.visualize()
        # Classify samples as the index of their clusters
        return self._getClusterLabels(self.clusters)