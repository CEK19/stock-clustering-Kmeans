from random import random
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(
    centers=6, n_samples=100, n_features=2, shuffle=True, random_state=40
)

kmeans = KMeans(n_clusters=6, random_state=0).fit(X)
print(kmeans.labels_)

  