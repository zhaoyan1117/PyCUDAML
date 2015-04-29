from __future__ import absolute_import

from ._KMeans import kmeans

class KMeans(object):

    def __init__(self, n_clusters, max_iter=1000, threshold=0.001):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.threshold = threshold

    def fit(self, X):
        self.means, self.cluster_assignments \
            = kmeans(self.n_clusters, X, self.max_iter, self.threshold)

        return self.cluster_assignments
