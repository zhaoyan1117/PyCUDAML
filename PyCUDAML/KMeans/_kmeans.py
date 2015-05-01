from __future__ import absolute_import

import numpy as np

from ._KMeans import kmeans

class KMeans(object):

    def __init__(self, n_clusters, max_iter=1000, threshold=0.001, seed=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.threshold = threshold
        self.seed = seed;

    def fit(self, X):
        self.means, self.cluster_assignments, \
        self.total_iter, self.loss, self.delta_percent \
            = kmeans(self.n_clusters,
                     X.astype(np.float32),
                     self.max_iter,
                     self.threshold,
                     self.seed)

        return self.cluster_assignments
