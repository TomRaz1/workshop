# MIT License

# Copyright (c) 2021 Arthur Spencer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

'''
k-means clustering script for clustering encoded data following dimensionality
reduction, or clustering raw dFC windows with either L1 or L2 distance
'''

import numpy as np
from pyclustering.cluster.kmeans import kmeans as pykmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import distance_metric, type_metric
import multiprocessing as proc
import mod

from sklearn.cluster import KMeans

def init_kmeans(in_):
    '''
    Initialize k-means using scikit-learn with fixed seed.
    '''
    xshape = in_[0]
    n_clusters = in_[1]
    dist = in_[2]
    seed = in_[3]

    if seed is not None:
        seed = int(seed)

    X = np.reshape(mod.global_X[:], xshape)

    kmeans_ = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10,
                     random_state=seed, max_iter=1000)
    kmeans_.fit(X)

    cent = kmeans_.cluster_centers_
    score = kmeans_.inertia_

    return cent, score


def init_process(X):
	mod.global_X = X


def kmeans(X, exemplar_inds, n_clusters=2, dist='euclidean'):
	'''
	Initialise k-means n_kmeans_init times in parallel using exemplar FC windows,
	then choose the set with the lowest sum of squared errors to initialise a run
	of k-means on all data points.
	'''

	if len(X)>6e4: # Must be in series for large amounts of data (hardware dependent)
		run_parallel = False
	else:
		run_parallel = True

	n_kmeans_init = 128

	seeds = np.random.randint(999, size=n_kmeans_init)
	xshape = X[exemplar_inds].shape

	if run_parallel:
		in_ = [(xshape, n_clusters, dist, seed) for seed in seeds]
		p = proc.Pool(8)
		init_X = proc.Array('f', np.reshape(X[exemplar_inds],-1), lock=False)

		with proc.get_context("spawn").Pool(8, initializer=init_process, initargs=(init_X,)) as pool:
			init_out = pool.map(init_kmeans, in_)
		pool.close()

		init_centroids = [None]*n_kmeans_init
		init_score = np.zeros(n_kmeans_init)
		for i in range(n_kmeans_init):
			init_centroids[i] = init_out[i][0]
			init_score[i] = init_out[i][1]

	else:
		init_process(np.reshape(X[exemplar_inds],-1))
		init_centroids = [None]*n_kmeans_init
		init_score = np.zeros(n_kmeans_init)
		for i, seed in enumerate(seeds):
			init_centroids[i], init_score[i] = init_kmeans((xshape, n_clusters, dist, seed))
			print('%i done' % i)

	print('Got k-means initialisers')

	# Choose the set of centroids with the lowest sum of squared errors
	init_centroids = init_centroids[np.argmin(init_score)]

	if dist=='l1':
		dist_metric = distance_metric(type_metric.MANHATTAN)
	else:
		dist_metric = distance_metric(type_metric.EUCLIDEAN)

	# perform a full run of kmeans on all data points with the chosen initial centroids
	kmeans_ = pykmeans(X, init_centroids, tolerance=1e-6, itermax=10000, metric=dist_metric)
	kmeans_.process()
	clusters = kmeans_.get_clusters()
	centroids = kmeans_.get_centers()
	score = kmeans_.get_total_wce()

	return clusters, centroids, score