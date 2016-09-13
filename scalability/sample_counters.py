from __future__ import print_function
import os
import time
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def kmeans_samples(data, k, n, filename):
    """
    Fit data to a K-means model with n clusters and m samples and record processing time to a csv file.

    :param data: Numpy array with data to classify
    :param k: Number of clusters to generated
    :param n: Desired sample size, generated randomly
    :return: Processing time (in seconds)
    """

    k_means = KMeans(n_clusters=k, init='k-means++', n_init=10)

    data = data[np.random.randint(low=0, high=len(data), size=n), :]

    t0 = time.time()
    k_means.fit(data)
    t1 = time.time() - t0

    project = os.path.realpath('.')
    csv = os.path.join(project, filename)

    if not os.path.exists(csv):
        with open(csv, mode='w') as timing:
            timing.write('samples,clusters,seconds\n')

    with open(csv, mode='a') as timing:
        timing.write('{},{},{}\n'.format(n, k, t1))

    print('K-means: {} samples, {} clusters, {} seconds'.format(n, k, t1))
    return t1


def dbscan_samples(data, eps, min_samples, n, filename):

    dbscan = DBSCAN(eps=eps,
                    min_samples=min_samples)

    data = data[np.random.randint(low=0, high=len(data), size=n), :]

    t0 = time.time()
    dbscan.fit(data)
    t1 = time.time() - t0

    clusters = len(np.unique(dbscan.labels_))

    project = os.path.realpath('.')
    csv = os.path.join(project, filename)

    if not os.path.exists(csv):
        with open(csv, mode='w') as timing:
            timing.write('epsilon,min_samples,n,clusters,seconds\n')

    with open(csv, mode='a') as timing:
        timing.write('{},{},{},{},{}\n'.format(eps, min_samples, n, clusters, t1))

    print('DBSCAN: epsilon={}, {} samples, {} clusters, {} seconds'.format(eps, n, clusters, t1))

    return t1


def minibatch_samples(data, k, n, batch_size, filename):
    """
    Fit data to a minibatch K-means model with n clusters and record processing time to a csv file.

    :param data: Numpy array with data to classify
    :param k: Number of clusters to generated
    :param n: Sample size, generated randomly
    :param batch_size: Desired batch size
    :return: Processing time (in seconds)
    """

    mb = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=10, batch_size=batch_size, init_size=max(30, k))

    data = data[np.random.randint(low=0, high=len(data), size=n), :]

    t0 = time.time()
    mb.fit(data)
    t1 = time.time() - t0

    project = os.path.realpath('.')
    csv = os.path.join(project, filename)

    if not os.path.exists(csv):
        with open(csv, mode='w') as timing:
            timing.write('samples,clusters,seconds,batchsize\n')

    with open(csv, mode='a') as timing:
        timing.write('{},{},{},{}\n'.format(n, k, t1, batch_size))

    print('Minibatch K-means: {} samples, {} clusters, {} seconds, {} batch size'.format(n, k, t1, batch_size))

    return t1