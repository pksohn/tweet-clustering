import pandas as pd
import os
import time
import multiprocessing
from sklearn.cluster import KMeans, MiniBatchKMeans
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batchstart')
parser.add_argument('--batchstop')
parser.add_argument('--step')
parser.add_argument('--timeout')
args = parser.parse_args()

batch_start, batch_end, step, timeout = 5, 100, 5, 10
if args.batchstart:
    start = int(args.batchstart)
if args.batchstop:
    end = int(args.batchstop)
if args.step:
    step = int(args.step)
if args.timeout:
    timeout = float(args.timeout)


# Load data and set up as numpy array
project = os.path.realpath('./../..')
datadir = os.path.join(project, 'data')

store = pd.HDFStore(os.path.join(datadir, 'tweets_1M.h5'))
subset = store.tweets_subset

data = subset.as_matrix(columns=['lat', 'lng'])


def kmeans_fit_timing(data, n):
    """
    Fit data to a K-means model with n clusters and record processing time to a csv file.

    :param data: Numpy array with data to classify
    :param n: Number of clusters to generated
    :return: Processing time (in seconds)
    """

    k_means = KMeans(n_clusters=n, init='k-means++', n_init=10)

    t0 = time.time()
    k_means.fit(data)
    t1 = time.time() - t0

    with open('kmeans_reference.csv', mode='a') as timing:
        timing.write('kmeans,{},{},\n'.format(n, t1))

    print n, t1
    return t1


def minibatch_timing(data, n, batch_size):
    """
    Fit data to a minibatch K-means model with n clusters and record processing time to a csv file.

    :param data: Numpy array with data to classify
    :param n: Number of clusters to generated
    :param batch_size: Desired batch size
    :param filename: Name of CSV file to save results for each iteration
    :return: Processing time (in seconds)
    """

    mb = MiniBatchKMeans(n_clusters=n, init='k-means++', n_init=10, batch_size=batch_size, init_size=max(30, n))

    t0 = time.time()
    mb.fit(data)
    t1 = time.time() - t0

    with open('kmeans_reference.csv', mode='a') as timing:
        timing.write('minibatch,{},{},{}\n'.format(n, t1, batch_size))

    print n, t1
    return t1


# Write new csv file
with open('kmeans_reference.csv', mode='w') as timing:
    timing.write('algorithm,clusters,seconds,batchsize\n')

# run kmeans test
kmeans_fit_timing(data=data, n=100)


# run minibatch for several minibatch sizes
for batchsize in range(batch_start, batch_end, step):
    p = multiprocessing.Process(target=minibatch_timing, args=(data, 100, batchsize))
    p.start()
    p.join(timeout)
    if p.is_alive():
        print "Ending process at {} seconds".format(timeout)
        p.terminate()
        p.join()
        break

store.close()