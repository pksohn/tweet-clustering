import pandas as pd
import os
import time
import multiprocessing
from sklearn.cluster import KMeans
import argparse


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--start')
parser.add_argument('--stop')
parser.add_argument('--step')
parser.add_argument('--timeout')
args = parser.parse_args()

start, end, step, timeout = 1, 10, 1, 10
if args.start:
    start = int(args.start)
if args.stop:
    end = int(args.stop)
if args.step:
    step = int(args.step)
if args.timeout:
    timeout = float(args.timeout)


# Load data and set up as numpy array
project = os.path.realpath('./..')
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

    with open('kmeans.csv', mode='a') as timing:
        timing.write('{},{}\n'.format(n, t1))

    print n, t1
    return t1

# Write new csv file
with open('kmeans.csv', mode='w') as timing:
    timing.write('clusters,seconds\n')

# Run function and end when processing time reaches a certain threshold
for n in range(1, end, step):
    p = multiprocessing.Process(target=kmeans_fit_timing, args=(data, n))
    p.start()
    p.join(timeout)
    if p.is_alive():
        print "Ending process at {} seconds".format(timeout)
        p.terminate()
        p.join()
        break

store.close()