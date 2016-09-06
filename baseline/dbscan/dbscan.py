import pandas as pd
import os
import time
import multiprocessing
from sklearn.cluster import DBSCAN
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--start')
parser.add_argument('--stop')
parser.add_argument('--step')
parser.add_argument('--timeout')
parser.add_argument('--samples')
args = parser.parse_args()

start, end, step, timeout, samples = 1, 10, 1, 10, 100
if args.start:
    start = float(args.start)
if args.stop:
    end = float(args.stop)
if args.step:
    step = float(args.step)
if args.timeout:
    timeout = float(args.timeout)
if args.samples:
    samples = float(args.samples)


# Load data and set up as numpy array
project = os.path.realpath('./../..')
datadir = os.path.join(project, 'data')

store = pd.HDFStore(os.path.join(datadir, 'tweets_1M.h5'))
subset = store.tweets_subset

data = subset.as_matrix(columns=['lat', 'lng'])


def dbscan_epsilon_check(data, eps, samples):
    """
    Fit data to a K-means model with n clusters and record processing time to a csv file.

    :param data: Numpy array with data to classify
    :param eps: Epsilon, or maximum distance for two samples to be considered in the same neighborhood
    :param samples: Minimum number of samples in a neighborhood for a point to be considered a core point.
    :return: Processing time (in seconds)
    """

    dbscan = DBSCAN(eps=eps,
                    min_samples=samples)

    t0 = time.time()
    dbscan.fit(data)
    t1 = time.time() - t0

    with open('dbscan.csv', mode='a') as timing:
        timing.write('{},{}\n'.format(eps, t1))

    print 'epsilon: {}, minimum samples: {}, time: {}'.format(eps, samples, t1)
    return t1


# Write new csv file
with open('dbscan.csv', mode='w') as timing:
    timing.write('clusters,seconds\n')

# Run function and end when processing time reaches a certain threshold
for eps in range(start, end, step):
    p = multiprocessing.Process(target=dbscan_epsilon_check, args=(data, eps, samples))
    p.start()
    p.join(timeout)
    if p.is_alive():
        print "Ending process at {} seconds".format(timeout)
        p.terminate()
        p.join()
        break

store.close()
