import pandas as pd
import os
import time
import multiprocessing
from sklearn.cluster import MiniBatchKMeans
import argparse

# Make sklearn be quiet
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--start')
parser.add_argument('--stop')
parser.add_argument('--step')
parser.add_argument('--timeout')
parser.add_argument('--batchsize')
args = parser.parse_args()

start, end, step, timeout, batchsize = 1, 10, 1, 10, 100
if args.start:
    start = int(args.start)
if args.stop:
    end = int(args.stop)
if args.step:
    step = int(args.step)
if args.timeout:
    timeout = float(args.timeout)
if args.batchsize:
    batchsize = int(args.batchsize)


# Load data and set up as numpy array
project = os.path.realpath('./../..')
datadir = os.path.join(project, 'data')

store = pd.HDFStore(os.path.join(datadir, 'tweets_1M.h5'))
subset = store.tweets_subset

data = subset.as_matrix(columns=['lat', 'lng'])


def minibatch_timing(data, n, batch_size, filename):
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

    with open(filename, mode='a') as timing:
        timing.write('{},{}\n'.format(n, t1))

    print n, t1
    return t1


filename = 'minibatch_{}_to_{}_by_{}_batchsize_{}.csv'.format(start, end, step, batchsize)

# Write new csv file
with open(filename, mode='w') as timing:
    timing.write('clusters,seconds\n')

# Run function and end when processing time reaches a certain threshold
for n in range(1, end, step):
    p = multiprocessing.Process(target=minibatch_timing, args=(data, n, batchsize, filename))
    p.start()
    p.join(timeout)
    if p.is_alive():
        print "Ending process at {} seconds".format(timeout)
        p.terminate()
        p.join()
        break

store.close()
