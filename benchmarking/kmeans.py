import pandas as pd
import os
import time
import multiprocessing
from sklearn.cluster import KMeans
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start')
parser.add_argument('--stop')
parser.add_argument('--step')
parser.add_argument('--timeout')
args = parser.parse_args()


project = os.path.realpath('./..')
datadir = os.path.join(project, 'data')

with pd.HDFStore(os.path.join(datadir, 'tweets_1M.h5')) as store:
    subset = store.tweets_subset
data = subset.as_matrix(columns=['lat', 'lng'])


def kmean_fit_timing(data, n):

    k_means = KMeans(n_clusters=n, init='k-means++', n_init=10)

    t0 = time.time()
    k_means.fit(data)
    t1 = time.time() - t0

    with open('kmeans.csv', mode='a') as timing:
        timing.write('{},{}\n'.format(n, t1))

        print n, t1
        return t1


start, end, step, timeout = 1, 10, 1, 10
if args.start:
    start = int(args.start)
if args.stop:
    end = int(args.stop)
if args.step:
    step = int(args.step)
if args.timeout:
    timeout = float(args.timeout)


with open('kmeans.csv', mode='w') as timing:
    timing.write('clusters,seconds\n')

for n in range(1, end, step):
    p = multiprocessing.Process(target=kmean_fit_timing, args=(data, n))
    p.start()
    p.join(timeout)
    if p.is_alive():
        print "Ending process at {} seconds".format(timeout)
        p.terminate()
        p.join()
        break
