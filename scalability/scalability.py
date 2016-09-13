import numpy as np
import pandas as pd
import os
import multiprocessing
import sample_counters

# Load data and set up as numpy array
project = os.path.realpath('./..')
datadir = os.path.join(project, 'data')

store = pd.HDFStore(os.path.join(datadir, 'tweets_1M.h5'))
subset = store.tweets_subset

data = subset.as_matrix(columns=['lat', 'lng'])

# K-means testing: increase clusters until timeout

timeout = 60
step = 500
k = 100

for n in range(100, 100000, step):
    p = multiprocessing.Process(target=sample_counters.kmeans_samples, args=(data, k, n,
                                                                             'kmeans_scale_by{}.csv'.format(step)))
    p.start()
    p.join(timeout)
    if p.is_alive():
        print "Ending process at {} seconds".format(timeout)
        p.terminate()
        p.join()
        break

# Minibatch testing: increase clusters until timeout for various batch sizes

batch_sizes = [5, 10, 20, 50, 100]
timeout = 60
step = 500
k = 100

for batchsize in batch_sizes:

    for n in range(100, 100000, step):

        filename = 'minibatch_scale_by{}_batchsize{}.csv'.format(step, batchsize)

        p = multiprocessing.Process(target=sample_counters.minibatch_samples, args=(data, k, n, batchsize, filename))
        p.start()
        p.join(timeout)
        if p.is_alive():
            print "Ending process at {} seconds".format(timeout)
            p.terminate()
            p.join()
            break

# DBSCAN testing:
eps = 0.00965606698736
step = 500
timeout = 60

for n in range(100, 100000, step):

    filename = 'dbscan_scale_by{}.csv'.format(step)

    p = multiprocessing.Process(target=sample_counters.dbscan_samples, args=(data, eps, 100, n, filename))
    p.start()
    p.join(timeout)
    if p.is_alive():
        print "Ending process at {} seconds".format(timeout)
        p.terminate()
        p.join()
        break

store.close()

