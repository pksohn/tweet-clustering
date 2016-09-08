import pandas as pd
import os
import multiprocessing
import timers


# Load data and set up as numpy array
project = os.path.realpath('./..')
datadir = os.path.join(project, 'data')

store = pd.HDFStore(os.path.join(datadir, 'tweets_1M.h5'))
subset = store.tweets_subset

data = subset.as_matrix(columns=['lat', 'lng'])

# K-means testing: increase clusters until timeout

timeout = 60
step = 5

for n in range(1, 1000, step):
    p = multiprocessing.Process(target=timers.kmeans_baseline, args=(data, n, 'kmeans_by{}.csv'.format(step)))
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
step = 50

for batchsize in batch_sizes:

    for n in range(1, 50000, step):

        filename = 'minibatch_by{}_batchsize{}.csv'.format(step, batchsize)

        p = multiprocessing.Process(target=timers.minibatch_timing, args=(data, n, batchsize, filename))
        p.start()
        p.join(timeout)
        if p.is_alive():
            print "Ending process at {} seconds".format(timeout)
            p.terminate()
            p.join()
            break

# DBSCAN testing:

# Distances in miles
start = 0.1
stop = 2.0
step = 0.1

for miles in range(start, stop, step):

    filename = 'dbscan_from{}_to{}_by{}.csv'.format(start, stop, step)

    # Rough approximation: 100 km / 1 degree lat or long
    kilometers = miles / 0.621371
    epsilon_degrees = kilometers / 100

    p = multiprocessing.Process(target=timers.dbscan_baseline, args=(data, epsilon_degrees, 100, filename))
    p.start()
    p.join(timeout)
    if p.is_alive():
        print "Ending process at {} seconds".format(timeout)
        p.terminate()
        p.join()
        break

store.close()