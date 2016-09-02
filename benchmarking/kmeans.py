import pandas as pd
import os
import sys
import time
import multiprocessing
from sklearn.cluster import KMeans

project = os.path.realpath('./..')
datadir = os.path.join(project, 'data')

with pd.HDFStore(os.path.join(datadir, 'tweets_1M.h5')) as store:
    subset = store.tweets_subset

# Convert the lat and lng columns to numpy array
data = subset.as_matrix(columns=['lat', 'lng'])


def kmean_fit_timing(data, n):

    k_means = KMeans(n_clusters=n,
                     init='k-means++',
                     n_init=10)

    t0 = time.time()
    k_means.fit(data)
    t1 = time.time() - t0

    with open('kmean.csv', mode='w') as timing:
        timing.write('{},'.format(t1))

        print n, t1
        return t1


if sys.argv[1]:
    end = int(sys.argv[1])
else:
    end = 10

if sys.argv[2]:
    step = int(sys.argv[2])
else:
    step = 1


for n in range(1, end, step):
    # Start bar as a process
    p = multiprocessing.Process(target=kmean_fit_timing, args=(data, n))
    p.start()

    # Wait for 10 seconds or until process finishes
    p.join(5)

    # If thread is still active
    if p.is_alive():
        print "running... let's kill it..."

        # Terminate
        p.terminate()
        p.join()
        break