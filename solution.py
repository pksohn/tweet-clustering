from __future__ import print_function
from __future__ import division
import os
import time
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans, DBSCAN

# Get rid of sklearn deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load data from HDF5 file
datadir = os.path.realpath('./data')
with pd.HDFStore(os.path.join(datadir, 'tweets_1M.h5')) as store:
    tweets = store.tweets

# Constrain to Bay Area
tweets = tweets.loc[(tweets.lat > 36.903929764) &
                    (tweets.lat < 38.853939589) &
                    (tweets.lng > -123.528897483) &
                    (tweets.lng < -121.213352822)]
print('Size of full dataset: {}'.format(len(tweets)))

# Set index to id for easy matching
tweets.set_index('id', inplace=True)

# Start timing implementation
t0 = time.time()

# MiniBatch section
mb = MiniBatchKMeans(n_clusters=100, init='k-means++', n_init=10, batch_size=1000)
data = tweets.as_matrix(columns=['lat', 'lng'])
mb.fit(data)
tweets['mb_cluster'] = mb.labels_   # Add labels back into DataFrame

# DBSCAN section
meters = 100      # Transform meters to degrees (roughly)
eps = meters / 100000

for i in tweets.mb_cluster.unique():
    subset = tweets.loc[tweets.mb_cluster == i]
    db = DBSCAN(eps=eps, min_samples=100)
    data = subset.as_matrix(columns=['lat', 'lng'])
    db.fit(data)
    subset['db_cluster'] = db.labels_
    tweets.loc[tweets.mb_cluster == i, 'db_cluster'] = subset['db_cluster']

# Set final cluster variable
tweets['cluster'] = tweets.mb_cluster + (tweets.db_cluster.replace(-1.0, np.nan) / 100)
print('Number of unique clusters generated: {}'.format(len(tweets.cluster.unique())))

t1 = time.time() - t0
print('Implementation time: {}'.format(t1))

# Save results
with pd.HDFStore(os.path.join(datadir, 'results.h5'), mode='w') as results:
    results['results'] = tweets

