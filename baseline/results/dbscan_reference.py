import pandas as pd
import os
import time
import multiprocessing
from sklearn.cluster import DBSCAN
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epstart')
parser.add_argument('--epstop')
parser.add_argument('--epstep')
parser.add_argument('--timeout')
args = parser.parse_args()

eps_start, eps_stop, eps_step, timeout = .25, 2.0, 0.5, 60
if args.batchstart:
    start = float(args.batchstart)
if args.batchstop:
    end = float(args.batchstop)
if args.step:
    step = float(args.step)
if args.timeout:
    timeout = float(args.timeout)


# Load data and set up as numpy array
project = os.path.realpath('./../..')
datadir = os.path.join(project, 'data')

store = pd.HDFStore(os.path.join(datadir, 'tweets_1M.h5'))
subset = store.tweets_subset

data = subset.as_matrix(columns=['lat', 'lng'])

