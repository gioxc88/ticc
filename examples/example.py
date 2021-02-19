from pathlib import Path

import numpy as np
import pandas as pd
from ticc import TICC

try:
    path = Path(__file__).absolute().parent.parent
except NameError:
    path = Path().absolute()

fname = path / 'data/data.txt'

X = pd.read_csv(fname, header=None)
np.random.seed(102)

cluster = TICC(
    window_size=1,
    n_clusters=8,
    lambda_parameter=11e-2,
    switch_penalty=600,
    max_iters=100,
    threshold=2e-5,
    n_jobs=-1,
    verbose=1
)
self = cluster
cluster.fit(X)


