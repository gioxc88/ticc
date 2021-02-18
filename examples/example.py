import numpy as np
import pandas as pd

from ticc import TICC

fname = 'example_data.txt'
X = pd.read_csv(fname, header=None)


ticc = TICC(
    window_size=1,
    n_clusters=8,
    lambda_parameter=11e-2,
    switch_penalty=600,
    max_iters=100,
    threshold=2e-5,
    n_jobs=1
)

ticc.fit(X.to_numpy())
