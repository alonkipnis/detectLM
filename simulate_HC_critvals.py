import numpy as np
from multitest import MultiTest
from tqdm import tqdm
import pandas as pd

nMonte = 10000
nn = [20, 50, 100, 200, 300, 500, 1000]
res = np.zeros((len(nn), nMonte))

for i,n in enumerate(nn):
    for j in tqdm(range(nMonte)):
        uu = np.random.rand(n)
        mt = MultiTest(uu, stbl=True)
        res[i,j] = mt.hc()[0]

def bootstrap_standard_error(xx, alpha, nBS = 1000):
    xxBS_vec = np.random.choice(xx, size=len(xx)*nBS, replace=True)
    xxBS = xxBS_vec.reshape([len(xx), -1])
    return np.quantile(xxBS, 1 - alpha, axis=0).std()

records = []
for al in [0.05, 0.01]:
    print(f"alpha={al}: n={nn}")
    for i,n in enumerate(nn):
        sBS = bootstrap_standard_error(res[i], 1 - al)
        print(f"{np.round(np.quantile(res[i], 1 - al), 3)} ({np.round(sBS,2)})", end=" | ")
        records.append(dict(alpha=al, n=n, q_alpha=np.quantile(res[i], 1 - al), std=sBS))
    print()

pd.DataFrame.from_dict(records).to_csv("HC_critvals.csv")