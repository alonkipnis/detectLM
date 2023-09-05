import numpy as np
from multitest import MultiTest
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt

nMonte = 10000
nn = [25, 50, 75, 90, 100, 115, 125, 150, 200, 250, 300, 400, 500]
res = np.zeros((len(nn), nMonte))

STBL = True

for i,n in enumerate(nn):
    for j in tqdm(range(nMonte)):
        uu = np.random.rand(n)
        mt = MultiTest(uu, stbl=STBL)
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
        q_alpha = np.quantile(res[i], 1 - al)
        print(f"{np.round(q_alpha, 3)} ({np.round(sBS,2)})", end=" | ")
        records.append(dict(alpha=al, n=n, q_alpha=q_alpha, std=sBS))
    print()

pd.DataFrame.from_dict(records).to_csv("HC_critvals.csv")
df = pd.read_csv('HC_critvals.csv')

plt.figure()
fig = plt.gcf()
fig.set_size_inches(10, 5, forward=True)

nn = df['n'].unique()
for c in df.groupby('alpha'):
    yy = c[1]['q_alpha'].values
    plt.plot(nn, yy, label=fr'$\alpha={c[0]}$')
    cu = yy + 1.96 * c[1]['std'].values
    cl = yy - 1.96 * c[1]['std'].values
    plt.fill_between(nn, cl, cu, alpha=.3)

plt.xticks(fontsize=20); plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.xlabel('n [token]', fontsize=20)
plt.ylabel(r'$\mathrm{HC}^{1-\alpha}$', fontsize=20)

plt.savefig("Figs/HC_critvals.png")
plt.show()