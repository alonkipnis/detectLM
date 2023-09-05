

import numpy as np
import pandas as pd
from multitest import MultiTest
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline
from src.fit_survival_function import fit_survival_func


def get_HC_survival_function(HC_null_sim_file=None, log_space=True, nMonte=10000, STBL=True):

    xx = {}
    if HC_null_sim_file is None:            
            nn = [25, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500]
            for n in nn:
                yy = np.zeros(nMonte)
                for j in tqdm(range(nMonte)):
                    uu = np.random.rand(n)
                    mt = MultiTest(uu, stbl=STBL)
                    yy[j] = mt.hc()[0]
                xx[n] = yy
    else:
        df = pd.read_csv(HC_null_sim_file, index_col=0)
        for n in df.index:
            xx[n] = df.loc[n]
        nn = df.index.tolist()

    xx0 = np.linspace(-1, 10, 57)
    zz = []
    for n in nn:
        univariate_survival_func = fit_survival_func(xx[n], log_space=log_space)
        zz.append(univariate_survival_func(xx0))
        
    func_log = RectBivariateSpline(np.array(nn), xx0, np.vstack(zz))

    if log_space:
        def func(x, y):
            return np.exp(-func_log(x,y))
        return func
    else:
        return func_log
    

def main():
    func = get_HC_survival_function(HC_null_sim_file=None)
    print("Pr[HC >= 3 |n=50] = ", func(50, 3)[0][0]) # 9.680113e-05
    print("Pr[HC >= 3 |n=100] = ", func(100, 3)[0][0]) # 0.0002335
    print("Pr[HC >= 3 |n=200] = ", func(200, 3)[0][0]) # 0.00103771
    

if __name__ == '__main__':
    main()
