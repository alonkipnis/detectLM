import numpy as np
from tqdm import tqdm
from multitest import MultiTest
import pandas as pd

from src.DetectLM import DetectLM
from src.fit_survival_function import fit_per_length_survival_function


model_name = "gpt2-xl"


params = {}
params['ignore-first-sentence'] = True
params['language-model-name'] = model_name
params['number-of-interpolation-points'] = 47
params['max-tokens-per-sentence'] = 50
params['min-tokens-per-sentence'] = 8
params['hc-type'] = "stbl"
params['sig-level'] = 0.05


crit_vals = pd.read_csv("HC_critvals.csv")



def get_null_data(params):
    df_null = pd.read_csv(params['null-data-file'])
    if params['ignore-first-sentence']: 
        df_null = df_null[df_null.num > 1]
    return df_null

def get_survival_function(df, G=101):
    """
    One survival function for every sentence length in tokens

    Args:
    :df:  data frame with columns 'response' and 'length'

    Return:
        bivariate function (length, responce) -> (0,1)

    """
    assert not df.empty
    value_name = "response" if "response" in df.columns else "logloss"

    df1 = df[~df[value_name].isna()]
    ll = df1['length']
    xx1 = df1[value_name]
    return fit_per_length_survival_function(ll, xx1, log_space=True, G=G)

       
def group_articles_to_minimum_length(df, min_length):
    """
    Rearrange group names so that every group has at least
    :min_length: elements
    """
    df_grouped = df.copy()
    df_grouped.loc[:, 'new_name'] = df['name'].copy()
    groups = list(df.groupby('name'))
    while len(groups) > 0:
        c = groups.pop(0)
        acc = len(c[1])
        while (acc <= min_length) and len(groups)>0:
            c1 = groups.pop(0)
            acc += len(c1[1])
            df_grouped.loc[df['name'] == c1[0], 'new_name'] = c[0]

    return df_grouped

def main():

    #dataset_name = 'wiki-long'
    #min_length = 200
    #eps = 0.1
    for dataset_name in ['wiki-long', 'news']:
        ds_machine = pd.read_csv(f"results/{model_name}_no_context_{dataset_name}_machine.csv")
        ds_human = pd.read_csv(f"results/{model_name}_no_context_{dataset_name}_human.csv")
        ds_human['human'] = True
        ds_machine['human'] = False
        joint_names = ds_machine.merge(ds_human, on='name', how='inner')['name'].tolist()
        ds_machine = ds_machine[ds_machine['name'].isin(joint_names)]
        
        ds_human = ds_human[ds_human['name'].isin(joint_names)]
        print(f"Total number of shared articles {len(joint_names)}")
        for eps in [0.1, 0.2]:
            for min_length in [100, 200]:
                params['null-data-file'] = f"results/{model_name}_no_context_{dataset_name}_machine.csv"

                df_null = get_null_data(params)
                pval_functions = get_survival_function(df_null, G=params['number-of-interpolation-points'])

                #ds_pool = ds_human[ds_human['name'].isin(joint_names)]
                #ds_sample = ds_human.groupby("name").sample(frac=eps)
                #ds_mixed = pd.concat([ds_machine[ds_machine['name'].isin(joint_names)], ds_sample])

                ds_sample = ds_human.groupby("name").sample(frac=eps)

                ds_mixed = pd.concat([ds_machine, ds_sample])
                print("Mixing rate (appx) = ", ds_mixed.groupby('name')['human'].mean().mean())

                ds_mixed_grouped = group_articles_to_minimum_length(ds_mixed, min_length)


                detectlm = DetectLM(lambda x: 0,
                                    pval_functions,
                                    min_len=params['min-tokens-per-sentence'],
                                    max_len=params['max-tokens-per-sentence'],
                                    HC_type=params['hc-type'],
                                    ignore_first_sentence=params['ignore-first-sentence']
                                    )
                stbl = True if params['hc-type']=='stbl' else False

                min_no_sentences = 10

                results = []
                for c in tqdm(ds_mixed_grouped.groupby('new_name')):
                    responses = c[1]['response']
                    lengths = c[1]['length']
                    if len(responses) > min_no_sentences:
                        pvals, _ = detectlm._get_pvals(responses, lengths)
                        pvals = np.vstack(pvals).squeeze()
                        mt = MultiTest(pvals, stbl=stbl)
                        hc = mt.hc()[0]
                        results.append(dict(id=c[0], HC=hc))

                t0 = crit_vals[(crit_vals.n == min_length) & (crit_vals.alpha == params['sig-level'])].q_alpha.values[0]
                acc = np.mean(pd.DataFrame.from_dict(results)['HC'] > t0)

                print(f"Model={model_name}, dataset={dataset_name}, epsilon={eps}, length={min_length} --> detection rate {acc}")


if __name__ == '__main__':
    main()