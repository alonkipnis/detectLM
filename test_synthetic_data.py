import numpy as np
from tqdm import tqdm
from multitest import MultiTest
import pandas as pd

from src.DetectLM import DetectLM
from src.fit_survival_function import fit_per_length_survival_function
from src.fit_HC_survival_function import get_HC_survival_function

model_name = "gpt2-xl"


params = {}
params['ignore-first-sentence'] = True
params['language-model-name'] = model_name
params['number-of-interpolation-points'] = 47
params['max-tokens-per-sentence'] = 50
params['min-tokens-per-sentence'] = 10
params['hc-type'] = "stbl"


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

    HC_survival_func = get_HC_survival_function("HC_null_sim_results.csv")

    output_report_filename = "results/synthetic_data_report_al_001_news.csv"
    params['sig-level'] = 0.01

    #dataset_name = 'wiki-long'
    #min_length = 200
    #eps = 0.1
    
    #dataset_name = 'wiki-long'
    #min_length = 200
    #eps = 0.1

    nMonte = 10

    report = []
    for _ in range(nMonte):
        for dataset_name in ['news']:
            ds_machine = pd.read_csv(f"results/{model_name}_no_context_{dataset_name}_machine.csv")
            ds_null = ds_machine.sample(frac=0.5)
            ds_machine = ds_machine.drop(index=ds_null.index)

            # based on null data for every dataset
            #pval_functions = get_survival_function(ds_null[ds_null.num > 1], G=params['number-of-interpolation-points'])
            pval_functions = get_survival_function(ds_null, G=params['number-of-interpolation-points'])

            ds_human = pd.read_csv(f"results/{model_name}_no_context_{dataset_name}_human.csv")
            ds_human['human'] = True
            ds_machine['human'] = False
            # "name" is the article ID:
            joint_names = ds_machine.merge(ds_human, on='name', how='inner')['name'].unique().tolist()
            print(f"Total number of shared articles {len(joint_names)}")

            ds_machine = ds_machine[ds_machine['name'].isin(joint_names)]        
            ds_human = ds_human[ds_human['name'].isin(joint_names)]
            
            lengths_machine = ds_machine.groupby('name')['num'].count().reset_index().rename(columns={'num':'machine_doc_length'})
            lengths_human = ds_human.groupby('name')['num'].count().reset_index().rename(columns={'num':'human_doc_length'})

            ds_pool =  ds_human.merge(lengths_machine, on='name', how='inner').merge(lengths_human, on='name', how='inner')
            min_length = 5
            ds_pool = ds_pool[(ds_pool['human_doc_length'] >= min_length) &  (ds_pool['machine_doc_length'] >= min_length)]
            print(f"Size of sentences to sample from is {len(ds_pool)}")

            for eps in [0, .1, .2]: # [0, 0.14, 0.35]:
                for min_length in [50, 100, 200]:
                    
                    #ds_sample = ds_human.groupby("name").sample(frac=eps)
                    ds_sample = pd.DataFrame()
                    for c in ds_pool.groupby('name'): # "name" is the article ID
                        k = int(np.ceil(c[1]['machine_doc_length'].values[0] * eps))
                        ds_sample = pd.concat([ds_sample, c[1].sample(n=k)])

                    ds_mixed = pd.concat([ds_machine, ds_sample])
                    ds_mixed_grouped = group_articles_to_minimum_length(ds_mixed, min_length)


                    detectlm = DetectLM(lambda x: 0, # we use pre-computed logperplexities
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
                        mix_rate = np.mean(c[1]['human'])
                        if len(responses) > min_no_sentences:
                            pvals, _ = detectlm._get_pvals(responses, lengths)
                            pvals = np.vstack(pvals).squeeze()
                            mt = MultiTest(pvals, stbl=stbl)  # HC test
                            hc = mt.hc()[0]                   # HC test
                            results.append(dict(id=c[0], HC=hc, len=len(responses), mix_rate=mix_rate))

                    #t0 = crit_vals[(crit_vals.n == min_length) & (crit_vals.alpha == params['sig-level'])].q_alpha.values[0]
                    HC_pvals = np.vstack([HC_survival_func(c['len'], c['HC']) for c in results])[:,0]
                    acc = np.mean(HC_pvals <= params['sig-level'])

                    avg_len = np.mean(np.vstack([c['len'] for c in results])).squeeze()
                    
                    avg_mix_rate = np.mean([c['mix_rate'] for c in results])
                    print("Avg. Mixing rate = ", avg_mix_rate)
                    report.append(dict(model=model_name, dataset=dataset_name, epsilon=eps, mix_rate=avg_mix_rate,
                                min_length=min_length, 
                                avg_length = avg_len,
                                detection_rate=acc))
                    print(f"Model={model_name}, dataset={dataset_name}, epsilon={eps}, length={min_length} --> detection rate {acc}")

                    pd.DataFrame(report).to_csv(output_report_filename)


if __name__ == '__main__':
    main()