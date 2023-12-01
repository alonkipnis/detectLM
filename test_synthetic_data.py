import numpy as np
from tqdm import tqdm
from multitest import MultiTest
import pandas as pd
import logging

from src.DetectLM import DetectLM
from src.fit_survival_function import fit_per_length_survival_function
from src.HC_survival_function import get_HC_survival_function

model_name = "gpt2-xl"
REPLACEMENTS = True

params = {}
params['ignore-first-sentence'] = False
params['language-model-name'] = model_name
params['number-of-interpolation-points'] = 47
params['max-tokens-per-sentence'] = 50
params['min-tokens-per-sentence'] = 10
params['hc-type'] = "stbl"
params['sig-level'] = 0.02

logging.basicConfig(level=logging.INFO)

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
    Rearrange group names of the dataframe so that every group has at least min_length elements.
    
    Args:
    min_length   minimum number of elements in each group at the end of rearrangement
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
    HC_survival_func = get_HC_survival_function("HC_null_sim_results.csv") # get the HC survival function 
    output_HC_vals_filename = f"results/synthetic_data_HV_vals.csv"
    
    nMonte = 30
    output_report_filename = f"results/synthetic_data_report_al_0{int(params['sig-level']*100)}_nMonte_{nMonte}.csv"

    report = []
    all_results = []
    for itr in range(nMonte):
        for dataset_name in ['abstracts', 'wiki-long', 'news']:
            logging.info(f"Loading {dataset_name} dataset")
            ds_machine = pd.read_csv(f"results/{model_name}_no_context_{dataset_name}_machine.csv")
            logging.info(f"Loaded {len(ds_machine)} sentences from {dataset_name} dataset")
            
            #Split articles into training and test sets:
            names_null = ds_machine.name.drop_duplicates().sample(frac=.5)
            ds_null = ds_machine[ds_machine.name.isin(names_null)]

            ds_test = ds_machine.drop(index=ds_null.index)
            # if REPLACEMENTS:
            #     ds_machine = ds_machine.sample(frac=1, replace=True).reset_index(drop=True) # shuffle the dataset
            # else: 
            #     ds_machine = ds_machine.sample(frac=1, replace=False).reset_index(drop=True) # shuffle the dataset
            #ds_null = ds_machine[:len(ds_machine)//2] # this is the training set (for the null distribution)
    
            logging.info(f"Set aside {len(ds_null)} sentences to fit the null distribution.")
            logging.info(f"Testing over {len(ds_test)} sentences.")
            
            if params['ignore-first-sentence']:
                pval_functions = get_survival_function(ds_null[ds_null.num > 1], G=params['number-of-interpolation-points'])
            else:
                pval_functions = get_survival_function(ds_null, G=params['number-of-interpolation-points'])

            ds_human = pd.read_csv(f"results/{model_name}_no_context_{dataset_name}_human.csv")
            ds_human['human'] = True
            ds_test['human'] = False
            # "name" is the article ID:
            joint_names = ds_test.merge(ds_human, on='name', how='inner')['name'].unique().tolist()
            logging.info(f"Total number of shared articles {len(joint_names)}")

            ds_test = ds_test[ds_test['name'].isin(joint_names)]        
            ds_human = ds_human[ds_human['name'].isin(joint_names)]
            
            lengths_machine = ds_test.groupby('name')['num'].count().reset_index().rename(columns={'num':'machine_doc_length'})
            lengths_human = ds_human.groupby('name')['num'].count().reset_index().rename(columns={'num':'human_doc_length'})

            ds_pool =  ds_human.merge(lengths_machine, on='name', how='inner').merge(lengths_human, on='name', how='inner')
            min_length = 5
            ds_pool = ds_pool[(ds_pool['human_doc_length'] >= min_length) & (ds_pool['machine_doc_length'] >= min_length)]
            ds_test = ds_test[ds_test['name'].isin(ds_pool['name'].unique())]
            logging.info(f"Number of sentences to sample from is {len(ds_pool)}")
            grp = ds_pool.groupby('name') # "name" is the article ID
            logging.info(f"Number of articles considered {len(grp)}")

            for eps in [0, .1, .2]: # [0, 0.14, 0.35]:
                for min_length in [50, 100, 200]:
                    logging.info(f"Simulating mixed articles:  eps={eps}, min_length={min_length}")
                    ds_sample = pd.DataFrame()

                    for c in grp:
                        k = int(c[1]['machine_doc_length'].values[0] * eps / (1-eps) + np.random.rand())
                        if len(c[1]) < k:
                            continue
                        ds_sample = pd.concat([ds_sample, c[1].sample(n=k)])  # sample k sentences from the human data
                    
                    ds_mixed = pd.concat([ds_test, ds_sample]) # we insert k sentences from the human data at the end of the article
                    # If you want to insert at random locations, use
                    # ds_mixed = ds_mixed.sample(frac=1).reset_index(drop=True)
                    ds_mixed_grouped = group_articles_to_minimum_length(ds_mixed, min_length)
                    

                    detectlm = DetectLM(lambda x: 0, # we use pre-computed logperplexities
                                        pval_functions,
                                        min_len=params['min-tokens-per-sentence'],
                                        max_len=params['max-tokens-per-sentence'],
                                        HC_type=params['hc-type'],
                                        ignore_first_sentence=params['ignore-first-sentence']
                                        )
                    stbl = True if params['hc-type']=='stbl' else False

                    results = []
                    for c in tqdm(ds_mixed_grouped.groupby('new_name')): # go over each extended article
                        responses = c[1]['response']
                        lengths = c[1]['length']
                        if len(responses) < min_length:
                            continue
                        mix_rate = np.mean(c[1]['human'])
                        pvals, _ = detectlm._get_pvals(responses, lengths)
                        pvals = np.vstack(pvals).squeeze()
                        mt = MultiTest(pvals, stbl=stbl)  # HC test
                        hc = mt.hc()[0]                   # HC test
                        results.append(dict(id=c[0], HC=hc, len=len(responses), mix_rate=mix_rate,
                                                dataset_name=dataset_name, epsilon=eps, model=model_name,
                                            min_length = min_length, itr=itr))
                    all_results.extend(results)
                    pd.DataFrame(all_results).to_csv(output_HC_vals_filename)
                    HC_pvals = np.vstack([HC_survival_func(c['len'], c['HC']) for c in results])[:,0]
                    det_rate = np.mean(HC_pvals <= params['sig-level'])

                    aa = np.linspace(0, 1, 10000)
                    P1 = np.mean(np.expand_dims(HC_pvals, 1) <= np.expand_dims(aa, 0), 0)
                    accuracy = (P1 + (1-aa)).max() / 2
                    avg_len = np.mean(np.vstack([c['len'] for c in results])).squeeze()
                    
                    avg_mix_rate = np.mean([c['mix_rate'] for c in results])
                    print("Actual avg. mixing rate = ", avg_mix_rate)
                    report.append(dict(model=model_name, dataset=dataset_name, epsilon=eps, mix_rate=avg_mix_rate,
                                number_of_articles=len(results),
                                min_length=min_length, 
                                avg_length = avg_len,
                                detection_rate=det_rate,
                                accuracy=accuracy,
                                itr=itr))
                    print(f"Model={model_name}, dataset={dataset_name}, number of articles={len(results)}, epsilon={eps},"
                          f"length={min_length} --> detection rate {det_rate}, accuracy={accuracy}")

                    pd.DataFrame(report).to_csv(output_report_filename)


if __name__ == '__main__':
    main()