# Evalaute results using pre-computed logloss per sentence
# HERE!!! Complete this part and run it


import torch
import pandas as pd
import logging
import numpy as np
import argparse
from src.DetectLM import DetectLM
from src.fit_survival_function import fit_per_length_survival_function
from src.HC_survival_function import get_HC_survival_function
import pathlib
import yaml
import re
import os
import json

logging.basicConfig(level=logging.INFO)


def get_survival_function(df, G=101):
    """
    Returns a survival function for every sentence length in tokens.

    Args:
    :df:  data frame with columns 'response' and 'length'
    :G:   number of interpolation points
    
    Return:
        bivariate function (length, responce) -> (0,1)

    """
    assert not df.empty
    value_name = "response" if "response" in df.columns else "logloss"

    df1 = df[~df[value_name].isna()]
    ll = df1['length']
    xx1 = df1[value_name]
    return fit_per_length_survival_function(ll, xx1, log_space=True, G=G)


def main():
    parser = argparse.ArgumentParser(description="Read responses from data and null files and run process. ")
    parser.add_argument('-i', type=str, help='input regex')
    parser.add_argument('-null', type=str, help='null file')
    parser.add_argument('-report-file', type=str, help='where to write results', default="report.csv")
    parser.add_argument('-conf', type=str, help='configurations file', default="conf.yml")
    
    args = parser.parse_args()

    with open(args.conf, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    
    report_filename = args.report_file

    HC_pval_func = get_HC_survival_function(HC_null_sim_file="HC_null_sim_results.csv")

    null_data_file = args.null
    df_null0 = pd.read_csv(null_data_file)

    input_file = args.i 

    # read json file:
    # use a json_loader to read input_file:

 
    with open(input_file, "r") as f:
        data_raw = json.load(f)

    try:
        # read json lines:
        data = {}
        with open('file') as f:
            for line in f:
                dl = json.loads(line)
                data[list(dl.keys())[0]] = list(dl.values())[0]
    except:
        logging.error("Could not read json as lines. Trying a different approach...")
        data = {}
        for k in data_raw:
            data[list(k.keys())[0]] = list(k.values())[0]
    

    
    lm_name = data[list(data.keys())[0]]['model']
    
    lo_precomputed_files = list(data.keys())

    print("Iterating over the files: ", lo_precomputed_files)

    results = {}
    for fn in lo_precomputed_files:
        per_file_results = {}
        logging.info(f"Parsing document {fn}...")

        # Creating null reponse after removing responses assocaited with the current file
        name = os.path.basename(fn)
        search_name = re.findall(r"([A-Za-z ]+)(?:mix| mix| edited.+|_edited.+|)?(?:.txt|.csv)?", name)
        if len(search_name) > 0:
            curr_name = search_name[0]
        else:
            logging.error(f"Could not extract name from {fn}")
        
        df_null = df_null0[df_null0['name'] != curr_name]
        logging.info(f"Removed {len(df_null0) - len(df_null)} entries from null data")
        if params['ignore-first-sentence']:
            df_null = df_null[df_null.num > 1]
        logging.info(f"Fitting LPPT P-value function...")
        pval_functions = get_survival_function(df_null, G=params['number-of-interpolation-points'])

        logging.basicConfig(level=logging.DEBUG)
        logging.debug("Parsing document...")
        
        max_tokens_per_sentence = params['max-tokens-per-sentence']
        min_tokens_per_sentence = params['min-tokens-per-sentence']

        detector = DetectLM(None, pval_functions,
                            min_len=min_tokens_per_sentence,
                            max_len=max_tokens_per_sentence,
                            length_limit_policy='truncate',
                            HC_type=params['hc-type'],
                            ignore_first_sentence=params['ignore-first-sentence']
                            )
        
        # read sentence info from pre-computed data
        responses = [r['response'] for r in data[fn]['sentences'] ]
        lengths = [len(r['sentence'].split()) for r in data[fn]['sentences'] ]
        org_sentences = [r['sentence'] for r in data[fn]['sentences'] ]
        org_tags = [r['tag'] for r in data[fn]['sentences'] ]

        res = detector.from_responses(responses, lengths)

        df = res['sentences']
        df['tag'] = org_tags

        print(df.groupby('tag').response.mean())
        print(df[df['mask']])
        len_valid = len(df[~df.pvalue.isna()])

        print("Length valid: ", len_valid)
        edit_rate = np.mean(df['tag'] == '<edit>')
        print(f"Num of Edits (rate) = {np.sum(df['tag'] == '<edit>')} ({edit_rate})")
        HC = res['HC']
        print(f"HC = {res['HC']}")
        HC_pvalue = HC_pval_func(len_valid, HC)[0][0]
        print(f"Pvalue (HC) = {HC_pvalue}")
        bonf = res['bonf']

        print(f"Bonferroni's P-value = {bonf}")
        print(f"Fisher = {res['fisher']}")
        print(f"Fisher (chisquared pvalue) = {res['fisher_pvalue']}")
        dfr = df[df['mask']]
        precision = np.mean(dfr['tag'] == '<edit>')
        recall = np.sum((df['mask'] == True) & (df['tag'] == '<edit>')) / np.sum(df['tag'] == '<edit>')
        print("Precision = ", precision)
        print("recall = ", recall)
        print("F1 = ", 2 * precision*recall / (precision + recall))

        per_file_results['metrics'] = dict(length=len_valid, edit_rate=edit_rate, HC=res['HC'], 
                                HC_pvalue=HC_pvalue, precision=precision, recall=recall, bonf=bonf, filename=fn)
        per_file_results['null-data'] = dict(filename = null_data_file, length=len(df_null))
        per_file_results['model'] = lm_name
        per_file_results['sentences'] = df.to_dict(orient='records')
        
        #plt.title("Hisogram of P-values")
        #plt.savefig("pvalue_hist.png")
        #plt.show()
        results[fn] = per_file_results['metrics']


    print(results)
    print(f"Saving report to {report_filename}")
    dfr = pd.DataFrame.from_dict(results).T
    dfr.to_csv(report_filename)


if __name__ == '__main__':
    main()