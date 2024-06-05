# iterate over many txt files

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import numpy as np
import argparse
from src.DetectLM import DetectLM
from src.PerplexityEvaluator import PerplexityEvaluator
from src.PrepareSentenceContext import PrepareSentenceContext
from src.fit_survival_function import fit_per_length_survival_function
from src.HC_survival_function import get_HC_survival_function
from glob import glob
import pathlib
import yaml
from pathlib import Path
import re
import os
import json


logging.basicConfig(level=logging.INFO)


def read_all_csv_files(pattern):
    df = pd.DataFrame()
    print(pattern)
    for f in glob(pattern):
        df = pd.concat([df, pd.read_csv(f)])
    return df


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


def mark_edits_remove_tags(chunks, tag="edit"):
    text_chunks = chunks['text']
    edits = []
    for i,text in enumerate(text_chunks):
        chunk_text = re.findall(rf"<{tag}>(.+)</{tag}>", text)
        if len(chunk_text) > 0:
            import pdb; pdb.set_trace()
            chunks['text'][i] = chunk_text[0]
            chunks['length'][i] -= 2
            edits.append(True)
        else:
            edits.append(False)

    return chunks, edits

def main():
    parser = argparse.ArgumentParser(description="Apply detector of non-GLM text to a text file or several text files (based on an input pattern)")
    parser.add_argument('-i', type=str, help='input regex', default="Data/ChatGPT/*.txt")
    parser.add_argument('-o', type=str, help='where to store per-sentence information', default="results/evaluations.json") 
    parser.add_argument('-report-file', type=str, help='where to write results', default="report.csv")

    parser.add_argument('-conf', type=str, help='configurations file', default="conf.yml")
    parser.add_argument('--context', action='store_true')
    parser.add_argument('--dashboard', action='store_true')
    parser.add_argument('--leave-out', action='store_true')
    
    args = parser.parse_args()

    with open(args.conf, "r") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    print("context = ", args.context)

    if args.context:
        null_data_file = params['context-null-data-file']
    else:
        null_data_file = params['no-context-null-data-file']
    lm_name = params['language-model-name']

    if not args.leave_out:
        logging.info(f"Fitting null log-loss survival function using data from {null_data_file}.")
        logging.info(f"Please verify that the data was obtained under the same context policy.")
        df_null = read_all_csv_files(null_data_file)
        if params['ignore-first-sentence']:
            df_null = df_null[df_null.num > 1]
        logging.info(f"Found {len(df_null)} log-loss values of text atoms in {null_data_file}.")
        pval_functions = get_survival_function(df_null, G=params['number-of-interpolation-points'])

    max_tokens_per_sentence = params['max-tokens-per-sentence']
    min_tokens_per_sentence = params['min-tokens-per-sentence']


    def init_detector(lm_name):
        if lm_name == `'noModel'`:
            logging.info(f"Loading DetectLM without a language model...")
            return DetectLM(None, pval_functions,
                            min_len=min_tokens_per_sentence,
                            max_len=max_tokens_per_sentence,
                            length_limit_policy='truncate',
                            HC_type=params['hc-type'],
                            ignore_first_sentence=params['ignore-first-sentence']
                            )

        logging.info(f"Loading Language model {lm_name}...")
        tokenizer = AutoTokenizer.from_pretrained(lm_name)
        model = AutoModelForCausalLM.from_pretrained(lm_name)
        
        if torch.backends.mps.is_available():
            device = 'mps'
            print("Using mps")
        elif torch.cuda.is_available():
            device = 'cuda'
            print("Using cuda")
        else:
            device = 'cpu'
            print("Using cuda")
        model.to(device)

        logging.info(f"Loading LPPT evaluator...")
        sentence_detector = PerplexityEvaluator(model, tokenizer)

        logging.debug("Initializing detector...")
        return DetectLM(sentence_detector, pval_functions,
                            min_len=min_tokens_per_sentence,
                            max_len=max_tokens_per_sentence,
                            length_limit_policy='truncate',
                            HC_type=params['hc-type'],
                            ignore_first_sentence=params['ignore-first-sentence']
                            )


    if args.context:
        context_policy = 'previous_sentence'
    else:
        context_policy = None

        
    HC_pval_func = get_HC_survival_function(HC_null_sim_file="HC_null_sim_results.csv")

    pattern = args.i
    
    output_file = args.o

    # check if the output file exists. If yes, append a number to the name:
    while os.path.exists(output_file):
        output_file = output_file.replace(".json", "_1.json")


    
    parser = PrepareSentenceContext(sentence_parser=params['parser'],
                                     context_policy=context_policy)

    detector = None
    lo_fns = glob(pattern)
    print("Iterating over the files: ", lo_fns)

    results = {}
    for text_file in lo_fns:
        per_file_results = {}
        logging.info(f"Parsing document {text_file}...")

        if args.leave_out:
            # Creating null reponse after removing responses assocaited with the current file
            logging.info(f"Reading null data from {null_data_file}...")
            df_null0 = read_all_csv_files(null_data_file)
            logging.info(f"Removing null entries associated with {text_file}...")
            df_null0.loc[:, 'title'] = df_null0['name'].str.extract(r"([A-Za-z \(\)]+)(?:mix| mix| edited.+|_edited.+|)?.txt")
            name = os.path.basename(text_file)
            search_name = re.findall(r"([A-Za-z ]+)(?:mix| mix| edited.+|_edited.+|)?(?:.txt|.csv)?", name)
            if len(search_name) > 0:
                curr_name = search_name[0]
            else:
                logging.error(f"Could not extract name from {text_file}")
            df_null = df_null0[df_null0['title'] != curr_name]
            logging.info(f"Removed {len(df_null0) - len(df_null)} entries from null data")
            if params['ignore-first-sentence']:
                df_null = df_null[df_null.num > 1]
            logging.info(f"Fitting LPPT P-value function...")
            pval_functions = get_survival_function(df_null, G=params['number-of-interpolation-points'])

        logging.basicConfig(level=logging.DEBUG)
        logging.debug("Parsing document...")
        
        if pathlib.Path(text_file).suffix == '.txt':
            with open(text_file, 'rt') as f:
                text = f.read()
            
            chunks = parser(text)
            logging.info("Testing parsed document")

            if detector is None:
                logging.debug("Initializing detector...")
                detector = init_detector(lm_name)
    
            res = detector(chunks['text'], chunks['context'], dashboard=args.dashboard)
            logging.basicConfig(level=logging.INFO)

            df = res['sentences']
            df['tag'] = chunks['tag']
            df.loc[df.tag.isna(), 'tag'] = 'no edits'
            #df['filename'] = text_file

            name = Path(text_file).stem

            #output_folder = "results/" # Path(output_file).parent
            #output_file = f"{output_folder}{name}_sentences.csv"
            #print("Saving per-sentence data to ", output_file)
            #df.to_csv(output_file)

        elif pathlib.Path(text_file).suffix == '.csv':
            df = pd.read_csv(text_file)
            df.loc[:, 'length'] = df['sentence'].apply(lambda x: len(x.split()))  # approximate length

            if detector is None:
                detector = init_detector("noModel")

            res = detector.from_responses(df['response'], df['length'])
        else:
            logging.error("Unknown file extension")
            exit(1)

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
                                HC_pvalue=HC_pvalue, precision=precision, recall=recall, bonf=bonf, filename=text_file)
        per_file_results['null-data'] = dict(filename = null_data_file, length=len(df_null))
        per_file_results['model'] = lm_name
        per_file_results['sentences'] = df.to_dict(orient='records')
        # store dictionary per_file_results to json file output_file:
        with open(output_file, 'a') as f:
            f.write(',\n')
            f.write(json.dumps({text_file: per_file_results}, indent=4))
        logging.info(f"Saved results to {output_file}")

        #plt.title("Hisogram of P-values")
        #plt.savefig("pvalue_hist.png")
        #plt.show()
        results[text_file] = per_file_results['metrics']


    report_filename = args.report_file
    print(results)
    print(f"Saving report to {report_filename}")
    dfr = pd.DataFrame.from_dict(results).T
    dfr.to_csv(report_filename)


if __name__ == '__main__':
    main()