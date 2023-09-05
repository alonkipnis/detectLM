import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import numpy as np
import argparse
import pathlib
from src.DetectLM import DetectLM
from src.PerplexityEvaluator import PerplexityEvaluator
from src.PrepareSentenceContext import PrepareSentenceContext
from glob import glob
from src.fit_survival_function import fit_per_length_survival_function
import yaml
import re
import json
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def read_all_csv_files(pattern):
    df = pd.DataFrame()
    print(pattern)
    for f in glob(pattern):
        df = pd.concat([df, pd.read_csv(f)])
    return df


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

    ll = df['length']
    xx1 = df[value_name]
    return fit_per_length_survival_function(ll, xx1, log_space=True, G=G)


def mark_edits_remove_tags(chunks, tag="edit"):
    text_chunks = chunks['text']
    edits = []
    for i, text in enumerate(text_chunks):
        chunk_text = re.findall(rf"<{tag}>(.+)</{tag}>", text)
        if len(chunk_text) > 0:
            import pdb;
            pdb.set_trace()
            chunks['text'][i] = chunk_text[0]
            chunks['length'][i] -= 2
            edits.append(True)
        else:
            edits.append(False)

    return chunks, edits


def main():
    parser = argparse.ArgumentParser(description='Test document for non-model sentences')
    parser.add_argument('-i', type=str, help='input bacth file')
    parser.add_argument('-o', type=str, help='output file', default="")
    parser.add_argument('-conf', type=str, help='configurations file', default="conf.yml")
    parser.add_argument('--context', action='store_true')
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

    logging.info(f"Using null data from {null_data_file} and fitting survival function")
    logging.info(f"Please verify that null data was obtained with the same context policy used in inference.")

    input_file = args.i

    if not (input_file in null_data_file):  # checking for name match
        print(f"Warning: null data file {null_data_file} may not match data file {input_file}.")
        print("Continue anyway? (Y/N)")
        a = input()
        if a.lower() == 'n':
            exit(1)

    df_null = read_all_csv_files(null_data_file)

    max_tokens_per_sentence = params['max-tokens-per-sentence']
    min_tokens_per_sentence = params['min-tokens-per-sentence']

    if params['ignore-first-sentence']:
        df_null = df_null[df_null.num > 1]
        logging.info(f"Null data contains {len(df_null)} records")

    pval_functions = get_survival_function(df_null, G=params['number-of-interpolation-points'])


    logging.info(f"Loading model and detection function...")

    logging.debug(f"Loading Language model {lm_name}...")
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    model = AutoModelForCausalLM.from_pretrained(lm_name)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)

    if args.context:
        context_policy = 'previous_sentence'
    else:
        context_policy = None

    if params['ignore-first-sentence'] or context_policy == 'previous_sentence':
        ignore_first_sentence = True
        logging.warning("Ignoring the first sentence")
    else:
        ignore_first_sentence = False

    sentence_detector = PerplexityEvaluator(model, tokenizer)
    logging.debug("Initializing detector...")
    detector = DetectLM(sentence_detector, pval_functions,
                        min_len=min_tokens_per_sentence,
                        max_len=max_tokens_per_sentence,
                        HC_type=params['hc-type'],
                        length_limit_policy='truncate',
                        ignore_first_sentence=ignore_first_sentence
                        )


    output_file = args.o
    if output_file == "":
        output_file = f"results/results_{pathlib.Path(input_file).stem}.json"
    logging.info("Making sure output file is valid")
    try:
        with open(output_file, "w") as f:
            pass
    except:
        logging.error(f"Could not open {output_file} for writing.")
        exit(1)

    logging.info(f"Reading the content of {input_file}...")
    with open(input_file, "r") as f:
        dataset = json.load(f)
    logging.info(f"Loaded a dataset of {len(dataset)} documents")

    parser = PrepareSentenceContext(engine='spacy', context_policy=context_policy)
    hc_scores = []
    fisher_pval_scores = []
    results = {}

    for k in tqdm(dataset):
        entry = dataset[k]
        text = entry['text']

        chunks = parser(text)

        logging.info("Testing parsed document")

        res = detector(chunks['text'], chunks['context'])
        df = res['sentences']

        if np.isnan(res['HC']):
            print(f"Could not process {k}")
            continue
        df['tag'] = chunks['tag']
        df.loc[df.tag.isna(), 'tag'] = 'not edit'

        empirical_edit_rate = np.mean(df['tag'] == '<edit>')
        theoretical_edit_rate = entry['eps']
        no_D = int(df['mask'].sum())
        TE = int(np.sum(df['tag'] == '<edit>'))
        TD = int(np.sum(df[df['mask']]['tag'] == '<edit>'))
        FD = int(np.sum(df[df['mask']]['tag'] == 'not edit'))
        FDP = FD / no_D
        TDP = TD / no_D
        if TE > 0:
            recall = TD / TE
        else:
            recall = np.nan
        hc = res['HC']
        fisher_pval = res['fisher_pvalue']

        results[k] = dict(
            no_discoveries=no_D, empirical_edit_rate=empirical_edit_rate,
            theoretical_edit_rate=theoretical_edit_rate,
            true_discoveries=TD, false_discoveries=FD, recall=recall,
            FDP=FDP, TDP=TDP, hc=hc, fisher_pval=fisher_pval)

        hc_scores.append(hc)
        fisher_pval_scores.append(fisher_pval)

        try:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=True)
        except:
            import pdb;
            pdb.set_trace()

        print("Discoveries (HC): ", np.mean(np.array(hc_scores) > 1.9))
        print("Discoveries (Fisher): ", np.mean(np.array(fisher_pval_scores) < 0.01))

if __name__ == '__main__':
    main()
