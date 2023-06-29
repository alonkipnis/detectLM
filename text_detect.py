import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import numpy as np
import argparse
from src.DetectLM import DetectLM
from src.PerplexityEvaluator import PerplexityEvaluator
from src.PrepareSentenceContext import PrepareSentenceContext
from fit_survival_function import fit_per_length_survival_function
from glob import glob
import pathlib
import yaml
import re


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
    parser = argparse.ArgumentParser(description='Test document for non-model sentences')
    parser.add_argument('-i', type=str, help='input text file', default="input file")
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
    logging.info(f"Please verify that null data was obtained with the same context")
    logging.info(f"policy used in inference.")

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

    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)

    if args.context:
        context_policy = 'previous_sentence'
    else:
        context_policy = None

    sentence_detector = PerplexityEvaluator(model, tokenizer)
    logging.debug("Initializing detector...")
    detector = DetectLM(sentence_detector, pval_functions,
                        min_len=min_tokens_per_sentence,
                        max_len=max_tokens_per_sentence,
                        length_limit_policy='truncate',
                        HC_type=params['hc-type'],
                        ignore_first_sentence=
                        True if context_policy == 'previous_sentence' else False
                        )

    input_file = args.i
    logging.info(f"Parsing document {input_file}...")

    if pathlib.Path(input_file).suffix == '.txt':
        engine = 'spacy'
        with open(input_file, 'rt') as f:
            text = f.read()
    else:
        logging.error("Unknown file extension")
        exit(1)

    parser = PrepareSentenceContext(engine=engine, context_policy=context_policy)
    chunks = parser(text)

    logging.info("Testing parsed document")
    res = detector(chunks['text'], chunks['context'])

    import pdb; pdb.set_trace()
    dashboard = detector.test_chunked_doc_hc_dashboard(chunks['text'], chunks['context'])

    df = res['sentences']

    df['tag'] = chunks['tag']
    df.loc[df.tag.isna(), 'tag'] = 'not edit'



    output_file = "out.csv"
    df.to_csv(output_file)
    df['pvalue'].hist

    print(df.groupby('tag').response.mean())
    print(df[df['mask']])
    len_valid = len(df[~df.pvalue.isna()])
    print("Length valid: ", len_valid)
    print(f"Num of Edits (rate) = {np.sum(df['tag'] == '<edit>')} ({np.mean(df['tag'] == '<edit>')})")
    print(f"HC = {res['HC']}")
    print(f"Fisher = {res['fisher']}")
    print(f"Fisher (chisquared pvalue) = {res['fisher_pvalue']}")
    dfr = df[df['mask']]
    precision = np.mean(dfr['tag'] == '<edit>')
    recall = np.sum((df['mask'] == True) & (df['tag'] == '<edit>')) / np.sum(df['tag'] == '<edit>')
    print("Precision = ", precision)
    print("recall = ", recall)
    print("F1 = ", 2 * precision*recall / (precision + recall))
    
    df['pvalue'].hist(bins=np.linspace(0,1,17))
    plt.title("Hisogram of P-values")
    plt.savefig("pvalue_hist.png")
    plt.show()



if __name__ == '__main__':
    main()