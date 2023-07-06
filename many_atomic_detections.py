"""
Apply the atomic chunk detector many times.
This is useful for:
 1. Characterizing the null distribution of a model with a specific context policy.
 2. Characterizing the power of the global detector against a mixtures from a specific domain.

 Note:
 The default output folder is "./results", hence make sure that such folder exists before running the script

"""
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import argparse
from src.PerplexityEvaluator import PerplexityEvaluator
from src.PrepareSentenceContext import PrepareSentenceContext
from src.dataset_loaders import (get_text_from_chatgpt_news_dataset,
                                 get_text_from_wiki_dataset,
                                 get_text_from_wiki_long_dataset,
                                 get_text_from_chatgpt_news_long_dataset,
                                 get_text_from_chatgpt_abstracts_dataset)
from glob import glob

logging.basicConfig(level=logging.INFO)


def process_text(text, atomic_detector, parser):
    chunks = parser(text)

    ids = []
    lengths = []
    responses = []
    context_lengths = []
    chunk_num = 0
    for chunk, context, length in zip(chunks['text'], chunks['context'], chunks['length']):
        chunk_num += 1
        res = atomic_detector(chunk, context)
        ids.append(chunk_num)
        lengths.append(length)
        responses.append(res)
        if context:
            context_lengths.append(len(context.split()))
        else:
            context_lengths.append(0)

    return dict(chunk_ids=ids, responses=responses, lengths=lengths, context_lengths=context_lengths)


def iterate_over_texts(dataset, atomic_detector, parser, output_file):
    ids = []
    lengths = []
    responses = []
    context_lengths = []
    names = []
    for d in tqdm(dataset):
        name = d['id']
        try:
            r = process_text(d['text'], atomic_detector, parser)
        except KeyboardInterrupt:
            break
        except:
            print(f"Error processing {name}")
            continue
        ids += r['chunk_ids']
        responses += r['responses']
        lengths += r['lengths']
        context_lengths += r['context_lengths']
        names += [name] * len(r['chunk_ids'])

        df = pd.DataFrame({'num': ids, 'length': lengths, 
                           'response': responses, 'context_length': context_lengths,
                           'name': names})
        logging.info(f"Saving results to {output_file}")
        df.to_csv(output_file)


def get_text_data_from_files(path, extension='*.txt'):
    logging.info(f"Reading text data from {path}...")
    lo_fns = glob(path + extension)
    for fn in lo_fns:
        logging.info(f"Reading text from {fn}")
        with open(fn, "rt") as f:
            yield dict(id=fn, text=f.read())


def main():
    parser = argparse.ArgumentParser(description='Apply atomic detector many times to characterize distribution')
    parser.add_argument('-i', type=str, help='database name or file', default="")
    parser.add_argument('-o', type=str, help='output folder', default="./results")
    parser.add_argument('-model-name', type=str, default='gpt2')
    parser.add_argument('--context', action='store_true')
    parser.add_argument('--human', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--describe-datasets', action='store_true')

    args = parser.parse_args()

    lo_data_loaders = {'wiki': get_text_from_wiki_dataset,
                       'wiki-long': get_text_from_wiki_long_dataset,
                       'news': get_text_from_chatgpt_news_dataset,
                       'news-long': get_text_from_chatgpt_news_long_dataset
                       }
    if args.describe_datasets:
        for k in lo_data_loaders:
            for author in ['machine', 'human']:
                print(f"Dataset {k} with author {author}:")
                ds = lo_data_loaders[k](text_field=f'{author}_text')
                print(f"\tSize = {ds.dataset_size}")
                print(f"\tNum rows = {ds.num_rows}")
                print(f"\tFeatures = {ds.features}")
        exit(1)

    lm_name = args.model_name

    if args.context:
        context_policy = 'previous_sentence'
    else:
        context_policy = 'no_context'

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

    dataset_name = args.i
    shuffle = args.shuffle

    author = 'human' if args.human else 'machine'

    if args.i == "wiki":
        logging.info("Processing wiki dataset...")
        ds = get_text_from_wiki_dataset(text_field=f'{author}_text', shuffle=shuffle)
    elif args.i == "wiki-long":
        logging.info("Processing wiki-long dataset...")
        ds = get_text_from_wiki_long_dataset(text_field=f'{author}_text', shuffle=shuffle)
    elif args.i == 'news':
        logging.info("Processing news dataset...")
        ds = get_text_from_chatgpt_news_dataset(text_field=f'{author}_text', shuffle=shuffle)
    elif args.i == 'news-long':
        logging.info("Processing news-long dataset...")
        ds = get_text_from_chatgpt_news_long_dataset(text_field=f'{author}_text', shuffle=shuffle)
    elif args.i == 'abstracts':
        logging.info("Processing reserch-abstracts dataset...")
        ds = get_text_from_chatgpt_abstracts_dataset(text_field=f'{author}_text', shuffle=shuffle)
    else:
        ds = get_text_data_from_files(args.i, extension='*.txt')
        dataset_name = 'files'

    if "/" in lm_name:
        lm_name_str = lm_name.split("/")[-1]
    else:
        lm_name_str = lm_name
    out_filename = f"{args.o}/{lm_name_str}_{context_policy}_{dataset_name}_{author}.csv"
    logging.info(f"Iterating over texts...")
    sentence_detector = PerplexityEvaluator(model, tokenizer)
    parser = PrepareSentenceContext(context_policy=context_policy)

    print(f"Saving results to {out_filename}")
    iterate_over_texts(ds, sentence_detector, parser, output_file=out_filename)


if __name__ == '__main__':
    main()
