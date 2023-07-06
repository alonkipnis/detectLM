import numpy as np
import argparse
import json
from tqdm import tqdm
from src.dataset_loaders import *
import spacy
import logging
logging.basicConfig(level=logging.INFO)


nlp = spacy.load("en_core_web_sm")


def mix_lists(lst0: list, lst1: list, eps: float) -> (list, list):
    """
    Mix elements from lst0 and lst1 according to the mixture probability eps

    Params:
        :lst0   the base list
        :lst1   the mixture list
        :eps    the probability of replacing an element in lst0 by the corresponding element in lst1

    Returns:
        :mixed list
        :I  indices of replaced elements
    """
    n1 = len(lst1)
    n0 = len(lst0)
    n = n0

    ii = np.zeros(n)
    if eps == 0:
        return lst0, ii

    k = int(eps * n + 0.5)

    a = np.random.choice(range(n), k, replace=False)
    ii[a] = 1

    mix = [lst1[np.random.randint(n1)] if ii[i] else lst0[i] for i in range(n)]
    return mix, ii

def add_tag(text: str, tag: str) -> str:
    """
    Add an opening and closing tag
    Args:
        text:
        tag:

    Returns:
        tagged text
    """
    return f"<{tag}> {text} </{tag}>"


def mix_documents(text0, text1, eps):
    sents0 = list(nlp(text0).sents)
    sents1 = list(nlp(text1).sents)
    sents01, ii = mix_lists(sents0, sents1, eps=eps)
    return "\n".join([add_tag(str(s), tag='edit') if ii[i] else str(s) for i, s in enumerate(sents01)])

def main():
    parser = argparse.ArgumentParser(description='Generate mixed document dataset')
    parser.add_argument('-i', type=str, help='data source name', default="")
    parser.add_argument('-o', type=str, help='output file', default="")
    parser.add_argument('-eps', type=float, help='size', default=0.1)
    parser.add_argument('-n', type=int, help='size', default=-1)
    parser.add_argument('-no-sentences', type=int, help='minimum number of sentence', default=1)
    parser.add_argument('--random', action='store_true')
    args = parser.parse_args()

    eps = args.eps
    data_source = args.i
    rnd = args.random
    n = args.n
    min_no_sents = args.no_sentences

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
        print("Unknown data source")
        exit(1)

    if n == -1:
        n = len(list(dataset))

    ds_df = dataset.to_pandas()

    if rnd:
        ds_df = ds_df.sample(n=n, replace=False)

    def get_appx_no_sents(st: str):
        #return len(list(nlp(st).sents))
        return len(list(st.split('.')))

    if args.o == "":
        out_filename = f"mixed_{data_source}.json"
    else:
        out_filename = args.o
    print(f"Iterating over texts...")

    dsr = ds_df

    mixed_dataset = {}
    for r in tqdm(dsr.iterrows()):
        doc0 = r[1]['machine_text']
        doc1 = r[1]['human_text']
        appx_no_sents = get_appx_no_sents(doc0)
        if appx_no_sents >= min_no_sents:
            mixed_doc = mix_documents(doc0, doc1, eps=eps)
            mixed_dataset[r[0]] = {'text': mixed_doc, 'eps': eps, 'source': data_source, 'appx_no_sents': appx_no_sents}

            print(f"Saving results to {out_filename}")
            with open(out_filename, "w") as outfile:
                # write the dictionary to the file as a JSON object
                json.dump(mixed_dataset, outfile, indent=True)


if __name__ == '__main__':
    main()
