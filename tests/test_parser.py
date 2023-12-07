import pandas as pd
import logging
import numpy as np
import argparse
import sys
sys.path.append('../')
from src.PrepareSentenceContext import PrepareSentenceContext
import pathlib
import yaml
import re

logging.basicConfig(level=logging.INFO)


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
        context_policy = 'previous_sentence'
    else:
        context_policy = None

    input_file = args.i
    if pathlib.Path(input_file).suffix == '.txt':
        engine = 'spacy'
        with open(input_file, 'rt') as f:
            text = f.read()
    else:
        logging.error("Unknown file extension")
        exit(1)

    parser = PrepareSentenceContext(sentence_parser=engine, context_policy=context_policy)
    chunks = parser(text)
    df = pd.DataFrame(dict(sentence=chunks['text'],
                           context=chunks['context'],
                           tag=chunks['tag']))

    df.loc[df.tag.isna(), 'tag'] = 'not edit'

    print(f"Edit rate = {np.mean(df['tag'] == '<edit>')}")
    print(df[df['tag'] != 'not edit'])



if __name__ == '__main__':
    main()
