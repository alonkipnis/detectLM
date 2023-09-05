"""
Script to compute log perplexity of an input text
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import argparse
from src.PerplexityEvaluator import PerplexityEvaluator


logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='Evaluate log-perplexity of the text')
    parser.add_argument('-text', type=str, help='input text', default="Hello world")
    args = parser.parse_args()

    text = args.text

    logging.info(f"Loading model and detection function...")
    lm_name = "gpt2"
    logging.debug(f"Loading Language model {lm_name}...")
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    model = AutoModelForCausalLM.from_pretrained(lm_name)

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model.to(device)

    print("INPUT TEXT:")
    print(text)
    print("Length [words] = ", len(text.split()))
    sentence_detector = PerplexityEvaluator(model, tokenizer)
    res = sentence_detector(text)
    print(f"Log-perplexity wrt {lm_name} = ", res)

if __name__ == '__main__':
    main()