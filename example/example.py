
"""
This script is an example of how to use the DetectLM class.
"""
from transformers import AutoTokenizer, AutoModelForCausalLM    

import sys
sys.path.append('../')
from src.PerplexityEvaluator import PerplexityEvaluator
from src.DetectLM import DetectLM
from src.PrepareSentenceContext import PrepareSentenceContext
import pickle

INPUT_FILE = 'example_text.txt'
LOGLOSS_PVAL_FUNC_FILE = 'logloss_pval_function.pkl'

# Load the logloss p-value function. Ususally one must fit this function using triaining data
# from the null class and ``fit_survival_function``.
# Here we load a pre-fitted function for the GPT-2 language model under Wikipedia-Introduction 
# dataset and no context.
with open(LOGLOSS_PVAL_FUNC_FILE, 'rb') as f:
    pval_function = pickle.load(f)

# Initialize LoglossEvaluator with a language model and a tokenizer
lm_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(lm_name)

sentence_detector = PerplexityEvaluator(AutoModelForCausalLM.from_pretrained(lm_name),
                    AutoTokenizer.from_pretrained(lm_name))

# initialize the detector...
detector = DetectLM(sentence_detector, pval_function,
                    min_len=8, max_len=50, length_limit_policy='truncate')

# parse text from an input file 
with open(INPUT_FILE, 'rt') as f:
    text = f.read()
parse_chunks = PrepareSentenceContext(context_policy=None)
chunks = parse_chunks(text)

# Test document
res = detector(chunks['text'], chunks['context'])
print(res)

"""

'HC': 1.3396600337668725, 'fisher': 20.49921190930749, 'fisher_pvalue': 0.02486927492187124}
"""