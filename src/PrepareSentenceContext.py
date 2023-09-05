import logging
import spacy
import re
from src.SentenceParser import SentenceParser


class PrepareSentenceContext(object):
    """
    Parse text and extract length and context information

    This information is needed for evaluating log-perplexity of the text with respect to a language model
    and later on to test the likelihood that the sentence was sampled from the model with the relevant context.
    """

    def __init__(self, sentence_parser='spacy', context_policy=None, context=None):
        if sentence_parser == 'spacy':
            self.nlp = spacy.load("en_core_web_sm", disable=["tagger", "attribute_ruler", "lemmatizer", "ner"])
        if sentence_parser == 'regex':
            logging.warning("Regex-based parser is not good at breaking sentences like 'Dr. Stone', etc.")
            self.nlp = SentenceParser()
        self.sentence_parser_name = sentence_parser

        self.context_policy = context_policy
        self.context = context

    def __call__(self, text):
        return self.parse_sentences(text)

    def parse_sentences(self, text):
        texts = []
        contexts = []
        lengths = []
        tags = []
        num_in_par = []
        previous = None

        text = re.sub("(</?[a-zA-Z0-9 ]+>\.?)\s+", r"\1.\n", text)  # to make sure that tags are in separate sentences
        parsed = self.nlp(text)

        running_sent_num = 0
        curr_tag = None
        for i, sent in enumerate(parsed.sents):
            # Here we try to track HTML-like tags. There might be
            # some issues because spacy sentence parser has unexpected behavior when it comes to newlines
            all_tags = re.findall(r"(</?[a-zA-Z0-9 ]+>)", str(sent))
            if len(all_tags) > 1:
                    logging.error(f"More than one tag in sentence {i}: {all_tags}")
                    exit(1)
            if len(all_tags) == 1:
                tag = all_tags[0]
                if tag[:2] == '</': # a closing tag
                    if curr_tag is None:
                        logging.warning(f"Closing tag without an opening tag in sentence {i}: {sent}")
                    else:
                        curr_tag = None
                else:
                    if curr_tag is not None:
                        logging.warning(f"Opening tag without a closing tag in sentence {i}: {sent}")
                    else:
                        curr_tag = tag
            else:  # if text is not a tag
                sent_text = str(sent)
                sent_length = len(sent)

                texts.append(sent_text)
                running_sent_num += 1
                num_in_par.append(running_sent_num)
                tags.append(curr_tag)
                lengths.append(sent_length)

                if self.context is not None:
                    context = self.context
                elif self.context_policy is None:
                    context = None
                elif self.context_policy == 'previous_sentence':
                    context = previous
                    previous = sent_text
                else:
                    context = None

                contexts.append(context)
        return {'text': texts, 'length': lengths, 'context': contexts, 'tag': tags,
                'number_in_par': num_in_par}