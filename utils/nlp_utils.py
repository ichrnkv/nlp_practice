import collections
import numpy as np
import re

import torch
from torch.utils.data import Dataset


def tokenize_text_regex(text, regex, lower=True, min_token_size=1):
    """
    Tokenize text using regular expression
    :param text: text without prepossessing, str
    :param regex: regular expression, re.compile
    :param lower: lowercase text, True ao False
    :param min_token_size: min token size, int
    """

    if lower:
        text = text.lower()
    all_tokens = regex.findall(text)
    return [token for token in all_tokens if len(token) >= min_token_size]


def tokenize_corpus(corpus, tokenizer, **tokenizer_kwargs):
    """
    Tokenize corpus using regular expression
    :param corpus: text corpus
    :param tokenizer: regex based tokenizer
    :param tokenizer_kwargs: tokenizer params
    """

    return [tokenizer(text, **tokenizer_kwargs) for text in corpus]


def build_vocabulary(tokenized_texts, max_size=100000, max_doc_freq=0.8, min_count=5, pad_word=None):
    """
    Builds vocabulary for  tokens in text
    :param tokenized_texts: tokenized text, list of strings
    :param max_size: vocabulary size, int
    :param max_doc_freq: max frequency for token, float
    :param min_count: min count of token uses, int
    :param pad_word: pad token, str
    :return: Vocabulary, Frequencies
    """
    word_counts = collections.defaultdict(int)
    doc_n = 0

    # building a dictionary
    for text in tokenized_texts:
        doc_n += 1
        unique_tokens = set(text)
        for token in unique_tokens:
            word_counts[token] += 1

    # removing most popular and unpopular tokens
    word_counts = {word: count for word, count in word_counts.items()
                   if count >= min_count and count / doc_n <= max_doc_freq}

    # sort descending
    sorted_word_counts = sorted(word_counts.items(),
                                reverse=True,  # descending
                                key=lambda x: x[1])

    # adding pad word if necessary
    if pad_word is not None:
        sorted_word_counts = [(pad_word, 0)] + sorted_word_counts

    # removing unpopular
    if len(sorted_word_counts) > max_size:
        sorted_word_counts = sorted_word_counts[:max_size]

    # token to id
    word2id = {word: i for i, (word, count) in enumerate(sorted_word_counts)}

    # frequency
    word2freq = np.array([count / doc_n for token, count in sorted_word_counts], dtype='float32')

    return word2id, word2freq


def texts_to_token_ids(tokenized_texts, word2id):
    """
    Transforms text to token ids
    :param tokenized_texts: tokenized_text, str
    :param word2id: fitted vocabulary, dict
    """
    return [[word2id[token] for token in text if token in word2id]
            for text in tokenized_texts]


def ensure_length(text, out_len, pad_value):
    """
    Checks the length of the text
    :param text: text transformed to ids, list of ints
    :param out_len: required length, int
    :param pad_value: padding value, int
    """

    if len(text) < out_len:
        text = list(text) + [pad_value] * (out_len - len(text))
    else:
        text = text[:out_len]
    return text


class PaddedDataset(Dataset):
    def __init__(self, texts, targets, out_len=100, pad_value=0):
        self.texts = texts
        self.targets = targets
        self.out_len = out_len
        self.pad_value = pad_value

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        txt = self.texts[item]
        txt = ensure_length(txt, self.out_len, self.pad_value)
        txt = torch.tensor(txt, dtype=torch.long)

        target = torch.tensor(self.targets[item], dtype=torch.long)

        return txt, target


if __name__ == '__main__':
    import pandas as pd

    data = pd.read_csv('./../data/IMDB Dataset.csv')
    texts = data['review']
    targets = data['sentiment'].map({'positive': 0, 'negative': 1})

    token_re = re.compile(r'[\w\d]+')
    tokenized = tokenize_corpus(corpus=texts, tokenizer=tokenize_text_regex, regex=token_re)
    vocab, freq = build_vocabulary(tokenized, max_size=100000, max_doc_freq=0.8, min_count=5, pad_word='PAD')

    print('Vocabulary size is {}'.format(len(vocab)))

    text_token_ids = texts_to_token_ids(tokenized, vocab)
    dataset = PaddedDataset(text_token_ids, targets.values, out_len=100, pad_value=0)
    print(dataset[1])
