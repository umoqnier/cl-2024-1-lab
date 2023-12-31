# -*- coding: utf-8 -*-
"""7_Modelos_del_lenguaje.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1U4zHw46NFaKFY6fcatQSRCOI8QoKmUh2
"""

import re
import requests
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import numpy as np
from itertools import chain

def get_corpus(file):
    r = requests.get(file)
    response_list = r.text.split("\n")
    response_list = response_list[23:] # para deshacernos del inicio que no pertenece al corpus en sí
    return response_list

def preprocess_corpus(corpus: list[list[str]]) -> list:
    clean_corpus = []
    for sent in corpus:
      if sent != '\r':
        sent = sent.split(' ') # dividimos en palabras
        clean_corpus.append([word.replace('\r', '').lower() for word in sent if re.match("^(?![0-9]+$)[\w\s]+$", word)])

    clean_corpus = [lst for lst in clean_corpus if lst] # borramos listas vacías que eran las que sólo tenían "\r"
    return clean_corpus

def vocabulary_factory():
    """Function that create a vocabulary

    Default method when a key is not in the dictionary changed to be the
    current lenght of the dictionary to provide a unique index for each
    new key.

    Example:
    >> vocab['test']
    0
    >> vocab['other']
    1
    >> vocab['test']
    0
    """
    vocab = defaultdict()
    vocab.default_factory = lambda: len(vocab)
    return vocab

def word_to_index(corpus: list[list[str]], vocab: defaultdict) -> list[int]:
    """Function that maps each word in a corpus to a unique index"""
    for sent in corpus:
        yield [vocab[word] for word in sent]

def get_index_to_word(vocab: defaultdict) -> dict:
    """Map indices as keys and words as values from a vocabulary"""
    return {index: word for word, index in vocab.items()}

def get_n_grams(indexed_sents: list[list[str]], n=2) -> chain:
    return chain(*[zip(*[sent[i:] for i in range(n)]) for sent in indexed_sents])

def get_model(sents: list[list[str]], vocabulary: defaultdict, n: int=2, l: float=1.0) -> tuple:

    # Get n_grams
    n_grams = get_n_grams(sents, n)

    # Get n_grams frequencies
    freq_n_grams = Counter(n_grams)

    # Get vocabulary length (without BOS/EOS)
    N = len(vocabulary) - 2
    # Calculate tensor dimentions for transition probabilities
    # For columns (conditional word) we consider the EOS element so we add 1
    dim = (N,)*(n-1) + (N+1,)

    # Transition tensor
    A = np.zeros(dim)
    # Initial Probabilities
    Pi = np.zeros(N)

    for n_gram, frec in freq_n_grams.items():
      # Fill the tensor with frequencies
      if n_gram[0] != BOS_IDX:
          A[n_gram] = frec
      # Getting initial frequencies
      elif n_gram[0] == BOS_IDX and n_gram[1] != EOS_IDX:
          Pi[n_gram[1]] = frec

    # Calculating probabilities from frequencies
    # We consider the parameter `l` for Lidstone Smoothing
    for h, b in enumerate(A):
      A[h] = ((b+l).T/(b+l).sum(n-2)).T

    # Calculating initial probabilities
    Pi = (Pi+l)/(Pi+l).sum(0)

    # We get our model
    return A, Pi

def get_perplexity(sentence: str, vocab: defaultdict, model: tuple) -> float:
    """devuelve la perplejidad logarítmica de una oración en un modelo y vocabulario dado"""
    A, Pi = model
    n = len(A.shape)
    indexed_sentence = [vocab[word] for word in sentence.split()]
    first_indexed_word = indexed_sentence[0]
    # Getting initial probability
    probabilities = []
    try:
        probabilities.append(np.log(Pi[first_indexed_word]))
    except:
        #print(f"[WARN] OOV for word as BOS with index={first_indexed_word}")
        probabilities.append(0.0)
    n_grams = get_n_grams([indexed_sentence], n)
    N = len(indexed_sentence)
    for n_gram in n_grams:
        try:
          probabilities.append(np.log(A[n_gram]))
        except:
          probabilities.append(0.0)
    #print(probabilities)
    return (-1/N) * sum(probabilities)

corpus = get_corpus("https://www.gutenberg.org/cache/epub/2000/pg2000.txt")
corpus = preprocess_corpus(corpus)
corpus = corpus[:400] # acotamos el corpus para no quedarnos sin ram

corpus_train, corpus_test = train_test_split(corpus, test_size=0.3)

vocab = vocabulary_factory()
indexed_sents = list(word_to_index(corpus_train, vocab))
BOS = "<s>"
EOS = "</s>"

BOS_IDX = max(vocab.values())+2
EOS_IDX = max(vocab.values())+1

vocab[BOS] = BOS_IDX
vocab[EOS] = EOS_IDX

indexed_corpus_train = [[BOS_IDX] + sent + [EOS_IDX] for sent in indexed_sents]
vocab_words = get_index_to_word(vocab)

trigram_model = get_model(indexed_corpus_train, vocab, n=3, l=1)

bigram_model = get_model(indexed_corpus_train, vocab, n=2, l=1)

test_sent = " ".join([" ".join(sentence) for sentence in corpus_test])

bigram_perplex = np.exp(get_perplexity(test_sent, vocab, bigram_model))
trigram_perplex = np.exp(get_perplexity(test_sent, vocab, trigram_model))

print(f"Perplejidad bigrama:  {bigram_perplex}")
print(f"Perplejidad trigrama: {trigram_perplex}")
