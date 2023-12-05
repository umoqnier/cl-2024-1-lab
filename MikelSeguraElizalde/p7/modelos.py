# Autor: Mikel Segura Elizalde
# Versión 1, noviembre 2023

import re
import requests
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import numpy as np
from itertools import chain
import math

# PREPARANDO EL CORPUS
print('preparando corpus...', end = '')

def replace_many(string, to_replace, replacement):
  result = string
  for symbol in to_replace:
    result = result.replace(symbol, replacement)
  return result

def preprocess_corpus(corpus: list[list[str]]) -> list:
    clean_corpus = []
    for sent in corpus:
        clean_corpus.append([word.lower() for word in sent if re.match("^(?![0-9]+$)[\w\s]+$", word)])
    return clean_corpus

raw_file = requests.get('https://www.gutenberg.org/cache/epub/2000/pg2000.txt')
raw = raw_file.text
raw_no_endline_marks = raw.replace('\r\n', ' ')
raw_normal_eos_punctuation = replace_many(raw_no_endline_marks, '?!;', '.')
raw_dot_split = raw_normal_eos_punctuation.split('.')
raw_words = [replace_many(sentence, '»¿¡,—', '').split(' ') for sentence in raw_dot_split]
corpus = preprocess_corpus(raw_words)[1000:1100]
corpus_train, corpus_test = train_test_split(corpus, test_size=0.3)
print('listo')

# PREPARANDO EL VOCABULARIO

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

vocab = vocabulary_factory()
indexed_sents = list(word_to_index(corpus_train, vocab))
indexed_sents_test = list(word_to_index(corpus_test, vocab))
BOS = "<s>"
EOS = "</s>"
BOS_IDX = max(vocab.values())+2
EOS_IDX = max(vocab.values())+1
vocab[BOS] = BOS_IDX
vocab[EOS] = EOS_IDX
indexed_corpus_train = [[BOS_IDX] + sent + [EOS_IDX] for sent in indexed_sents]

# DEFINIENDO LOS MODELOS DE n-GRAMAS

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

# APLICACIÓN: PROBABILIDAD DE UNA CADENA

def get_sent_probability(sentence: str, vocab: defaultdict, model: tuple) -> float:
    A, Pi = model
    # Getting the n from n-grams
    n = len(A.shape)
    indexed_sentence = [vocab[word] for word in sentence.split()]
    first_indexed_word = indexed_sentence[0]
    # Getting initial probability
    try:
        probability = np.log(Pi[first_indexed_word])
    except:
        print(f"[WARN] OOV for word as BOS with index={first_indexed_word}")
        probability = 0.0
    # Getting n-grams of the sentence
    n_grams = get_n_grams([indexed_sentence], n)
    for n_gram in n_grams:
        try:
          probability += np.log(A[n_gram])
        except:
          print(f"[WARN] OOV for n_gram={n_gram}")
          probability += 0.0
    return probability

# DEFINICIÓN DE LA EVALUACIÓN DE LOS MODELOS MEDIANTE PERPLEJIDAD

def perplexity(vocab:defaultdict, model:tuple, tests:list[list[str]], per_word=False)->float:
  model_tests = [get_sent_probability(' '.join(sent), vocab, model) for sent in tests]
  if per_word == False:
    N = len(tests)
  else:
    N = 0
    for sent in tests:
      N += len(sent)
  log_sum = 0
  for x in model_tests:
    log_sum += x
  power = -(1/N)*log_sum
  return np.exp(power)

print('obteniendo el modelo de bigramas...', end = '')
bigram_model = get_model(indexed_corpus_train, vocab, n=2, l=1)
print('listo')
print('obteniendo el modelo de trigramas...', end = '')
trigram_model = get_model(indexed_corpus_train, vocab, n=3, l=1)
print('listo\n')
print(f'perplejidad del modelo de bigramas: {perplexity(vocab, bigram_model, corpus_test)}')
print(f'perplejidad del modelo de trigramas: {perplexity(vocab, trigram_model, corpus_test)}')
print(f'perplejidad por palabras del modelo de bigramas: {perplexity(vocab, bigram_model, corpus_test, True)}')
print(f'perplejidad por palabras del modelo de trigramas: {perplexity(vocab, trigram_model, corpus_test, True)}')

input('presiona enter para salir')