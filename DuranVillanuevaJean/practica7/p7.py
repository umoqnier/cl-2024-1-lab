# Preprocesando el texto
import requests

def get_tags_map():
    tags_raw = requests.get("https://fegalaz.usc.es/~gamallo/aulas/lingcomputacional/corpus/quijote-es.txt").text #.split("\n")
    return tags_raw

raw_corpus = get_tags_map()

def delete_subchain(main_string, subchain,change):
    return main_string.replace(subchain,change)

corpus = delete_subchain(delete_subchain(delete_subchain(raw_corpus,'\n',' '),':',''),';','')
corpus = corpus.split(' ')

def is_subchain_present(main_string, subchain):
    return main_string.find(subchain) != -1

"""
  Iteramos sobre el corpus. Vamos agregando cada palabra a una lista y terminamos de llenar dicha lista cuando una palabra tiene un punto "."
  Entonces ahora creamos otra lista agregando los siguientes elementos de la misma manera.
"""
def final_corpus(corpus):
  final_corpus = []
  l = []
  for c in corpus:
    if is_subchain_present(c,"."):
      c_aux = delete_subchain(c,".",'')
      l.append(c_aux)
      final_corpus.append(l)
      l = []
    else:
      l.append(c)
  return final_corpus

corpus = final_corpus(corpus)

"""# Modelo de lenguaje"""

from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import numpy as np
from itertools import chain
import math

BOS = "<s>"
EOS = "</s>"

def sentence_to_indices(sentece: list[str], vocab: dict):
    """Maps each word of the given sentence into its index
    found in vocab. If the word is not found, it maps to -1
    """
    words = [BOS] + sentece + [EOS]
    indices = []
    for word in words:
      if word in vocab:
        indices.append(vocab[word])
      else:
        indices.append(-1)
    return indices

def perplexity(sentence: str, n: int, model: tuple, vocab: dict):
    """
    A, Pi = model
    indices = sentence_to_indices(sentence, vocab)
    n_grams = list(zip(*[indices[i:] for i in range(n)]))
    probabilities = []

    first_word = n_grams[0][1]
    if first_word in A:
      probabilities.append(Pi[first_word])

    for n_gram in n_grams:
      if n_gram in A:
        probabilities.append(math.log(A[n_gram]))

    N = len(indices)
    log_perplexity = -1 // N * sum(probabilities)
    """

    ##
    A, Pi = model
    indexed_sentence = [vocab[word] for word in sentence.split()]
    first_indexed_word = indexed_sentence[0]
    # Getting initial probability
    try:
        probability = np.log(Pi[first_indexed_word])
    except:
        #print(f"[WARN] OOV for word as BOS with index={first_indexed_word}")
        probability = 0.0

    # Getting n-grams of the sentence
    n_grams = get_n_grams([indexed_sentence], n)
    for n_gram in n_grams:
        try:
          probability += np.log(A[n_gram])
        except:
          #print(f"[WARN] OOV for n_gram={n_gram}")
          probability += 0.0
    N = len(indexed_sentence)
    log_perplexity = -1 // N * probability
    ##
    return log_perplexity

def word_to_index(corpus: list[list[str]], vocab: defaultdict) -> list[int]:
  """Function that maps each word in a corpus to a unique index"""
  for sent in corpus:
    yield [vocab[word] for word in sent]

def get_vocabulary(corpus: list[list[str]]):
  """Given a corpus, builds a vocabulary where each word is given an index"""
  vocab = defaultdict()
  vocab.default_factory = lambda: len(vocab)

  indexed_sents = list(word_to_index(corpus, vocab))
  BOS_IDX = max(vocab.values())+2
  EOS_IDX = max(vocab.values())+1
  vocab[BOS] = BOS_IDX
  vocab[EOS] = EOS_IDX
  indexed_sents = [[BOS_IDX] + sent + [EOS_IDX] for sent in indexed_sents]

  return vocab, indexed_sents

def get_n_grams(indexed_sents: list[list[str]], n=2) -> chain:
  return chain(*[zip(*[sent[i:] for i in range(n)]) for sent in indexed_sents])

def get_model(sents: list[list[str]], vocabulary: defaultdict, n: int = 2, l: float=1.0) -> tuple:
  n_grams = get_n_grams(sents, n)
  m_grams = get_n_grams(sents, n - 1)
  freq_n_grams = Counter(n_grams)
  freq_m_grams = Counter(m_grams)

  N = len(vocabulary) - 2 #vocabulary length (without BOS/EOS)
  Pi = np.zeros(N)
  dist = {}

  for n_gram, frec in freq_n_grams.items():
    if n_gram[0] != vocabulary[BOS]:
      dist[n_gram] = frec
      pass
    elif n_gram[0] == vocabulary[BOS] and n_gram[1] != vocabulary[EOS]:
      Pi[n_gram[1]] = frec

  # Calculating initial probabilities
  Pi = (Pi+l)/(Pi+l).sum(0)
  A = { n_gram: probablity(n_gram,freq_m_grams,dist,l,N) for n_gram, _ in dist.items() }

  return A, Pi

def probablity(n_gram,freq_m_grams,dist,l,N):
  m_gram = n_gram[:-1]
  m_count = freq_m_grams[m_gram]
  n_count = dist[n_gram]
  return (n_count + l) / (m_count + l * N)

from sklearn.model_selection import train_test_split
corpus_train, corpus_test = train_test_split(corpus, test_size=0.3)

vocab, index_sents = get_vocabulary(corpus_train)

n_grams = get_n_grams(index_sents, 2)
freq_n_grams = Counter(n_grams)

def average_perplexity(corpus, n, model, vocab):
  perplexities = 0
  for sentence in corpus:
    perplexities += perplexity(" ".join(sentence), n, model, vocab)
  return perplexities // len(corpus)

bigram_model = get_model(index_sents, vocab, n=2, l=1)
trigram_model = get_model(index_sents, vocab, n=3, l=1)

perplexity_average_3gram = average_perplexity(corpus_test, 3, trigram_model, vocab)
perplexity_average_2gram = average_perplexity(corpus_test, 2, bigram_model, vocab)

print("Preparando el modelo, calculando perplejidad...\n")
print("Perplejidades promedio de los modelos:")
print("Bigramas:", perplexity_average_2gram," >  Trigramas:", perplexity_average_3gram)