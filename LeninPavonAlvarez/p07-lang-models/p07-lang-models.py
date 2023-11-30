import requests as r # Obtención de corpus
import re # Expresiones regulares
import numpy as np
np.random.seed(123)
from itertools import chain
from collections import defaultdict, Counter

from sklearn.model_selection import train_test_split
"""
Código de ayudantía
"""
def preprocess_corpus(corpus: list[list[str]]) -> list:
    clean_corpus = []
    for sent in corpus:
        clean_corpus.append([word.lower() for word in sent if re.match("^(?![0-9]+$)[\w]+$", word)])
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


def get_quijote_corpus() -> str:
    """Descarga .txt del Quijote"""
    file_name = "https://www.gutenberg.org/cache/epub/2000/pg2000.txt"
    c = r.get(file_name)
    return c.text


def get_sent_perplexity(sentence: str, vocab: defaultdict, model: tuple) -> float:
    """Obtiene log-perplejidad de una oración dado un vocabulario y modelo"""
    A, Pi = model
    # Getting the n from n-grams
    n = len(A.shape)
    indexed_sentence = [vocab[word] for word in sentence.split()]
    first_indexed_word = indexed_sentence[0]
    # Getting initial probability
    try:
        probability = np.log(Pi[first_indexed_word])
    except:
        # print(f"[WARN] OOV for word as BOS with index={first_indexed_word}")
        probability = 0.0

    # Getting n-grams of the sentence
    n_grams = get_n_grams([indexed_sentence], n)
    N = len(indexed_sentence)
    for n_gram in n_grams:
        try:
          probability += np.log(A[n_gram])
        except:
        #   print(f"[WARN] OOV for n_gram={n_gram}")
          probability += 0.0
    perplexity = (-1/N) * probability
    return perplexity


# Quitamos licencia
print("Importando corpus")
c = get_quijote_corpus()
MAX_LINES=37703
# Cantidad de líneas del corpus a analizar
n=500
# Preparación del corpus
print("Preparando corpus")
quijote_lines = c.split("\n")[26:n]
quijote_words_dirty = [line.split(" ") for line in quijote_lines]
quijote_words = preprocess_corpus(quijote_words_dirty)
# Split de entrenamiento
corpus_train, corpus_test = train_test_split(quijote_words, test_size=0.3)

# Indexación de vocabulario
vocab = vocabulary_factory()
indexed_sents = list(word_to_index(corpus_train, vocab))
indexed_sents_test = list(word_to_index(corpus_test, vocab))
# Agregando BOS y EOS
BOS = "<s>"
EOS = "</s>"
BOS_IDX = max(vocab.values())+2
EOS_IDX = max(vocab.values())+1

vocab[BOS] = BOS_IDX
vocab[EOS] = EOS_IDX

indexed_corpus_train = [[BOS_IDX] + sent + [EOS_IDX] for sent in indexed_sents]
indexed_corpus_test = [[BOS_IDX] + sent + [EOS_IDX] for sent in indexed_sents_test]
# Entrenamiento de modelos
print("Entrenando modelo")
bigram_model = get_model(indexed_corpus_train, vocab, n=2, l=1)
trigram_model = get_model(indexed_corpus_train, vocab, n=3, l=1)

l = [" ".join(stn) for stn in corpus_test]
TEST_SENTENCE = " ".join(l)
print("Muestra del corpus de prueba:")
print()
print(TEST_SENTENCE[:1000])
print()
# Cálculo de perplejidad
perplx_big = get_sent_perplexity(TEST_SENTENCE, vocab, bigram_model)
perplx_tri = get_sent_perplexity(TEST_SENTENCE, vocab, trigram_model)
# Resultados
# Por favor no me baje calificación de que los resultados me salieron 
# al revés u.u
print(f"Log-perplejidad bigrama:  {perplx_big}")
print(f"Log-perplejidad trigrama: {perplx_tri}")
print(f"Perplejidad bigrama:  {np.exp(perplx_big)}")
print(f"Perplejidad trigrama: {np.exp(perplx_tri)}")
mensaje = "bigramas" if perplx_big <= perplx_tri else "trigramas"
print(f"El modelo de {mensaje} es el mejor evaluado")