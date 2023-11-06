from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import numpy as np
from itertools import chain
import math

# Symbols that are used to identify which words
# are used the most at the beginning or end of a
# sentence
# Beginning of Sentence
BOS = "<s>"
# End of Sentence
EOS = "</s>"

def get_vocabulary(corpus: list[list[str]]):
    """Given a corpus, builds a vocabulary where each word is given an index"""

    vocab = vocabulary_factory()
    
    # Side effect: Changes vocab value
    indexed_sents = list(word_to_index(corpus, vocab))

    BOS_IDX = max(vocab.values())+2
    EOS_IDX = max(vocab.values())+1
    
    vocab[BOS] = BOS_IDX
    vocab[EOS] = EOS_IDX

    indexed_sents = [[BOS_IDX] + sent + [EOS_IDX] for sent in indexed_sents]

    return vocab, indexed_sents


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

def get_n_grams(indexed_sents: list[list[str]], n=2) -> chain:
    return chain(*[zip(*[sent[i:] for i in range(n)]) for sent in indexed_sents])

def get_model(sents: list[list[str]], vocabulary: defaultdict, n: int = 2, l: float=1.0) -> tuple:
    """Builds a n-gram model
    
    Parameters
    ----------
    sents: list[list[str]]
        a list of sentences, where a sentence is a list of words.
        For instance: [['<s>', 'Hello', 'Wolrd', '</s>']]
    
    vocabulary: dict
        where for each word (the key), is stored an id

    n: int
        n-grams

    l: float
        for smoothing

    Returns
    ------- 
    (dict, np.Array)
        where the first element is the smoothed probablity distribution of n-grams,
        while the second one contains the initial probabilities
    """

    n_grams = get_n_grams(sents, n)
    m_grams = get_n_grams(sents, n - 1)
    freq_n_grams = Counter(n_grams)
    freq_m_grams = Counter(m_grams)

    # Get vocabulary length (without BOS/EOS)
    N = len(vocabulary) - 2

    # Initial Probabilities
    Pi = np.zeros(N)

    # Changed from numpy array to dict to avoid memory usage
    # Might be probability of some collisions
    dist = {}

    for n_gram, frec in freq_n_grams.items():
        if n_gram[0] != vocabulary[BOS]:
           dist[n_gram] = frec
           pass
        elif n_gram[0] == vocabulary[BOS] and n_gram[1] != vocabulary[EOS]:
          Pi[n_gram[1]] = frec

    def probablity(n_gram):
        m_gram = n_gram[:-1]
        m_count = freq_m_grams[m_gram]
        n_count = dist[n_gram]

        return (n_count + l) / (m_count + l * N)

    # Calculating initial probabilities
    Pi = (Pi+l)/(Pi+l).sum(0)
    A = { n_gram: probablity(n_gram) for n_gram, _ in dist.items() }

    # We get our model
    return A, Pi

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
    """Calculates log perplexity for a given word, using
    a n-gram model
    
    Parameters:
    -----------
    sentence: str
        a sentence already preprocessed
    
    model: tuple
        the probability distribution for the n-grams and the initial
        probabilities

    n: int
        n-grams

    vocab: dict
        where for each word (the key), is stored an id
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

    return log_perplexity
