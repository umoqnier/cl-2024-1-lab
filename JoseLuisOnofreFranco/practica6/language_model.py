from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter
import numpy as np
from itertools import chain

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
        the number of grams to make

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
    A = {}

    for n_gram, frec in freq_n_grams.items():
        if n_gram[0] != vocabulary[BOS]:
           A[n_gram] = frec
           pass
        elif n_gram[0] == vocabulary[BOS] and n_gram[1] != vocabulary[EOS]:
          Pi[n_gram[1]] = frec

    def probablity(n_gram):
        m_gram = n_gram[:-1]
        m_count = freq_m_grams[m_gram]
        n_count = A[n_gram]

        return (n_count + l) / (m_count + l * N)

    # Calculating initial probabilities
    Pi = (Pi+l)/(Pi+l).sum(0)
    result = { n_gram: probablity(n_gram) for n_gram, _ in A.items() }

    # We get our model
    return result, Pi
