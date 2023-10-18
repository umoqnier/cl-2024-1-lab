import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
from re import sub
from wordcloud import WordCloud


def extract_pos_tags(corpus: list[tuple[str, str]]) -> list[str]:
  """Given a `corpus` with items: (word, tag), gets only the tags,
  including repetitions"""
  
  tags = []
  for sentence in corpus:
    for item in sentence:
      _, tag = item
      tags.append(tag)

  return tags


def extract_words_from_sentence(sentence: str) -> list:
    return sub(r'[^\w\s\']', ' ', sentence).lower().split()


def preprocess_corpus(corpus: list) -> list:
    """Given a parallel `corpus` [[lang 1] [lang2]], extracts the words
    from both language sentence corpus"""

    word_list_l1 = []
    word_list_l2 = []
    for row in corpus:
        word_list_l1.extend(extract_words_from_sentence(row[0]))
        word_list_l2.extend(extract_words_from_sentence(row[1]))
    return word_list_l1, word_list_l2


def get_frequencies(corpus: list[str], n: int) -> list:
    """Gets the frequencies of the most common `n` elements of a given
    `corpus`"""

    counter = Counter(corpus)
    return [_[1] for _ in counter.most_common(n)]


def get_character_frequencies(words: list, most_common: int):
    """Gets the frequencies of single characters from the `most_common` 
    characters of a given list of `words`"""

    chacaters = []
    for word in words:
        chacaters.extend([*word])

    return get_frequencies(chacaters, most_common)


def to_ngrams(sentence: str, n: int):
    """Converts a `sentence` into `n` grams. For instance:
    "Mi querida", n=2 -> ["mi", "qu", "ue", "er", "ri", "id", "da"]
    """

    sent = sentence.replace(" ", "")
    ngrams = []
    for i in range(len(sent) - n + 1):
        ngram = sent[i:i + n]
        ngrams.append(ngram)
    return ngrams


def get_ngrams_frequencies(words, n, most_common):
    """Gets the frequencies of the `most_common` `n`-grams of a 
    given list of `words`"""

    ngrams = [ to_ngrams(word, n) for word in words ]
    flattened = [item for sublist in ngrams for item in sublist]

    return get_frequencies(flattened, most_common)


def plot_frequencies(frequencies: list, title="Freq of words") -> None:
    x = list(range(1, len(frequencies)+1))
    plt.figure()
    plt.plot(x, frequencies, "-v")
    plt.xlabel("Freq rank (r)")
    plt.ylabel("Freq (f)")
    plt.title(title)

def plot_list_frequencies(list_frequencies: list[tuple[str, list]], title="") -> None:
    
    min_size = min([ len(freqs) for _, freqs in list_frequencies ])
    x = list(range(1, min_size+1))

    plt.figure()
    for item in list_frequencies:
        freq_label, frequencies = item 
        plt.plot(x, frequencies, "-v", label=freq_label)

    plt.xlabel("Freq rank (r)")
    plt.ylabel("Freq (f)")
    plt.legend()
    plt.title(title)
    plt.show()

def generate_zipf_frequencies(n, a=1.5, N=100000):
    zipf_distribution = np.random.zipf(a, N)
    zipf_numbers_freqs = get_frequencies(Counter(zipf_distribution), n)

    return zipf_numbers_freqs

def plot_log_with_zipf(frequencies: list, label: str = "") -> None:
    x = list(range(1, len(frequencies) + 1))
    zipf_numbers_freqs = generate_zipf_frequencies(len(frequencies))

    plt.figure()
    plt.loglog(x, zipf_numbers_freqs, label="Zipf generated")
    plt.loglog(x, frequencies, "-v", label=label)
    plt.legend()
    plt.show()


def plot_log_with_list_zipf(list_frequencies: list[tuple[str, list]]) -> None:
    min_size = min([ len(freqs) for _, freqs in list_frequencies ])
    zipf_numbers_freqs = generate_zipf_frequencies(min_size)

    x = list(range(1, min_size+1))
    plt.figure()
    for item in list_frequencies:
        freq_label, frequencies = item 
        plt.loglog(x, frequencies, "-v", label=freq_label)

    plt.loglog(x, zipf_numbers_freqs, label="Zipf generated")
    plt.legend()
    plt.show()


def plot_wordcloud(most_common_words: list[str]) -> None:
    """Plots a wordcloud given a list of word. It's important to sort the list
    in respect to the frequency of the word, considering the first word the
    most frequent and the last the less"""

    wordcloud = WordCloud(width=800, height=400, background_color='white')

    text = " ".join(most_common_words)
    wordcloud.generate(text)

    # Display the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()