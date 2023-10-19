# Autor: Mikel Segura Elizalde
# Versión 1, octubre 2023

import nltk
from nltk.corpus import stopwords
from nltk.corpus import cess_esp
from nltk.corpus import brown
from elotl import corpus as elotl_corpus
from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 6]
from re import sub
import numpy as np
nltk.download("cess_esp")
nltk.download('brown')
nltk.download('stopwords')

"""# POS"""

def get_frequencies(vocabulary: Counter, n: int) -> list:
    return [_[1] for _ in vocabulary.most_common(n)]

spanish_counter = Counter([tag[0][1] for tag in cess_esp.tagged_sents()])
frequencies_spanish_pos = get_frequencies(spanish_counter, 100)

english_counter = Counter([tag[0][1] for tag in brown.tagged_sents()])
frequencies_english_pos = get_frequencies(english_counter, 100)

x = list(range(1, 101))
plt.figure()
plt.plot(x, frequencies_english_pos, "-v", label="English POS tagging")
plt.plot(x, frequencies_spanish_pos, "-v", label="Spanish POS tagging")
plt.legend()
plt.show()

"""# CARACTERES"""

axolotl = elotl_corpus.load("axolotl")
tsunkua = elotl_corpus.load("tsunkua")
def extract_words_from_sentence(sentence: str) -> list:
    return sub(r'[^\w\s\']', ' ', sentence).lower().split()
def preprocess_corpus(corpus):
    # Obtener la oración de L1,
    # quitar signos de puntuación y
    # obtiene la lista de palabras
    word_list_l1 = []
    word_list_l2 = []
    for row in corpus:
        word_list_l1.extend(extract_words_from_sentence(row[0]))
    # Obtener la oración de L1,
    # quitar signos de puntuación y
    # obtiene la lista de palabras
        word_list_l2.extend(extract_words_from_sentence(row[1]))
    return word_list_l1, word_list_l2
spanish_words_na, nahuatl_words = preprocess_corpus(axolotl)
spanish_words_oto, otomi_words = preprocess_corpus(tsunkua)

frequencies_nahuatl_characters = get_frequencies(Counter(''.join(nahuatl_words)), 60)
frequencies_spanish_characters = get_frequencies(Counter(''.join(cess_esp.words())), 60)
frequencies_otomi_characters = get_frequencies(Counter(''.join(otomi_words)), 60)

x = list(range(1, 61))
plt.figure()
plt.plot(x, frequencies_nahuatl_characters, "-v", label="Nahuatl characters")
plt.plot(x, frequencies_otomi_characters, "-v", label="Otomi characters")
plt.plot(x, frequencies_spanish_characters, "-v", label="Spanish characters")
plt.legend()
plt.show()

"""# 2-GRAMAS"""

def twogram(word):
    return [word[i]+word[i+1] for i in list(range(0,len(word)-1))]
def twograms(words):
  result = []
  for word in words:
    if len(word) > 1:
      result += twogram(word)
  return result

frequencies_spanish_twograms = get_frequencies(Counter(twograms(cess_esp.words())), 200)
frequencies_nahuatl_twograms = get_frequencies(Counter(twograms(nahuatl_words)), 200)
frequencies_otomi_twograms = get_frequencies(Counter(twograms(otomi_words)), 200)

x = list(range(1, 201))
plt.figure()
plt.plot(x, frequencies_nahuatl_twograms, "-v", label="Nahuatl twograms")
plt.plot(x, frequencies_otomi_twograms, "-v", label="Otomi twograms")
plt.plot(x, frequencies_spanish_twograms, "-v", label="Spanish twograms")
plt.legend()
plt.show()

"""# NUBES DE PALABRAS"""

spanish_stopwords = set(nltk.corpus.stopwords.words('spanish'))

wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = spanish_stopwords,
                min_font_size = 10).generate(' '.join(cess_esp.words()))
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()

vocabulary = Counter(cess_esp.words())
zipf_stopwords = [frequency[0] for frequency in vocabulary.most_common(313)]
wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = zipf_stopwords,
                min_font_size = 10).generate(' '.join(cess_esp.words()))
plt.figure(figsize = (8, 8), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()