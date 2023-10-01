# Bibliotecas
from collections import Counter,defaultdict
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [15, 6]
from re import sub
import numpy as np
from elotl import corpus as elotl_corpus
import nltk
from nltk.corpus import brown, cess_esp,stopwords
from rich.console import Console
from rich.table import Table
from rich.style import Style
# nltk.download("cess_esp")

# Código de ayudantía
# def preprocess_corpus(corpus):
#     # Obtener la oración de L1,
#     # quitar signos de puntuación y
#     # obtiene la lista de palabras
#     word_list_l1 = []
#     word_list_l2 = []
#     for row in corpus:
#         word_list_l1.extend(extract_words_from_sentence(row[0]))
#     # Obtener la oración de L1,
#     # quitar signos de puntuación y
#     # obtiene la lista de palabras
#         word_list_l2.extend(extract_words_from_sentence(row[1]))
#     return word_list_l1, word_list_l2



def sum_dicts(dic1: dict, dic2: dict) -> dict:
    return {k: dic1.get(k, 0) + dic2.get(k, 0) for k in set(dic1) | set(dic2)}

def sort_dict(dic: dict, max_to_min=True) -> dict:
    return {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse = max_to_min)}

def extract_chars_from_sentence(sentence, spaces=False, numbers = False):
    chars = {}
    for word in sub(r'[^\w\s\']', ' ', sentence).lower().split():
        w = Counter(word)
        chars = sum_dicts(chars,w)
    return chars

def chars_from_corpus(corpus, spaces=False, numbers = False) -> dict:
    chars = {}
    for word in corpus:
        chars = sum_dicts(Counter(word.lower()), chars)
    return chars

# Función modificada de ayudantía
def preprocess_corpus(corpus):
    # Obtener la oración de L1,
    # quitar signos de puntuación y
    # obtiene una cuenta de caracteres
    char_dict_l1 = {}
    char_dict_l2 = {}
    for row in corpus:
        char_dict_l1 = sum_dicts(extract_chars_from_sentence(row[0]), char_dict_l1)
    # Obtener la oración de L1,
    # quitar signos de puntuación y
    # obtiene la una cuenta de caracteres
        char_dict_l2 = sum_dicts(extract_chars_from_sentence(row[0]), char_dict_l2)
    return char_dict_l1, char_dict_l2
"""
Parte 1
"""
def get_frequencies(vocabulary: Counter, n: int = 1) -> list:
    return [_[1] for _ in vocabulary.most_common(n)]

def get_frequencies_dict(vocabulary: dict, n: int = 1) -> list:
    freqs = []
    for idx, value in enumerate(vocabulary):
        if idx >= n:
            break
        freqs.append([value, vocabulary[value]])
    return freqs

def def_value():
    return 0

def process_tag_corpus(corpus):
    chars = defaultdict(def_value)
    for word, tag in corpus:
        chars[tag] += 1
    return chars

def plot_corpus_tag(tags, show = False, with_labels = False):
    x = []
    y = []
    for t in tags:
        x.append(t[0])
        y.append(t[1])
    if not with_labels:
        x = [i for i in range(1,len(y)+1)]
    if show:
        plt.plot(x,y)
        plt.show()
    return x,y
# Código ayudantía
def extract_words_from_sentence(sentence: str) -> list:
    return sub(r'[^\w\s\']', ' ', sentence).lower().split()

def preprocess_corpus_words(corpus):
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
# n=100  

# brown_char = sort_dict(process_tag_corpus(brown.tagged_words()))
# cess_esp_char = sort_dict(process_tag_corpus(cess_esp.tagged_words()))
# tags_brown = get_frequencies_dict(brown_char,n)
# tags_cess_esp = get_frequencies_dict(cess_esp_char,n)
# brown_x, brown_y = plot_corpus_tag(tags_brown)
# cess_esp_x, cess_esp_y = plot_corpus_tag(tags_cess_esp)
# plt.plot(brown_x, brown_y)
# plt.plot(cess_esp_x, cess_esp_y)
# plt.show()
"""
Parte 2
"""
# # Importar corpus de nahúatl, otomí y español
axolotl = elotl_corpus.load("axolotl")
tsunkua = elotl_corpus.load("tsunkua")
# # Pre procesamiento de corpus
# spanish_char_na, nahuatl_char = preprocess_corpus(axolotl)
# spanish_char_oto, otomi_char = preprocess_corpus(tsunkua)
# D = sort_dict(nahuatl_char)
# plt.plot(range(len(D)), list(D.values()))
# plt.xticks(range(len(D)), list(D.keys()))
# plt.show()
"""
Parte 3
"""
spanish_words_na, nahuatl_words = preprocess_corpus_words(axolotl)
spanish_words_oto, otomi_words = preprocess_corpus_words(tsunkua)

def get_ngram(corpus,n=2)->list:
    count = 0
    ngrams = []
    current = ""
    for word in corpus:
        for char in word:
            count += 1
            current += char
            if count == n:
                ngrams.append(current)
                current = ""
                count = 0
    return ngrams

def plot_frequencies(frequencies: list, title="Freq of words"):
    x = list(range(1, len(frequencies)+1))
    plt.plot(x, frequencies, "-v")
    plt.xlabel("Freq rank (r)")
    plt.ylabel("Freq (f)")
    plt.title(title)


# most_common_count = 100
# nahuatl_ngram = Counter(get_ngram(nahuatl_words))
# nahuatl_ngram_freqs = get_frequencies(nahuatl_ngram, most_common_count)
# plot_frequencies(nahuatl_ngram_freqs, f"Frequencies for Nahúatl {most_common_count} most common")
# plt.show()
"""
Parte 4

Stopwords
"""
# Palabras más comunes
# nahuatl_vocabulary = Counter(nahuatl_words)
nahuatl_es_vocabulary = Counter(spanish_words_na)
# otomi_vocabulary = Counter(otomi_words)
otomi_es_vocabulary = Counter(spanish_words_oto)

table = Table(title = "Stopwords")
table.add_column("Corpus", style ="cyan")
table.add_column("Palabra", style="magenta")
table.add_column("Stopword", style="green")
placeholders = stopwords.words()
corpus = ["nahuatl", "otomí"]
i = 0
for vocabulary in [nahuatl_es_vocabulary,otomi_es_vocabulary]:
    for word in vocabulary.most_common(25):
        stpwrd = "Sí" if word[0] in placeholders else "No"
        table.add_row(
            corpus[i],
            word[0],
            stpwrd
            )
    i += 1
console = Console()
console.print(table)
# Respuestas stopwords coinciden con las paabras más comunes en Zipf?
# Realizar nube de palabras sans stopwords et palabras de zipf

# Lenguaje aleatorio con distribuciones Poisson, Uniforme Normal con media en cero, bi/tri normal, log normal, Zipf, 
# Usemos enlace egalitario para generar un nodo de tipo "letra" y cada rama es una palabra