# Bibliotecas
from collections import Counter,defaultdict
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [9, 6]
from re import sub
import numpy as np
from elotl import corpus as elotl_corpus
import nltk
from nltk.corpus import brown, cess_esp,stopwords
from rich import print
from rich.console import Console
from rich.table import Table
from rich.style import Style
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
# nltk.download("cess_esp")

console = Console()
layout = Layout()


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
"""
Presentación
"""
texto_presentación = """Práctica 4. Zipf
Lenin Pavón Alvarez"""
print(Panel(Text(texto_presentación,style="bold magenta",justify="center"), border_style="magenta"))


"""
Parte 1
"""
# Texto presentación
print(Panel(Text("Parte 1: POS",style="bold cyan",justify="center"), border_style="cyan"))
texto_p1 = "Vamos a procesar el corpus [bold][red]brown[/red][/bold] y [bold][red]cess_esp[/red][/bold] donde veremos que las etiquetas POS siguen la distribución Zipf."
print(texto_p1)
# Inicio procesamiento brown
n=75  
print("Procesando el corpus [bold red]brown[/bold red]")
brown_char = sort_dict(process_tag_corpus(brown.tagged_words()))
print("Obteniendo las palabras más frecuentes de [bold red]brown[/bold red]")
tags_brown = get_frequencies_dict(brown_char,n)
print("Procesando [bold red]brown[/bold red] para su visualización")
brown_x, brown_y = plot_corpus_tag(tags_brown)
print("[bold red]brown[/bold red] procesado")


# Inicio procesamiento cess_esp
print("Procesando el corpus [bold red]cess_esp[/bold red]")
cess_esp_char = sort_dict(process_tag_corpus(cess_esp.tagged_words()))
print("Obteniendo las palabras más frecuentes del corpus [bold red]cess_esp[/bold red]")
tags_cess_esp = get_frequencies_dict(cess_esp_char,n)
print("Procesando [bold red]cess_esp[/bold red] para su visualización")
cess_esp_x, cess_esp_y = plot_corpus_tag(tags_cess_esp)
print("[bold red]cess_esp[/bold red] procesado")


# Graficación con subplots
fig, axs = plt.subplots(2)
fig.suptitle('Parte 1. POS')
axs[0].plot(brown_x, brown_y,'tab:purple')
axs[0].set_title('Brown')
axs[0].set(ylabel='Freq(r)')
axs[1].plot(cess_esp_x, cess_esp_y,'tab:purple')
axs[1].set_title('cess_esp')
axs[1].set(xlabel='Freq. rank (r)', ylabel='Freq(r)')
plt.show()


"""
Parte 2
"""
# Texto presentación
print(Panel(Text("Parte 2: Caracteres",style="bold cyan",justify="center"), border_style="cyan"))
texto_p2 = "Habiendo procesado las etiquetas POS " + \
    "vamos a analizar la distribución de caracteres " + \
    "en náhuatl ([bold red]axolotl[/bold red]) " + \
    "en otomí ([bold red]tsunkua[/bold red]) " +\
    "y sus respectivos corpora paralelos en español."
print(texto_p2)


#
# Importar corpus de nahúatl, otomí y español
#

# Náhuatl
print("Importando [bold red]axolotl[/bold red]")
axolotl = elotl_corpus.load("axolotl")
print("Procesando [bold red]axolotl[/bold red]")
spanish_char_na, nahuatl_char = preprocess_corpus(axolotl)

# Otomí
print("Importando [bold red]tsunkua[/bold red]")
tsunkua = elotl_corpus.load("tsunkua")
print("Procesando [bold red]tsunkua[/bold red]")
spanish_char_oto, otomi_char = preprocess_corpus(tsunkua)

#
# Visualización
#
print("Graficando...")

fig, axs = plt.subplots(2, 2)
# Axolotl
nah_plot = sort_dict(nahuatl_char)
axs[0, 0].plot(range(len(nah_plot)), list(nah_plot.values()))
# axs[0, 0].set_xticks(range(len(nahuatl_char)), list(nahuatl_char.keys()))
axs[0, 0].set_title('Náhuatl')

# Axolotl esp
spa1_plot = sort_dict(spanish_char_na)
axs[0, 1].plot(range(len(spa1_plot)), list(spa1_plot.values()))
# axs[0, 1].set_xticks(range(len(spanish_char_na)), list(spanish_char_na.keys()))
axs[0, 1].set_title('Español (axolotl)')

# Tsunkua
oto_plot = sort_dict(otomi_char)
axs[1, 0].plot(range(len(oto_plot)), list(oto_plot.values()))
# axs[1, 0].set_xticks(range(len(otomi_char)), list(otomi_char.keys()))
axs[1, 0].set_title('Otomí')

# Tsunkua esp
spa2_plot = sort_dict(spanish_char_oto)
axs[1, 1].plot(range(len(spa2_plot)), list(spa2_plot.values()))
# axs[1, 1].set_xticks(range(len(spanish_char_oto)), list(spanish_char_oto.keys()))
axs[1, 1].set_title('Español (tsunkua)')

for ax in axs.flat:
    ax.set(xlabel='Freq. rank (r)', ylabel='Freq(r)')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()


"""
Parte 3
"""
# Texto presentación
print(Panel(Text("Parte 3: Caracteres",style="bold cyan",justify="center"), border_style="cyan"))
texto_p3 = "Habiendo procesado los caracteres " + \
    "vamos a analizar la distribución de 2-gramas sobre " + \
    "los mismos corpora."
print(texto_p3)


# Procesamiento axolotl
print("Pre-rocesando [bold red]axolotl[/bold red]")
spanish_words_na, nahuatl_words = preprocess_corpus_words(axolotl)
print("Pre-Procesando [bold red]tsunkua[/bold red]")
spanish_words_oto, otomi_words = preprocess_corpus_words(tsunkua)

most_common_count = 100

# for id,corp in enumerate([spanish_words_na, nahuatl_words,spanish_words_oto, otomi_words]):
#     print(f"Creando 2-gramas de [bold red]{corp_title[id]}[/bold red]")
#     ngram = Counter(get_ngram(nahuatl_words))
#     print(f"Contando 2-gramas de [bold red]{corp_title[id]}[/bold red]")
#     ngram_freqs = get_frequencies(ngram, most_common_count)
#     plot_frequencies(ngram_freqs, f"Frequencies for {most_common_count} most common 2-grams")

fig, axs = plt.subplots(2, 2)
# Axolotl
nah_ngram = Counter(get_ngram(nahuatl_words))
nah_plot = get_frequencies(nah_ngram, most_common_count)
axs[0, 0].plot(range(len(nah_plot)), nah_plot)
# axs[0, 0].set_xticks(range(len(nahuatl_char)), list(nahuatl_char.keys()))
axs[0, 0].set_title('Náhuatl')

# Axolotl esp
spa1_ngram = Counter(get_ngram(spanish_words_na))
spa1_plot = get_frequencies(spa1_ngram, most_common_count)
axs[0, 1].plot(range(len(spa1_plot)), spa1_plot)
# axs[0, 1].set_xticks(range(len(spanish_char_na)), list(spanish_char_na.keys()))
axs[0, 1].set_title('Español (axolotl)')

# Tsunkua
oto_ngram = Counter(get_ngram(otomi_words))
oto_plot = get_frequencies(oto_ngram, most_common_count)
axs[1, 0].plot(range(len(oto_plot)), oto_plot)
# axs[1, 0].set_xticks(range(len(otomi_char)), list(otomi_char.keys()))
axs[1, 0].set_title('Otomí')

# Tsunkua esp
spa2_ngram = Counter(get_ngram(spanish_words_oto))
spa2_plot = get_frequencies(spa2_ngram, most_common_count)
axs[1, 1].plot(range(len(spa2_plot)), spa2_plot)
# axs[1, 1].set_xticks(range(len(spanish_char_oto)), list(spanish_char_oto.keys()))
axs[1, 1].set_title('Español (tsunkua)')

for ax in axs.flat:
    ax.set(xlabel='Freq. rank (r)', ylabel='Freq(r)')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()
"""
Parte 4

Stopwords
"""
# Texto presentación
print(Panel(Text("Parte 4: Caracteres",style="bold cyan",justify="center"), border_style="cyan"))
texto_p4 = "Habiendo procesado los 2-gramas " + \
    "vamos a analizar si las palabras " + \
    "más comunes (en español) de " +\
    "[bold red]axolotl[/bold red] y "+\
    "[bold red]tsunkua[/bold red] coinciden "+\
    "con las [bold red]stopwords[/bold red] de "+\
    "[cyan]nltk[cyan]."
print(texto_p4)

#
# Palabras más comunes
#
print("Contando palabras en [bold red]axolotl[/bold red]")
nahuatl_es_vocabulary = Counter(spanish_words_na)
print("Contando palabras en [bold red]tsunkua[/bold red]")
otomi_es_vocabulary = Counter(spanish_words_oto)
print("Visualizando...\n")


# Visualización
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
# Realizar nube de palabras sans stopwords et palabras de zipf

# Lenguaje aleatorio con distribuciones Poisson, Uniforme Normal con media en cero, bi/tri normal, log normal, Zipf, 
# Usemos enlace egalitario para generar un nodo de tipo "letra" y cada rama es una palabra