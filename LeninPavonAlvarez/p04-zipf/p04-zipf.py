"""
Bibliotecas
"""
# NLP
from elotl import corpus as elotl_corpus
from nltk.corpus import brown, cess_esp,stopwords
# Procesamiento de datos
from collections import Counter,defaultdict
from re import sub
# Visualización de datos
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [9, 6]
from rich import print
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text
from wordcloud import WordCloud

# Permite impresión de tablas 
console = Console()
layout = Layout()

# Funciones de diccionarios
def sum_dicts(dic1: dict, dic2: dict) -> dict:
    """Sum of two dictionaries of frequencies"""
    return {k: dic1.get(k, 0) + dic2.get(k, 0) for k in set(dic1) | set(dic2)}


def sort_dict(dic: dict, max_to_min=True) -> dict:
    """Sort a dictionary of frequencies"""
    return {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse = max_to_min)}


# Procesamiento de corpus
def extract_chars_from_sentence(sentence: list[str]) -> dict:
    """Create dictionary of frequencies from a sentence"""
    chars = {}
    for word in sub(r'[^\w\s\']', ' ', sentence).lower().split():
        w = Counter(word)
        chars = sum_dicts(chars,w)
    return chars


def chars_from_corpus(corpus: list[str]) -> dict:
    """Create dictionary of frequencies from a corpus"""
    chars = {}
    for word in corpus:
        chars = sum_dicts(Counter(word.lower()), chars)
    return chars


# Función modificada de ayudantía
def preprocess_corpus(corpus: list) -> tuple:
    """
    Create a dictionary of frequencies of characters from 
    a parallel corpus for L1 and L2
    """
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
    """Return n most common words of a counter"""
    return [_[1] for _ in vocabulary.most_common(n)]


def get_frequencies_dict(vocabulary: dict, n: int = 1) -> list:
    """Return n most common words of a dictionary"""
    freqs = []
    for idx, value in enumerate(vocabulary):
        if idx >= n:
            break
        freqs.append([value, vocabulary[value]])
    return freqs


def def_value():
    """Return 0 if value is not in dictionary"""
    return 0

def process_tag_corpus(corpus: list[tuple]):
    """
    Create dictionary of frequencies for the tags 
    of a tagged corpus
    """
    chars = defaultdict(def_value)
    for word, tag in corpus:
        chars[tag] += 1
    return chars


def plot_corpus_tag(tags: list[tuple], show: bool = False, with_labels: bool = False):
    """
    Prepares dictionary of frequencies for the tags 
    of a tagged corpus for plotting
    """
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


def get_ngram(corpus: list[str],n: int = 2)->list:
    """
    Creates corpus of n-grams from a corpus of words
    """
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


# Código ayudantía
def extract_words_from_sentence(sentence: str) -> list[str]:
    """Return words from sentence"""
    return sub(r'[^\w\s\']', ' ', sentence).lower().split()

def preprocess_corpus_words(corpus):
    """
    Create a dictionary of frequencies of words from 
    a parallel corpus for L1 and L2
    """
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
Parte 1: POS
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
Parte 2: Caracteres
"""
# Texto presentación
print(Panel(Text(
    "Parte 2: Caracteres",
    style="bold cyan",
    justify="center"), border_style="cyan"))
texto_p2 = "Habiendo procesado las etiquetas POS " + \
    "vamos a analizar la distribución de caracteres " + \
    "en náhuatl ([bold red]axolotl[/bold red]) " + \
    "en otomí ([bold red]tsunkua[/bold red]) " +\
    "y sus respectivos corpora paralelos en español."
print(texto_p2)


#
# Importar corpus de nahúatl, otomí y español
#


# Axolotl (Náhuatl)
print("Importando [bold red]axolotl[/bold red]")
axolotl = elotl_corpus.load("axolotl")
print("Procesando [bold red]axolotl[/bold red]")
spanish_char_na, nahuatl_char = preprocess_corpus(axolotl)


# Tsunkua (Otomí)
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
axs[0, 0].set_title('Náhuatl')

# Axolotl esp
spa1_plot = sort_dict(spanish_char_na)
axs[0, 1].plot(range(len(spa1_plot)), list(spa1_plot.values()))
axs[0, 1].set_title('Español (axolotl)')

# Tsunkua
oto_plot = sort_dict(otomi_char)
axs[1, 0].plot(range(len(oto_plot)), list(oto_plot.values()))
axs[1, 0].set_title('Otomí')

# Tsunkua esp
spa2_plot = sort_dict(spanish_char_oto)
axs[1, 1].plot(range(len(spa2_plot)), list(spa2_plot.values()))
axs[1, 1].set_title('Español (tsunkua)')

# x-label and y-label
for ax in axs.flat:
    ax.set(xlabel='Freq. rank (r)', ylabel='Freq(r)')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()


"""
Parte 3: 2-gramas
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

# Los k n-gramas más comunes
most_common_count = 100

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

# x-label and y-label
for ax in axs.flat:
    ax.set(xlabel='Freq. rank (r)', ylabel='Freq(r)')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.show()


"""
Parte 4: stopwords
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

#
# Visualización
#

# Preparación de tabla
table = Table(title = "Stopwords")
table.add_column("Corpus", style ="cyan")
table.add_column("Palabra", style="magenta")
table.add_column("Stopword", style="green")
placeholders = stopwords.words()
corpus = ["nahuatl", "otomí"]
i = 0
n=50

# Creación de tabla
for vocabulary in [nahuatl_es_vocabulary,otomi_es_vocabulary]:
    for word in vocabulary.most_common(n):
        stpwrd = "Sí" if word[0] in placeholders else "No"
        table.add_row(
            corpus[i],
            word[0],
            stpwrd
            )
    i += 1

# Impresión de tabla
console.print(table)


# Corpus conjunto
print("Ahora sumemos la frecuencia de las palabras en ambos corpus\n")
esp_vocabulary = sort_dict(sum_dicts(nahuatl_es_vocabulary,otomi_es_vocabulary))

# Preparación de tabla de corpus conjunto
table = Table(title = "Stopwords")
table.add_column("Palabra", style="magenta")
table.add_column("Stopword", style="green")
placeholders_alt = []

# Creación de tabla de corpus conjunto
for idx,word in enumerate(esp_vocabulary.keys()):
    if idx >= n:
        break
    stpwrd = "Sí" if word in placeholders else "No"
    placeholders_alt.append(word)
    table.add_row(
        word,
        stpwrd
        )

# Impresión de tablas de corpus conjunto
console.print(table)


#
# Nubes de palabra del corpus conjunto
#

print("Graficando las nubes de palabras usando [bold red]stopwords[/bold red]\n")

# Creación de nubes de palabra
wordcloud_stopwords = WordCloud(width = 1000, height = 500).generate_from_frequencies({key:val for key,val in esp_vocabulary.items() if key not in placeholders})
wordcloud_stopwords_alt = WordCloud(width = 1000, height = 500).generate_from_frequencies({key:val for key,val in esp_vocabulary.items() if key not in placeholders_alt})

# Visualización de nubes de palabra
fig, axs = plt.subplots(2,figsize=(15,8))
fig.suptitle('Parte 4. Nubes de palabras')
axs[0].imshow(wordcloud_stopwords)
axs[0].set_title('Stopwords de nltk')
axs[0].set_axis_off()
axs[0].set_title('Stopwords de Zipf')
axs[1].imshow(wordcloud_stopwords_alt)
axs[1].set_axis_off()
plt.show()

# Lenguaje aleatorio con distribuciones Poisson, Uniforme Normal con media en cero, bi/tri normal, log normal, Zipf, 
# Usemos enlace egalitario para generar un nodo de tipo "letra" y cada rama es una palabra