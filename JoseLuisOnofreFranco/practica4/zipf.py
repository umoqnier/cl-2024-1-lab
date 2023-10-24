# # Ley de Zipf

# Dependencias a utilizar:

import nltk
import numpy as np
import requests
import pandas as pd
from elotl import corpus as elotl_corpus
from util import *

# ## 1. Obtener las frecuencias a otros niveles de las lenguas (español, nahúatl y otomí) y comprobar si se cumple zipf

# ### 1.1 Para las etiquetas POS del español e inglés
#

# +
nltk.download("brown")
nltk.download("cess_esp")

from nltk.corpus import brown
from nltk.corpus import cess_esp

en_corpus = brown.tagged_sents()
es_corpus = cess_esp.tagged_sents()

en_pos = extract_pos_tags(en_corpus)
es_pos = extract_pos_tags(es_corpus)

most_common = 250
en_frequencies = get_frequencies(en_pos, most_common)
es_frequencies = get_frequencies(es_pos, most_common)
# -

# Primero se obtuvieron las frecuencias de los etiquetas del idioma **inglés**. Se puede apreciar que al graficar las frecuencias, la forma de la gráfica se asemeja a **Zipf**.

plot_frequencies(en_frequencies, title="Frequencies of english POS tags")

# Después se graficó con *log-log*, teniendo un $\alpha = 1.5$. La desviación no es tan pronunciada respecto a **Zipf**

plot_log_with_zipf(en_frequencies, label="English")

# Lo mismo se realizó con las etiquetas POS para el idioma **español**.

plot_frequencies(es_frequencies, title="Frequencies of Spanish POS tags")

# Al graficar con *log-log*, se aprecia una menor desviación que respecto al inglés, aunque esto puede variar según el corpus y el valor de $\alpha$.

plot_log_with_zipf(es_frequencies, label="Spanish")

# ### 1.2 Caractéres de las lenguas Otomí, Náhuatl y Español

# En esta sección se graficó la frecuencia de cada carácter de las palabras que aparecen en los corpus de *elotl*. Se observa que en estas no se asemeja tanto a Zipf, aunque puede ser debido a la existencia de los pocos carácteres que representan cada lengua, y puede variar.

# +
axolotl = elotl_corpus.load("axolotl")
tsunkua = elotl_corpus.load("tsunkua")

spanish_words_na, nahuatl_words = preprocess_corpus(axolotl)
spanish_words_oto, otomi_words = preprocess_corpus(tsunkua)

# +
es_char_freqs = get_character_frequencies(spanish_words_na, most_common=50)
na_char_freqs = get_character_frequencies(nahuatl_words, most_common=50)
zipf_freqs = ("Zipf", generate_zipf_frequencies(50, N=600000))


freqs = [ ("Spanish character frequencies", es_char_freqs), (" Náhuatl character frequencies", na_char_freqs)]
plot_list_frequencies(freqs + [zipf_freqs])
# -

plot_log_with_list_zipf(freqs)

# +
es_char_freqs = get_character_frequencies(spanish_words_oto, most_common=45)
oto_char_freqs = get_character_frequencies(otomi_words, most_common=45)
zipf_freqs = ("Zipf", generate_zipf_frequencies(45))

freqs = [ ("Spanish character frequencies", es_char_freqs), 
         (" Otomi character frequencies", oto_char_freqs)]
plot_list_frequencies(freqs + [zipf_freqs])
# -

plot_log_with_list_zipf(freqs)

# ### 1.3 n-gramas de caractéres (n=2)

# Se obtuvo como resultado gráficas donde al graficar las secuencias se asemejan a **Zipf**.

# +
es_ngram_freqs = get_ngrams_frequencies(spanish_words_oto, n=2, most_common=300)
oto_ngram_freqs = get_ngrams_frequencies(otomi_words, n=2, most_common=300)

freqs = [ ("Spanish 2-grams frequencies", es_ngram_freqs), (" Otomi 2-grams frequencies", oto_ngram_freqs)]
plot_list_frequencies(freqs)
# -

plot_log_with_list_zipf(freqs)

na_ngram_freqs = get_ngrams_frequencies(nahuatl_words, n=2, most_common=300)
plot_frequencies(na_ngram_freqs, title="Frequencies of Náhualt 2-grams")

plot_log_with_zipf(na_ngram_freqs, label="Náhualt 2-grams")

# ## 2. Stopwords y Zipf

# Para generar las nubes de palabras, se utilizó las stopwords de *nltk* y las frecuencias de las palabras en español del corpus obtenido de *elotl*.

# +
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('spanish')[:10]

counter = Counter(spanish_words_na)
most_common_words = [ word for word, _ in counter.most_common(10) ]
# -

plot_wordcloud(stop_words)

plot_wordcloud(most_common_words)

# Se obtienen resultados similares, aunque pueden variar el orden en el que son más frecuentes. Sin embargo, hay que considerar que entre más se amplia el número, por ejemplo, tomar los 50 más frecuentes, se pueden obtener otros resultados, pues depende del corpus donde se obtienen estas frecuencias. Por ejemplo, si se amplia a los 50 más frecuentes se obtiene:

stop_words = stopwords.words('spanish')[:50]
plot_wordcloud(stop_words)

counter = Counter(spanish_words_na)
most_common_words = [ word for word, _ in counter.most_common(50) ]
plot_wordcloud(most_common_words)

# En la primer nube de palabras se compone de determinantes, pero en la segunda hay palabras como *año, señor y tierra* que denotan un posible sesgo de donde se obtuvieron estas frecuencias. Sin embargo, las dos nubes comparten en su mayoría las palabras, que se puede demostrar obteniendo la intersección de ambas listas. Esto da un punto de partida para que empíricamente se pueda eliminar las stopwords si es necesario.

set1 = set(most_common_words)
set2 = set(stop_words)
len(set1 & set2)

