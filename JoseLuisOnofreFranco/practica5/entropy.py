# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="HnGS10b-EpCP"
# # Tokenización

# %% [markdown] id="p_HVQjBuF3dm"
# ## Entropía
#
# Podemos calcular la entropía de un texto usando la entropía de Shannon, que nos indica qué tan impredecible es encontrar palabras diferentes en un corpus, lo cual nos ayuda a observar la morfología de una lengua. Podemos calcularla de esta manera:
#
# $$H(T) = -\sum_{i=1}^{|V|} p(t_i) \log_{2}(p(t_i))$$
#
# donde $T$ es un texto con palabras $V=\{t_1, t_2, ..., t_n\}$.
#
#

# %% [markdown] id="MueEuE4dH8pS"
# ## Entropía de diferentes corpus
#
# Para poder obtener la entropía de un texto, vamos a hacer uso de `scypi`.

# %% id="_mijUugxKItm"
from scipy.stats import entropy
from collections import Counter

def calculate_entropy(corpus: str) -> float:
  "Given a corpus, calculates its Shannon Entropy"

  character_count = Counter(corpus)

  total_characters = len(corpus)
  character_probabilities = [count / total_characters for count in character_count.values()]

  shannon_entropy = entropy(character_probabilities, base=2)

  return shannon_entropy


# %% [markdown] id="DI6Asm3NIl74"
# ### Entropía de un corpus en español

# %% id="Zso_XIb7IqJg"
import requests
from re import sub


# %% id="Rm52zTpQItdC"
def download_spa_bible():
  return requests.get("https://raw.githubusercontent.com/ximenina/theturningpoint/main/Detailed/corpora/corpusPBC/spa-x-bible-reinavaleracontemporanea.txt.clean.txt").text

def clean_corpus(corpus):
  """From a given corpus, erases all characters that are not letters"""
  return sub(r'[^\w\s\']', ' ', corpus).lower()


# %% id="iR96kc_MJPUG"
bible = clean_corpus(download_spa_bible())

# %% [markdown] id="6W9nGMRM37gd"
# Calculamos la entropía del texto del corpus **sin tokenizar**:

# %% colab={"base_uri": "https://localhost:8080/"} id="ufMgpySsK5h0" outputId="381d1409-a11a-459c-9b8a-068afa62bc1c"
bible_entropy = calculate_entropy(bible)
print("Entropía del corpus de la Biblia en español: ", bible_entropy)

# %% [markdown] id="pcrucoIq4GLl"
# Después, tokenizamos el texto de la biblia. Para esto, utilizamos **BPE native**, entrenado con el corpus de `cess_esp`.

# %% colab={"base_uri": "https://localhost:8080/"} id="_70g-O-2Fa6i" outputId="2dfa61da-4184-4edd-8a1a-29018d974550"
import nltk

nltk.download("cess_esp")

from nltk.corpus import cess_esp


# %% colab={"base_uri": "https://localhost:8080/"} id="QPyCER2sPOnv" outputId="b434b2af-d661-4da9-c53f-e37ffcae4085"
# To use BPE native
# !pip install subword-nmt

# %% id="eHoK7u0KiNJo"
def write_corpus(corpus: str, filename: str) -> None:
  """Writes the corpus into the filesystem"""
  with open(filename, "w") as f:
    f.write(corpus)

def read_corpus(filename: str) -> str:
  corpus = ""
  with open("nahuatl_axolotl_tokenized.txt", "r") as f:
    corpus = "".join(f.readlines())

  return corpus


# %% id="jjPB6MbtFucG"
cess_sents = cess_esp.sents()
cess_plain_text = " ".join([" ".join(sentence) for sentence in cess_sents])
cess_plain_text = sub(r"[-|_]", " ", cess_plain_text)

write_corpus(cess_plain_text, "cess_plain.txt")

# %% colab={"base_uri": "https://localhost:8080/"} id="UhSVIF-cGOxa" outputId="bea3ab2c-980e-48dd-f43a-779a0d2ed919"
# !subword-nmt learn-bpe -s 300 < cess_plain.txt > cess.model

# %% id="tVJhjJJG663i"
write_corpus(bible, "bible.txt")

# %% id="6weI-dRi7Dhs"
# !subword-nmt apply-bpe -c cess.model < bible.txt > bible_tokenized.txt

# %% colab={"base_uri": "https://localhost:8080/"} id="E9eTyC7N7LUQ" outputId="e8dc8b12-0d07-4de7-fb1a-fa9d2b450474"
bible_tokenized = read_corpus("bible_tokenized.txt")
bible_tokenized_entropy = calculate_entropy(bible_tokenized)
print("Entropía del corpus de la Biblia tokenizada en español: ", bible_tokenized_entropy)

# %% [markdown] id="CV4GQjg_MGdu"
# ### Entropía de un corpus en Nahuatl

# %% [markdown] id="IMXE9PvI_cV2"
# Primero, calculamos la entropía de un texto en náhuatl **sin tokenizar**.

# %% id="tB_Zoe2wGHXn" colab={"base_uri": "https://localhost:8080/"} outputId="d2c4b117-bdc1-47fd-e34a-5f8527ff9cbd"
# !pip install elotl

# %% colab={"base_uri": "https://localhost:8080/"} id="41B5K3vZH3dT" outputId="a4bbaffa-71f7-4995-bca1-5ab5cd123a24"
from elotl import corpus as elotl_corpus

def preprocess_elotl_corpus(corpus: list) -> tuple:
    """Separates the corpus of each language of a given
    elotl parallel corpus"""

    lang1 = []
    lang2 = []

    for row in corpus:
        lang1.append(row[0])
        lang2.append(row[1])

    corpus1 = clean_corpus(" ".join(lang1))
    corpus2 = clean_corpus(" ".join(lang2))

    return corpus1, corpus2


axolotl = elotl_corpus.load("axolotl")
_, nahuatl_axolotl = preprocess_elotl_corpus(axolotl)

axolotl_entropy = calculate_entropy(nahuatl_axolotl)
print("Entropía de axolotl en náhuatl: ", axolotl_entropy)

# %% [markdown] id="KdBZP8-0_094"
# Después, cálculamos la entropía del texto **tokenizado**. Para esto, usamos una porción del corpus para entrenar. Sin embargo, como el cálculo de la entropía sin tokenizar usamos todo el corpus, usaremos todo el corpus tokenizado para calcular la entropía.

# %% id="L3_c3iZhIHAO"
train_rows_count = len(axolotl) - round(len(axolotl)*.30)

axolotl_train = axolotl[:train_rows_count]
axolotl_words_vanilla_train = " ".join([word for row in axolotl_train for word in row[1].lower().split()])

write_corpus(axolotl_words_vanilla_train, "axolotl_plain_vanilla.txt")

# %% colab={"base_uri": "https://localhost:8080/"} id="nZP4WpvbJn82" outputId="55a40eb9-88fe-46f9-8e8d-26c266ea2f92"
# !subword-nmt learn-bpe -s 300 < axolotl_plain_vanilla.txt > axolotl_vanilla.model

# %% id="YXgHulmzhrZ1"
write_corpus(nahuatl_axolotl, "nahuatl_axolotl.txt")

# %% id="VuIJ9FxWjsVo"
# !subword-nmt apply-bpe -c axolotl_vanilla.model < nahuatl_axolotl.txt > nahuatl_axolotl_tokenized.txt

# %% id="2vCL1X6MkpIq"
with open("nahuatl_axolotl_tokenized.txt", "r") as f:
  nahuatl_tokenized = "".join(f.readlines())

# %% colab={"base_uri": "https://localhost:8080/"} id="GCBDG87IlSV-" outputId="a46d147c-0e4e-4315-93ea-a67575611149"
axolotl_tokenized_entropy = calculate_entropy(nahuatl_tokenized)
print("Entropía de axolotl tokenizado en náhuatl: ", axolotl_tokenized_entropy)

# %% [markdown] id="e-A1Ac_t8X0E"
# ## Comparando resultados

# %% colab={"base_uri": "https://localhost:8080/"} id="p7UZaI6S8YZp" outputId="89c77875-a2cc-4199-95db-ede8d02ca760"
import pandas as pd

columns = ["vanilla", "tokenizado"]
rows = ["español", "náhuatl"]

df = pd.DataFrame([[bible_entropy, bible_tokenized_entropy], [axolotl_entropy, axolotl_tokenized_entropy]], rows, columns)
print("Resultados del cálculo de la entropía: \n")
print(df)

# %% [markdown] id="S5671Fep-eyy"
# * ¿Aumento o disminuyó la entropia para los corpus?
#
#   Para los dos corpus disminuyó la entropía al tokenizar. Esto se debe a que se usó BPE, que nos ayudó a tokenizar el corpus a nivel subpalabra. Esto disminuye nuestro alfabeto, que antes consistía en todas las palabras del corpus, a uno más reducido, pero que aparece más frecuente.
#
# * ¿Qué significa que la entropia aumente o disminuya en un texto?
#
#   Nos indica la diversidad de palabras que hay en el texto, aunque no necesariamente de diferentes tipos de lexemas. Esto nos puede ayudar a conocer la morfología de las lenguas, pues las que tienen una rica morfología tienen una entropía alta. Por ejemplo, en el español se tiene: *yo corro, nosotros corremos, ellos corren*. Sin embargo, en el inglés se tiene: *I run, we run, they run*.
#
#   Aunque, si un texto suele repetir palabras, puede que también la entropía disminuya, porque es problable que nos encontremos las mismas palabras una y otra vez.
#
# * ¿Como influye la tokenizacion en la entropía de un texto?
#
#   Como lo sugerido en el primer punto, al tokenizar el texto, estamos "comprimiendo el texto". Entonces en vez de tener diversidad de palabras como `[comemos, corremos, vemos, comimos, corrimos, vimos]`, puede que el tokenizador haga lo siguiente: `[corr, com, v, emos, imos]`, lo cual obtengamos símbolos más frecuentes en el texto.
