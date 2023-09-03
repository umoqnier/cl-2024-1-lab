#!/usr/bin/env python
# coding: utf-8

# # Práctica 2: Análisis morfológico (parte 1)

# Obtener 10 oraciones al azar del conjunto de pruebas del corpus SIGMORPH 2022
# 
# - Track: Sentences
# - Guardarlas en un objeto pandas [OPCIONAL]


import random
import requests
import pandas as pd
from nltk.stem.snowball import EnglishStemmer
from nltk.tokenize import wordpunct_tokenize
import spacy
from tabulate import tabulate


r = requests.get("https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/spa.word.test.gold.tsv")

def get_files(lang: str, track: str = "word") -> list[str]:
    """Genera una lista de nombres de archivo basados en el idioma y el track

    Parameters:
    ----------
    lang : str
        Idioma para el cual se generarán los nombres de archivo.
    track : str, optional
        Track del shared task de donde vienen los datos (por defecto es "word").

    Returns:
    -------
    list of str
        Una lista de nombres de archivo generados para el idioma y la pista especificados.
    """
    return [
        f"{lang}.{track}.test.gold",
        f"{lang}.{track}.dev",
    ]

def get_raw_corpus(files: list) -> list:
    """Descarga y concatena los datos de los archivos tsv desde una URL base.

    Parameters:
    ----------
    files : list
        Lista de nombres de archivos (sin extensión) que se descargarán
        y concatenarán.

    Returns:
    -------
    list
        Una lista que contiene los contenidos descargados y concatenados
        de los archivos tsv.
    """
    result = []
    for file in files:
        print(f"Downloading {file}.tsv")
        r = requests.get(f"https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/{file}.tsv")
        response_list = r.text.split("\n")
        result.extend(response_list[:-1])
    return result


# Nota: el único idioma en la intersección fue el inglés

def get_10_random_sentences():
    files = get_files("eng", "sentence")
    print("files", files)
    raw = get_raw_corpus([files[1]])
    random_lines = random.sample(raw, 10)
    sentences = []
    for line in random_lines:
        sentence, tagged = line.split("\t")
        sentences.append(sentence)
    return sentences


# Usar las bibliotecas spacy y nltk para realizar los siguientes procesos:
# - Stemming (nltk)
# - Lemmatization (spacy)
# - Obtención de información morfologica (spacy)
# - Imprimir la información en pantalla (formato libre)


# Para que funcione lo siguiente, se sebe tener descargado el modelo en_core_web_sm de spacy.
# Lo puedes descargar ejecutando lo soguiente (en el notebook debe ejecutarse en raw)
# !python -m spacy download en_core_web_sm


nlp = spacy.load('en_core_web_sm')

def get_word_analysis(word):
    word_info = {}
    word_info['word'] = word
    stemmer = EnglishStemmer()
    word_info['stem'] = stemmer.stem(word)
    tokens = nlp(word)
    word_info['lemma'] = tokens[0].lemma_
    word_info['morphological_info'] = tokens[0].morph.to_dict()
    return word_info

def get_sentence_analysis(sentence):
    words = wordpunct_tokenize(sentence)
    words_info = []
    for word in words:
        words_info.append(get_word_analysis(word))
    return words_info


# Imprimir toda la información:
sentences = get_10_random_sentences()

for sent in sentences:
    df_sent = pd.DataFrame(get_sentence_analysis(sent))
    print("Sentence:", sent)
    print(tabulate(df_sent, headers='keys', tablefmt='psql', showindex=False))

