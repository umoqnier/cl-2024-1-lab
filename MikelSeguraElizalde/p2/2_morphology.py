#Autor: Mikel Segura Elizalde
#Número de cuenta: 420004231
#Versión: 1; Septiembre 3, 2023

import requests
import spacy
import random
from nltk.stem import *

stemmer = PorterStemmer()
nlp = spacy.load("en_core_web_sm")
r = requests.get("https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/eng.sentence.test.gold.tsv")
raw_data = r.text.split("\n")

nueva_muestra = ""
while not nueva_muestra:
  sample_sentences = []
  for element in random.sample(raw_data, 10):
    sample_sentences.append(element.split("\t"))

  for element in sample_sentences:
    print(element[0])
    for token in nlp(element[0]):
      print(f"  -  {token}  <{token.pos_} {token.morph}>  STEM <{stemmer.stem(token.text)}>  LEMMA <{token.lemma_}>")
  nueva_muestra = input(">Presiona enter para generar 10 oraciones más.")