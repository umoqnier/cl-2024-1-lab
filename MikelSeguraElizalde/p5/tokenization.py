# Autor: Mikel Segura Elizalde
# Versión 1, octubre 2023

import nltk
import elotl.corpus
import requests
import math
import os
import re
from collections import Counter
from nltk.corpus import cess_esp as cess
axolotl = elotl.corpus.load("axolotl")
nltk.download("cess_esp")

cess_sents = cess.sents()
cess_words = cess.words()
cess_plain_text = " ".join([" ".join(sentence) for sentence in cess_sents])
cess_plain_text = re.sub(r"[-|_]", " ", cess_plain_text)

with open("cess_plain.txt", "w") as f:
    f.write(cess_plain_text)
os.system("subword-nmt learn-bpe -s 300 <cess_plain.txt> cess.model")

BIBLE_FILE_NAMES = {"spa": "spa-x-bible-reinavaleracontemporanea", "eng": "eng-x-bible-kingjames"}

def get_bible_corpus(lang: str) -> str:
    file_name = BIBLE_FILE_NAMES[lang]
    r = requests.get(f"https://raw.githubusercontent.com/ximenina/theturningpoint/main/Detailed/corpora/corpusPBC/{file_name}.txt.clean.txt")
    return r.text

def write_plain_text_corpus(raw_text: str, file_name: str) -> None:
    with open(f"{file_name}.txt", "w") as f:
        f.write(raw_text)

spa_bible_plain_text = get_bible_corpus('spa')
spa_bible_words = spa_bible_plain_text.replace("\n", " ").split()

def entropy(list_of_words):
  result = 0
  length = len(list_of_words)
  types = Counter(list_of_words)
  for word in types:
    probability = types[word]/length
    result += probability*math.log2(probability)
  return -result

write_plain_text_corpus(spa_bible_plain_text, "spa-bible")
os.system("subword-nmt apply-bpe -c cess.model <spa-bible.txt> spa_bible_tokenized.txt")

with open("spa_bible_tokenized.txt", "r") as f:
    tokenized_text = f.read()
spa_bible_tokenized = tokenized_text.split()

train_rows_count = len(axolotl) - round(len(axolotl)*.30)
axolotl_train = axolotl[:train_rows_count]
axolotl_test = axolotl[train_rows_count:]
axolotl_words_vanilla_train = [word for row in axolotl_train for word in row[1].lower().split()]

write_plain_text_corpus(" ".join(axolotl_words_vanilla_train), "axolotl_plain_vanilla")
os.system("subword-nmt learn-bpe -s 300 <axolotl_plain_vanilla.txt> axolotl_vanilla.model")

axolotl_test_words = [word for row in axolotl_test for word in row[1].lower().split()]

write_plain_text_corpus(" ".join(axolotl_test_words), "axolotl_plain_test")
os.system("subword-nmt apply-bpe -c axolotl_vanilla.model <axolotl_plain_test.txt> axolotl_vanilla_tokenized.txt")

with open("axolotl_vanilla_tokenized.txt") as f:
    axolotl_test_tokenized = f.read().split()

print(f"entropía de La Biblia en español sin tokenizar: {entropy(spa_bible_words)}")
print(f"entropía de La Biblia en español tokenizada: {entropy(spa_bible_tokenized)}")
print(f"entropía de Axolotl sin tokenizar: {entropy(axolotl_test_words)}")
print(f"entropía de Axolotl tokenizada: {entropy(axolotl_test_tokenized)}")

input("presione enter para salir")