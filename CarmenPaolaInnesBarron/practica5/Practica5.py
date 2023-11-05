import requests 
import elotl.corpus
import subword_nmt.apply_bpe as apply_bpe
import subword_nmt.learn_bpe as learn_bpe
from collections import Counter
import math
import os

def get_text_entropy(text):
    words = text.lower().split()
    text_length = len(words)
    word_freq = Counter(words)
    entropy = 0.0
    for word in word_freq:
        probability = word_freq[word] / text_length
        entropy -= probability * math.log2(probability)
    return entropy

def get_bible_corpus() -> str:
    file_name = "spa-x-bible-reinavaleracontemporanea"
    r = requests.get(f"https://raw.githubusercontent.com/ximenina/theturningpoint/main/Detailed/corpora/corpusPBC/{file_name}.txt.clean.txt")
    return r.text

def tonize_text(text: str):
    print(text)
    return

def write_plain_text_corpus(raw_text: str, file_name: str) -> None:
    with open(f"{file_name}.txt", "w") as f:
        f.write(raw_text)

def tokenize_with_bpe(text, bpe_model):
    bpe = apply_bpe.BPE(codes=open(bpe_model))
    tokenized_text = bpe.process_line(text)
    
    return tokenized_text

# nombres de los archivos
bible_file_name= 'bible_plain_text.txt'
nahuatl_file_name = 'nahuatl_plain_text.txt'
bible_model_filename = 'bible.model'
nahuatl_model_filename = 'nahuatl.model'

if os.path.exists(bible_file_name):
    with open(bible_file_name, 'r', encoding='utf-8') as file:
        bible_text = file.read()
else:
    bible_text = get_bible_corpus().replace("\n", " ").lower()
    with open(bible_file_name, 'w', encoding='utf-8') as file:
        file.write(bible_text)

if os.path.exists(nahuatl_file_name):
    with open(nahuatl_file_name, 'r', encoding='utf-8') as file:
        nahuatl_text = file.read()
else:
    nahuatl_corpus = elotl.corpus.load("axolotl")
    nahuatl_text = "".join([word for row in nahuatl_corpus for word in row[1].lower()]) #para que sea una paabra y no una lista, así lo guardamos

    with open(nahuatl_file_name, 'w', encoding='utf-8') as file:
        file.write(nahuatl_text)

if not os.path.exists(bible_model_filename):
    print("Entrenando modelo para el español")
    learn_bpe.learn_bpe(open(bible_file_name), open(bible_model_filename, "w"), num_symbols = 80)
    bible_tokenized = tokenize_with_bpe(bible_text, bible_model_filename)
else:
    bible_tokenized = tokenize_with_bpe(bible_text, bible_model_filename)

if not os.path.exists(nahuatl_model_filename):
    print("Entrenando modelo para el nahuatl")
    learn_bpe.learn_bpe(open(nahuatl_file_name), open(nahuatl_model_filename, "w"), num_symbols = 200)
    nahuatl_tokenized = tokenize_with_bpe(nahuatl_text, nahuatl_model_filename)
else:
    nahuatl_tokenized = tokenize_with_bpe(nahuatl_text, nahuatl_model_filename)

# obtener entropía
esp_entropy = get_text_entropy(bible_text)
esp_tokenized_entropy = get_text_entropy(bible_tokenized)
nah_entropy = get_text_entropy(nahuatl_text)
nah_tokenized_entropy = get_text_entropy(nahuatl_tokenized)
print(f"Entropía del español (sin tokenizar): {esp_entropy:.2f}")
print(f"Entropía del nahuatl (sin tokenizar): {nah_entropy:.2f}")
print("------------------------------------------------")
print(f"Entropía del español (tokenizado): {esp_tokenized_entropy:.2f}")
print(f"Entropía del nahuatl (tokenizado): {nah_tokenized_entropy:.2f}")