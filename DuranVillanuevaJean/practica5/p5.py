"""
pip install elotl
pip install requests
pip install subword_nmt
"""

#imports
import requests
import elotl.corpus
import subword_nmt.apply_bpe as apply_bpe
import subword_nmt.learn_bpe as learn_bpe
from collections import Counter
import math



#Creando archivos y carpetas
import os
if not os.path.exists("corpus"): 
    os.makedirs("corpus")
    #bible
bible_corpus_dir = os.path.join("corpus","bible_corpus.txt")
bible_corpus_tokenize_dir = os.path.join("corpus","bible_corpus_tokenize.txt")
    #nahuatl
nahuatl_corpus_dir = os.path.join("corpus","nahuatl_corpus.txt")
nahuatl_corpus_tokenize_dir = os.path.join("corpus","nahuatl_corpus_tokenize.txt")



#Obteniendo cada corpus
    #bible
bible_corpus = requests.get(f"https://raw.githubusercontent.com/ximenina/theturningpoint/main/Detailed/corpora/corpusPBC/spa-x-bible-reinavaleracontemporanea.txt.clean.txt")
bible_corpus = bible_corpus.text
bible_corpus = bible_corpus.replace("\n", " ").lower()
with open(bible_corpus_dir, 'w', encoding='utf-8') as file:
    file.write(bible_corpus)
    #nahuatl
nahuatl_corpus = elotl.corpus.load("axolotl")
nahuatl_corpus = "".join([word for row in nahuatl_corpus for word in row[1].lower()])
# axolotl_words_vanilla_train = [word for row in axolotl_train for word in row[1].lower().split()] #diego
with open(nahuatl_corpus_dir, 'w', encoding='utf-8') as file:
    file.write(nahuatl_corpus)



#Entrenando y aplicando el modelo
def tokenize_learn(corpus,corpus_tokenize,num_symbols):
    learn_bpe.learn_bpe(open(corpus), open(corpus_tokenize, "w"), num_symbols = num_symbols)
    
def tokenize_apply(corpus,corpus_tokenize):
    bpe = apply_bpe.BPE(codes=open(corpus_tokenize))
    return bpe.process_line(corpus)

    #para el tokenize
        #bible
tokenize_learn(bible_corpus_dir,bible_corpus_tokenize_dir,250)
b_t = tokenize_apply(bible_corpus_dir,bible_corpus_tokenize_dir)
        #nahuatl
tokenize_learn(nahuatl_corpus_dir,nahuatl_corpus_tokenize_dir,400)
n_t = tokenize_apply(nahuatl_corpus_dir,nahuatl_corpus_tokenize_dir)



#Formula para obtener la entropia
def get_text_entropy(corpus):
    text = corpus.split()
    # Count the frequency of each character in the text
    char_counts = Counter(text)

    # Calculate the probability of each character
    total_chars = len(text)
    probabilities = [count / total_chars for count in char_counts.values()]

    # Calculate entropy using the formula: H(X) = -Σ P(x) * log2(P(x))
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

    return entropy



#Obteniendo la entropia
print("<<<<<<<<<Entropía>>>>>>>>>\n")
    #español
    #Entropia del corpus crudo y del corpus tokenizado
print("Español - la biblia (sin tokenizar): ",get_text_entropy( open(bible_corpus_dir,'r').read()))
print("-vs-")
print("Español - la biblia (tokenizado): ",get_text_entropy(b_t))
print()

    #nahuatl
    #Entropia del corpus crudo y del corpus tokenizado
print("Nahuatl (sin tokenizar): ",get_text_entropy( open(nahuatl_corpus_dir,'r').read()))
print("-vs-")
print("Nahuatl (tokenizado): ",get_text_entropy(n_t))