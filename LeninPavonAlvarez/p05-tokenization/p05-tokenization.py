import os # Comandos
import math # Cálculo de la entropía
import requests # Obtención de corpus
# NLP
from elotl import corpus as elotl_corpus
from re import sub
from collections import Counter

from rich import print
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

# Permite impresión de tablas 
console = Console()
layout = Layout()
BIBLE_FILE_NAMES = {"spa": "spa-x-bible-reinavaleracontemporanea", "eng": "eng-x-bible-kingjames"}

def get_bible_corpus(lang: str) -> str:
    file_name = BIBLE_FILE_NAMES[lang]
    r = requests.get(f"https://raw.githubusercontent.com/ximenina/theturningpoint/main/Detailed/corpora/corpusPBC/{file_name}.txt.clean.txt")
    return r.text

def write_plain_text_corpus(raw_text: str, file_name: str) -> None:
    with open(f"{file_name}.txt", "w") as f:
        f.write(raw_text)

# TODO: Entropía de texto
def entropy_of_text(probabilities:list, frequencies:list) -> float:
    """Calcula la entropía de Shannon de un texto"""
    total = 0
    for idx,p in enumerate(probabilities):
        total -= p*math.log(p,2)*frequencies[idx]
    return total

def entropy_of_vocab(probabilities:list) -> float:
    """Calcula la entropía de Shannon de un texto"""
    total = 0
    for p in probabilities:
        total -= p*math.log(p,2)
    return total

def tokenize_BPE_native(corpus: list, filename: str, n:int=300) -> None:
    os.system(f"subword-nmt learn-bpe -s {n} < {corpus}.txt > {filename}.model")
    return

def apply_BPE_model(stdin_name: str, model_name: str, stdout_name: str) -> None:
    os.system(f"subword-nmt apply-bpe -c {model_name}.model < {stdin_name}.txt > {stdout_name}-tokenized.txt")

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

def freqs_from_corpus(corpus: list[str]) -> (Counter,dict):
    corpus_lower = [word.lower() for word in corpus]
    freqs = Counter(corpus_lower)
    normalization = sum(freqs.values())
    freqs_rel = {key:value/normalization for (key,value) in freqs.items()}
    return freqs,freqs_rel

def write_plain_text_corpus(raw_text: str, file_name: str) -> None:
    with open(f"{file_name}.txt", "w") as f:
        f.write(raw_text)


texto_presentación = """Práctica 5. Tokenización
Lenin Pavón Alvarez"""
print(Panel(Text(texto_presentación,style="bold magenta",justify="center"), border_style="magenta"))
# TODO: Medir entropía de biblia, axolotl 
print(Panel(Text("Parte 1: Entropía",style="bold cyan",justify="center"), border_style="cyan"))
texto_p1 = "Vamos a usar la entropía de Shannon para obtener la"+\
    "entropía del vocabulario sumando [bold cyan]p_i * math.log(p_i,2)[/bold cyan] "+\
    "y la entropía del texto multiplicando por las frecuencias de cada tipo."
print(texto_p1)
# Entropía
# Obteniendo corpus de la biblia
print("Obteniendo corpora")
spa_bible_plain_text = get_bible_corpus('spa')
spa_bible_words = spa_bible_plain_text.replace("\n", " ").split()
# Obteniendo corpus de axolotl
axolotl = elotl_corpus.load("axolotl")
spanish_words_na, nahuatl_words = preprocess_corpus_words(axolotl)
# Procesamiento de corpus
print("Procesando corpora")
spa_bible_freqs,spa_bible_freqs_rel = freqs_from_corpus(spa_bible_words)
axolotl_freqs,axolotl_freqs_rel = freqs_from_corpus(nahuatl_words)
axolotl_spa_freqs,axolotl_spa_freqs_rel = freqs_from_corpus(spanish_words_na)
print("Calculando entropía")
spa_bible_entropia = entropy_of_text(spa_bible_freqs_rel.values(),list(spa_bible_freqs.values()))
axolotl_entropia = entropy_of_text(axolotl_freqs_rel.values(),list(axolotl_freqs.values()))
axolotl_spa_entropia = entropy_of_text(axolotl_spa_freqs_rel.values(),list(axolotl_spa_freqs.values()))

spa_bible_entropia_vocab = entropy_of_vocab(spa_bible_freqs_rel.values())
axolotl_entropia_vocab = entropy_of_vocab(axolotl_freqs_rel.values())
axolotl_spa_entropia_vocab = entropy_of_vocab(axolotl_spa_freqs_rel.values())
# Corpus en texto plano
axolotl_plain_text = "\n".join([sentences[1] for sentences in axolotl])
axolotl_spa_plain_text = "\n".join([sentences[0] for sentences in axolotl])
# Tokenización BPE
filenames = ["spa-bible","nah-axolotl","spa-axolotl"]
corpora = [spa_bible_plain_text, axolotl_plain_text, axolotl_spa_plain_text]
entropia_token = []
entropia_vocab_token = []
# Entrenando modelos de BPE
print("Tokenizando corpus con BPE nativo")
for idx,name in enumerate(filenames):
    write_plain_text_corpus(corpora[idx], name)
    tokenize_BPE_native(name, name)
    apply_BPE_model(name, name, name)
    tokenized_path = name + "-tokenized.txt"
    with open(tokenized_path) as f:
        file_tokenized = f.read().split()
    freq, freq_rel = freqs_from_corpus(file_tokenized)
    entropia_token.append(entropy_of_text(freq_rel.values(),list(freq.values())))
    entropia_vocab_token.append(entropy_of_vocab(freq_rel.values()))


# TODO: Visualizar entropía
corpora_names = filenames
bpe = ["NO", "NO", "NO", "SÍ", "SÍ", "SÍ"]
entropia = [spa_bible_entropia,axolotl_entropia, axolotl_spa_entropia] + entropia_token
entropia_vocab = [spa_bible_entropia_vocab, axolotl_entropia_vocab, axolotl_spa_entropia_vocab]+ entropia_vocab_token
# Preparación de tabla
table = Table(title = "Entropía")
table.add_column("Corpus", style ="cyan")
table.add_column("BPE", style ="green")
table.add_column("Entropía (texto)", style="magenta")
table.add_column("Entropía (vocabulario)", style="magenta")
for idx, elem in enumerate(corpora_names):
    table.add_row(corpora_names[idx],bpe[idx],str(entropia[idx]), str(entropia_vocab[idx]))
    table.add_row(corpora_names[idx],bpe[idx+3],str(entropia[idx+3]), str(entropia_vocab[idx+3]))
# Impresión de tabla
console.print(table)
print(Panel(Text("Parte 2: Insights",style="bold cyan",justify="center"), border_style="cyan"))
print("1. Veamos que en todos otros corpus [red]aumentó[/red] "+\
    "la entropía del texto.")
print("2. Pensemos en el valor de entropía"+\
    "como la cantidad de información que provee cada"+\
    "símbolo. Cuando sólo tenemos un símbolo, la entropía "+\
    "vale cero, porque el símbolo sale con probabilidad 1. "+\
    "Cuando tenemos más símbolos, menor probabilidad de que "+\
    "salga y por ende mayor información aporta cada símbolo. "+\
    "Al disminuir la cantidad de símbolos totales, aumentamos "+\
    "la probabilidad de que cualquier tipo en particular "+\
    "aparezca y disminuimos la entropía del texto. En cambio, "+\
    "al aumentar la cantidad de tipos, disminuimos las "+\
    "probabilidades de que aparezca cualquier tipo en particular, "+\
    "y por tanto aumenta la entropía del texto."
    )
print("3. La tokenización va a disminuir la entropía (vocabulario) si se "+\
    "aumenta la cantidad de tipos. "+\
    "En cambio si aumenta la frecuencia de los "+\
    "tokens, va a aumentar la entropía (texto), que va a pasar más notablemente en "+\
    "lenguajes aglutinantes puesto que suelen tener más elementos "+\
    "con frecuencia 1."
      )
# tokenización?
# TODO: ¿Tokenización >,=,< entropía?

# ---
# TODO: Normalizar Náhuatl
# Modelo con texto normalizado usando BPE Native
# 