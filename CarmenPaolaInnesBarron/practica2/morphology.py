import requests
import random
import spacy 
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

MODELS = {
    "spa": "es_core_news_sm",
    "eng": "en_core_web_sm"
}

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


def extract_morpho_info(sentence, nlp, stemmer):
    """
    Extrae el análisis morfológico y el lema de las palabras en una oración

    Parameters:
    ----------
    sentence : string
        Oración para hacer el análisis
    
    nlp : object
        modelo que se debe usar para el análisis

    Returns:
    -------
    void
        Imprime el análisis y el lema por palabra
    """
    doc = nlp(sentence)
    for token in doc:
        print(f"-> palabra: {token}")
        print("-> análisis:", token.morph.to_dict())
        print(f"-> lema: {token.lemma_}")
        print(f"-> stem: {stemmer.stem(token.text)}" )
        print("____________")

# obtenemos el corpus con las oraciones (la data)
files = get_files('eng', 'sentence')
corpus = get_raw_corpus(files=files)
total_sentences = 1
index = random.sample(range(0, len(corpus)), total_sentences) # obtener los índices de 10 oraciones aleatorias

# declaramos el modelo de spacy para inglés
nlp_en = spacy.load(MODELS["eng"])

cont = 10
ps = PorterStemmer()

for i in index:
    # de cada oración vamos hacer el análisis
    sentence = corpus[i].split("\t")[0] # obtenemos la oración
    print(f"---- Oración n.{cont} ----")
    print(f'"{sentence}"')
    print("*** Análisis morfológico, lematización y stemming:")
    extract_morpho_info(sentence=sentence, nlp=nlp_en, stemmer=ps)
    cont += 1