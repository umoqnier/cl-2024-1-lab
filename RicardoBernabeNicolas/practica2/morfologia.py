
import requests
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import spacy

LANGS = {
    "ces": "Czech",
    "eng": "English",
    "fra": "French",
    "hun": "Hungarian",
    "spa": "Spanish",
    "ita": "Italian",
    "lat": "Latin",
    "rus": "Russian",
}
CATEGORIES = {
    "100": "Inflection",
    "010": "Derivation",
    "101": "Inflection, Compound",
    "000": "Steam",
    "011": "Derivation, Compound",
    "110": "Inflection, Derivation",
    "001": "Compound",
    "111": "Inflection, Derivation, Compound"
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
        #print(f"Downloading {file}.tsv")   
        r = requests.get(f"https://raw.githubusercontent.com/sigmorphon/2022SegmentationST/main/data/{file}.tsv")
        response_list = r.text.split("\n")
        result.extend(response_list[:-1])
    return result

def raw_corpus_to_dataframe(corpus_list: list, lang: str) -> pd.DataFrame:
    """Convierte una lista de datos de corpus en un DataFrame

    Parameters:
    ----------
    corpus_list : list
        Lista de líneas del corpus a convertir en DataFrame.
    lang : str
        Idioma al que pertenecen los datos del corpus.

    Returns:
    -------
    pd.DataFrame
        Un DataFrame de pandas que contiene los datos del corpus procesados.
    """
    data_list = []
    for line in corpus_list:
        try:
            word, tagged_data, category = line.split("\t")
        except ValueError:
            # Caso donde no existe la categoria
            word, tagged_data = line.split("\t")
            category = "NOT_FOUND"
        morphemes = tagged_data.split()
        stem = morphemes[0]
        data_list.append({"words": word, "stems": stem, "morph": morphemes, "category": category, "lang": lang})
    df = pd.DataFrame(data_list)
    df["word_len"] = df["words"].apply(lambda x: len(x))
    df["stem_len"] = df["stems"].apply(lambda x: len(x))
    df["morph_len"] = df["morph"].apply(lambda x: len(x))
    return df



#Ejercicio 1 



#Obtenemos el corupus en inglés y vamos a hacer el análisis en inglés

print("\n""\n")

files = get_files("eng","sentence")
raw_eng = get_raw_corpus(files)
df = raw_corpus_to_dataframe(raw_eng, lang="eng")

sentences_raw = df.sample(n = 10)                   #Obtenmos las 10 oraciones en formato pandas


print("DATASET FRAGMENT")
print(sentences_raw) 
print("===============================================================================================================================")



sentences = sentences_raw.iloc[:,0].tolist()        #Sacamos sólo las oraciones



ps = PorterStemmer()                                #Cargamos el stemmer
nlp = spacy.load("en_core_web_sm")


def sentence_stemmer(sentence):    
    """Toma una sentencia y de cada palabra, obtiene su stem

    Parameters:
    ----------
    sentence : string
        sentencia a analizar    
    Returns:
    -------
    words : list
        lista de los stems de las palabras
    
    """


    words = word_tokenize(sentence)
    for w in words:
        print(w, " : ", ps.stem(w))  
    return words  

def sentence_lemma(sentence):
    """Toma una sentencia y la lematiza

    Parameters:
    ----------
    sentence : string
        sentencia a lematizar   
    Returns:
    -------
    final_string : string
        oración lematizada  
    """
    doc = nlp(sentence)
    empty_list = []
    for token in doc:
        empty_list.append(token.lemma_)
    final_string = ' '.join(map(str,empty_list))
    print("Lemmatizated sentence: ",final_string)
    return final_string

def analize_sentence(sentence):
    """Obtiene los stems y analiza una sentencia

    Parameters:
    ----------
    sentence : string
        sentencia a analizar     
    """

    print("\n")
    print("Sentence: ",sentence)
    print("===================================================================================================") 
    print("Words stems")       
    sentence_stemmer(sentence)
    print("===================================================================================================") 
    sentence_lemma(sentence)
    print("\n")
    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

    

for sentence in sentences:
    analize_sentence(sentence)





