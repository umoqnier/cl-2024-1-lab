import spacy
import nltk
import requests
import pandas as pd
from nltk.tokenize import *
from nltk.stem.snowball import *
nltk.download('punkt')


pd.set_option('display.max_colwidth', None)

LANGS = {
    #"ces": "Czech",
    "eng": "English",
    #"fra": "French",
    #"hun": "Hungarian",
    #"spa": "Spanish",
    #"ita": "Italian",
    #"lat": "Latin",
    #"rus": "Russian",
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
        #f"{lang}.{track}.test.gold",
        f"{lang}.{track}.test",
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

def raw_corpus_to_dataframe_sentences(corpus_list: list, lang: str) -> pd.DataFrame:
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
            sentence, morph_seg = line.split("\t")
        except ValueError:
            # Caso donde no existe la categoria
            sentence = line
            morph_seg = "N/D"
        #words = line.split()
        data_list.append({"sentence": sentence, "morph-seg": morph_seg, "lang": lang})
    df = pd.DataFrame(data_list)
    df["sen_len"] = df["sentence"].apply(lambda x: len(x))
    #df["stem_len"] = df["stems"].apply(lambda x: len(x))
    #df["morph_len"] = df["morph"].apply(lambda x: len(x))
    return df


def get_corpora() -> dict :
    corpora = {}
    for lang in LANGS : corpora[lang] =  raw_corpus_to_dataframe_sentences( get_raw_corpus(get_files(lang,"sentence")) ,lang=lang)
    return corpora

def get_sentence_list(corpora : dict , lang : str, count : int) -> [str]:
    lst = []
    for i in range(count) : lst.append (corpora[lang].sample(1)["sentence"].to_string(index=False))
    return lst

def get_stems(lang : str,sentence : str) -> [str]:
    lst = []
    stemmer = SnowballStemmer(LANGS[lang].lower())
    words = word_tokenize(sentence)
    for word in words : lst.append(stemmer.stem(word))
    return lst

def get_lemmas(lang: str, text :str) -> [str]:
    doc =  nlp(text)
    lst = []
    for stem in doc : lst.append(stem.lemma_)
    return lst

def get_analysis(lang : str, sentence :str) -> []:
    doc = nlp(sentence)
    return [(w.text, w.morph) for w in doc]


N = 10
corpora = get_corpora()

print("Analizador Morfológico de oraciones")

print(f"Lenguas disponibles: {(LANGS)}")

lang = input("lang>> ")
print(f"Selected language: {LANGS[lang]}") if lang else print("Adios 👋🏼")
match lang:
    case "eng":
        nlp = spacy.load("en_core_web_sm")
        import en_core_web_sm
        nlp = en_core_web_sm.load()

while lang:
    print("recuperando oraciones de prueba...\n")
    sentences = get_sentence_list(corpora,lang,N)
    opt = " "
    while True and opt:  
        try:
            print("Oraciones disponibles para analizar:")
            for i in range(N) : print("    "+str(i+1)+". "+ sentences[i])
            opt = input("\nPor favor, selecciona alguna para continuar[1-10]>> ")
            n = int(opt)-1
            if (n > 0 and n < 10):
                text = sentences[n]
                stems = get_stems(lang,text)
                lemmas = get_lemmas(lang, text)
                analysis = get_analysis(lang, text)
                print("\n"+text+"\n\nAnálisis Morfológico de la oración")
                for i in range(len(stems)): 
                    print(str(analysis[i]) + " \t Raíz: '" + stems[i] + "', \t Lema: '" + lemmas[i]+"'")
                print("\n")
                input("continuar...>> ")
        except ValueError:
            continue
    lang = input("lang>> ")
    print(f"Selected language: {lang_codes[lang]}") if lang else print("Adios 👋🏼")