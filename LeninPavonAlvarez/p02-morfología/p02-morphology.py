"""
Práctica 02 <<Morfología>>


"""
# Instalar en_core_web_sm
# Bibliotecas necesarias
import nltk
import ssl
try:
    nltk.data.find('tokenizers/punkt.zip')
except:
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.download('punkt')
import spacy
import requests
import pandas as pd
from rich.console import Console
from rich.table import Table
# Lenguajes que podemos trabajar
"""
Hay 3 lenguajes de los que podemos escoger: ces, eng, y mon.
Ni spacy ni nltk tienen mongol, y nltk no cuenta con checo.
El único lenguaje en el que podemos trabajar es inglés.
"""
LANGS = {
    "eng": "English"
}
# Código de ayudantía modificado
def get_files(lang: str, track: str = "sentence") -> list[str]:
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
            sentence, parsed_sentence= line.split("\t")
        except ValueError:
            # Caso donde no existe la categoría
            sentence, parsed_sentence = line.split("\t")
            # category = "NOT_FOUND"
        # morphemes = tagged_data.split()
        # stem = morphemes[0]
        data_list.append({"sentence": sentence, "parsed_sentence": parsed_sentence, "lang": lang})
    df = pd.DataFrame(data_list)
    df["sentence_len"] = df["sentence"].apply(lambda x: len(x.split()))
    # df["stem_len"] = df["stems"].apply(lambda x: len(x))
    # df["morph_len"] = df["morph"].apply(lambda x: len(x))
    return df

def get_corpora() -> pd.DataFrame:
    """Obtiene y combina datos de corpus de diferentes idiomas en un DataFrame
    obteniendo corpora multilingüe

    Returns:
    -------
    pd.DataFrame
        Un DataFrame que contiene los datos de corpus combinados de varios idiomas.
    """
    corpora = pd.DataFrame()
    for lang in LANGS:
        files = get_files(lang)
        raw_data = get_raw_corpus(files)
        dataframe = raw_corpus_to_dataframe(raw_data, lang)
        corpora = dataframe if corpora.empty else pd.concat([corpora, dataframe], ignore_index=True)
    return corpora
# Código de la práctica

# CLI
if __name__ == '__main__':
    # 1. Obtener 10 oraciones al azar del conjunto de pruebas del corpus SIGMORPH 2022
    corpora = get_corpora()
    print("Selecting sentences")
    number_sentences = 10
    sentences = corpora.sample(n=number_sentences)
    sentences["words"] = sentences["sentence"].apply(lambda x: nltk.tokenize.word_tokenize(x))
    # 2.1 Stemming (nltk)
    print("Sentences collected, starting stemming")
    # Esta línea de código hace que no pueda stemmizar en checo
    ps = nltk.stem.snowball.SnowballStemmer("english")
    sentences["stems"] = sentences["words"].apply(lambda x: [ps.stem(word) for word in x])
    # 2.2 Lemmatization (spacy)
    print("Stemming finished, starting lemmatization")
    nlp = spacy.load("en_core_web_sm")
    sentences["lemmas"] = sentences["sentence"].apply(lambda x: [token.lemma_ for token in nlp(x)])
    # 2.3 Información morfológica
    print("Lemmatization finished, getting morphological information")
    sentences["morphology"] = sentences["sentence"].apply(lambda x: [token.morph.to_dict() for token in nlp(x)])
    # 2.4 Impresión en pantalla
    print("Printing results")
    print(sentences)
    console = Console()
    for index, s in sentences.iterrows():
        print("Sentence:")
        print("| " + s["sentence"])
        # stems_joined = ' '.join(s["stems"])
        # lemmas_joined = ' '.join(s["lemmas"])
        # print(f"| Stemmization and lemmatization")
        table = Table(title = "Stemmization and lemmatization")
        table.add_column("Word",style="cyan")
        table.add_column("Stem",style="magenta")
        table.add_column("Lemma",style="green")
        for word,stem,lemma in zip(s["words"],s["stems"],s["lemmas"]):
            table.add_row(word, stem, lemma)
        console.print(table)
        for word, morph in zip(s["words"],s["morphology"]):
            table_w = Table(title = f"<<{word}>> morphology")
            table_w.add_column("Label",style="cyan")
            table_w.add_column("Value",style="magenta")
            for key, val in morph.items():
                table_w.add_row(key, val)
            console.print(table_w)