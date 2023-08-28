import requests as r
import itertools
from Levenshtein import distance

response = r.get("https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/en_US.txt")

lang_codes = {
  "ar": "Arabic (Modern Standard)",
  "de": "German",
  "en_UK": "English (Received Pronunciation)",
  "en_US": "English (General American)",
  "eo": "Esperanto",
  "es_ES": "Spanish (Spain)",
  "es_MX": "Spanish (Mexico)",
  "fa": "Persian",
  "fi": "Finnish",
  "fr_FR": "French (France)",
  "fr_QC": "French (Québec)",
  "is": "Icelandic",
  "ja": "Japanese",
  "jam": "Jamaican Creole",
  "km": "Khmer",
  "ko": "Korean",
  "ma": "Malay (Malaysian and Indonesian)",
  "nb": "Norwegian Bokmål",
  "nl": "Dutch",
  "or": "Odia",
  "ro": "Romanian",
  "sv": "Swedish",
  "sw": "Swahili",
  "tts": "Isan",
  "vi_C": "Vietnamese (Central)",
  "vi_N": "Vietnamese (Northern)",
  "vi_S": "Vietnamese (Southern)",
  "yue": "Cantonese",
  "zh": "Mandarin"
}
iso_lang_codes = list(lang_codes.keys())


def response_to_dict(ipa_list: list) -> dict:
    """Parse to dict the list of word-IPA

    Each element of text hae the format:
    [WORD][TAB][IPA]

    Parameters
    ----------
    ipa_list: list
        List with each row of ipa-dict raw dataset file

    Returns
    -------
    dict:
        A dictionary with the word as key and the phonetic
        representation as value
    """
    result = {}
    for item in ipa_list:
        item_list = item.split("\t")
        result[item_list[0]] = item_list[1]
    return result

def get_ipa_dict(iso_lang: str) -> dict:
    """Get ipa-dict file from Github

    Parameters:
    -----------
    iso_lang:
        Language as iso code

    Results:
    --------
    dict:
        Dictionary with words as keys and phonetic representation
        as values for a given lang code
    """
    response = r.get(f"https://raw.githubusercontent.com/open-dict-data/ipa-dict/master/data/{iso_lang}.txt")
    raw_data = response.text.split("\n")
    return response_to_dict(raw_data[:-1])

def get_ipa_transcriptions(word: str, dataset: dict) -> list[str]:
    """Search for word in a given dataset of IPA phonetics

    Given a word this function return the IPA transcriptions

    Parameters:
    -----------
    word: str
        A word to search in the dataset
    dataset: dict
        A dataset for a given language code
    Returns
    -------
    """
    return dataset.get(word.lower(), "NOT FOUND").split(", ")


def retrieve_dataset(lang: str, data: dict = {}) -> dict:
    """Search in lang in the given dataset 

    Search lang as the key in the data dictionary an returns it.
    if it doesn't exists, downloads it.

    Parameters:
    -----------
    lang: str
        A language code to retrieve the dataset of the language
        should be any from 'iso_lang_codes'
    data: dict
        A dataset that maps a lang_code -> dictionary of IPA 
        transcriptions for the given language
    Returns
    -------
        A dictionary of dictionaries of IPA transcripts
    """
    if lang not in data.keys():
        if lang in iso_lang_codes:
            print("Corpus no encontrado. Descargando...")
            data = {lang : get_ipa_dict(lang)}
        else:
            print (lang + " no existe en la base de datos.")
    return data

def lookup_word(string: str, sub_dataset)-> list[str]:
    """Search for a word in the given subdataset of IPA phonetics.

    Given a string, this function searches the IPA transcriptions.
    If it doesn't find any matches, it looks for similar words using 
    'Levenshtein distance' and makes a suggestion, asking the user
    to re-enter the correct term.

    Parameters:
    -----------
    string: str
        A string to search in the dataset.
    sub_dataset: dict
        A dataset for a given language code.

    Returns:
    -------
        A list of strings with the IPA transcriptions for the given word.
    """ 
    result = get_ipa_transcriptions(string, sub_dataset) 
    if result == ['NOT FOUND']:
        for word in sub_dataset.keys():
            if distance(string, word) <= 1:
                print("Quizá quisiste decir '"+ word + "' en lugar de " + string + ".")
                break
        print("Introduce la palabra correctamente a continuación.")
        string = input(">> ")
        if string:
            return lookup_word(string, sub_dataset)
        else:
            return ['']
    else:
        return result
    

print("Representación fonética de palabras")
print(f"Lenguas disponibles: {(iso_lang_codes)}")
dataset = {}
lang = input("lang>> ")
print(f"Selected language: {lang_codes[lang]}") if lang else print("Adios 👋🏼")
while lang:
    words_lists = []
    dataset = retrieve_dataset(lang, dataset)
    sub_dataset = dataset[lang]
    query =  input(f"  [{lang}]words>> ").strip().split()
    for word in query : words_lists.append( lookup_word(word, sub_dataset))
    print("\n".join([' '.join(str(y) for y in x) for x in itertools.product(*words_lists)]))
    while query:
        query = input(f"  [{lang}]words>> ").strip().split()
        words_lists = []
        for word in query : words_lists.append( lookup_word(word, sub_dataset))
        print("\n".join([' '.join(str(y) for y in x) for x in itertools.product(*words_lists)]))
    lang = input("lang>> ")
    print(f"Selected language: {lang_codes[lang]}") if lang else print("Adios 👋🏼")