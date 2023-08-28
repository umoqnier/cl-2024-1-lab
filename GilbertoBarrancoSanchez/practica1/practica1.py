import requests as r

"""
The following package will help us to calculate distance between words.

"""
import textdistance

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
    return dataset.get(word.lower(), "NOT FOUND").split(". ")


def get_dataset(lang: str) -> dict:
    """Download dataset from ipa-dict github
    Given a string of a lang code, download an available dataset.

    Returns
    -------
    dict
        Lang code as key and dictionary with words-transcriptions
        as values
    """
    return {lang: get_ipa_dict(lang)}

def get_sentences(sentence_list: list):
  """ Obtain all the possible phonetic representations of a sentence

  Parameters:
    -----------
    iso_lang:
        List of different phonetic representations of each word in a sentence.

    Results:
    --------
    print:
        List of different phonetic representations of a sentence
  """

  sentences = ['']

  if sentence_list != []:

    for word_list in sentence_list:

      new_sentences = []

      for sentence in sentences:

        for word in word_list:

          """ 'if' is in order to avoid spaces in the beginning of the sentence"""
          if word_list == sentence_list[0]:
            new_sentence = sentence + word
            new_sentences.append(new_sentence)
          else:
            new_sentence = sentence + ' ' + word
            new_sentences.append(new_sentence)

      sentences = new_sentences
  else:
    return None

  print("Phonetic representations:\n")
  for sentence in sentences:
    print("-", sentence)

def get_near(word:str, sub_dataset: dict):
    """ Shows similiar words to 'word' which are contained in the dictionary sub_dataset

  Parameters:
    -----------
    word:
        A word which is not a key in sub_dataset

    Results:
    --------
    print:
        List of similar words to word
  """
    print(f"La palabra '{word}' no fue encontrada. Algunas sugerencias son:")

    for key in sub_dataset.keys():
      if textdistance.damerau_levenshtein(word, key) <= 1:
        print(key, sub_dataset[key])

print("Representación fonética de frases")

print(f"Lenguas disponibles: {(iso_lang_codes)}")

lang = input("lang>> ")
print(f"Selected language: {lang_codes[lang]}") if lang else print("Adios 👋🏼")

while lang:
    sub_dataset = get_dataset(lang)[lang]
    squery = input(f"  [{lang}]sentence>> ")

    while squery:
      query = squery.split()
      sentence_list = []

      for word in query:
        result = get_ipa_transcriptions(word, sub_dataset)
        result = result[0].split(",")

        if result[0] == "NOT FOUND":
          get_near(word, sub_dataset)
          sentence_list = []
          break
        else:
          sentence_list. append(result)

      get_sentences(sentence_list)

      squery = input(f"  [{lang}]sentence>> ")

    lang = input("lang>> ")
    print(f"Selected language: {lang_codes[lang]}") if lang else print("Adios 👋🏼")

