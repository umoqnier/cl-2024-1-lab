# %%
import requests as r

# %%
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

# %%
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

# %%


def get_dataset() -> dict:
    """Download corpora from ipa-dict github

    Given a list of iso lang codes download available datasets.

    Returns
    -------
    dict
        Lang codes as keys and dictionary with words-transcriptions
        as values
    """
    return {code: get_ipa_dict(code) for code in iso_lang_codes}

dataset = get_dataset()



# %%
def levenshtein_distance(s: str, t: str) -> int:
    """Returns the levenshtein distance beetween two words,
    which is the minimum number of single-character edits 
    (insertions, deletions or substitutions) required to change one word into the other

    Algorithm from https://en.wikipedia.org/wiki/Levenshtein_distance
    
    Parameters:
    -----------
    s: str
        A word to compare
    t: str
        A word to compare

    Returns:
    --------
    int
        the levenshtein distance between s and t
    """

    word1 = list(s)
    word2 = list(t)
    m = len(word1)
    n = len(word2)
    distances = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    for i in range(1, m + 1):
        distances[i][0] = i

    for j in range(1, n + 1):
        distances[0][j] = j 


    for j in range(1, n + 1):
        for i in range(1, m + 1):
            substitution_cost = 1
            if word1[i - 1] == word2[j - 1]:
                substitution_cost = 0
            
            distances[i][j] = min(distances[i-1][j] + 1,
                         distances[i][j-1] + 1,
                         distances[i-1][j-1] + substitution_cost)
    return distances[m][n]

# %%
def phrase_query(phrase: str, dataset: dict) -> dict:
    """Gets the IPA transcriptions for every word in a given phrase
    
    Parameters:
    -----------
    phrase: str
        A phrase whose words are to look up in the dataset
    dataset: dict
        A dataset for a given language code
    
    Returns
    -------
    dict
        A list containing each word with their transcriptions
    """
    words = phrase.split()
    transcriptions = dict()
    for word in words:
        transcriptions[word] = get_ipa_transcriptions(word, dataset)

    return transcriptions

def print_transcription_query(query, transcriptions: dict) -> None:
    for word in transcriptions:
        print(word, " | ", ", ".join(transcriptions[word]))

def similar_words(query: str, dataset: dict, distance: int = 2) -> list[str]:
    """Given a word and a dataset, finds the words that can be similar
    
    Parameters:
    -----------
    query: str
        A word to search for similarities
    dataset: dict
        A dataset for a given language code
    dintance: int
        Levenshtein distance between the words
    
    Returns
    -------
    list
        of words that are similar
    """

    similar = []
    for word in dataset:
        if levenshtein_distance(query, word) < distance:
            similar.append(word)

    return similar

# %%
print("Phonetic representation of words")

print(f"Available languages: {(iso_lang_codes)}")

lang = input("lang>> ")
print(f"Selected language: {lang_codes[lang]}") if lang else print("Adios 👋🏼")
dataset = dict()
while lang:
    if lang not in dataset:
        print(f"Downloading {lang} dataset...")
        dataset[lang] = get_ipa_dict(lang)
        print(f"{lang} dataset downloaded ✅")
    sub_dataset = dataset[lang]
    query = "empty query"
    while query:
        query = input(f"  [{lang}]phrase>> ")
        result = phrase_query(query, sub_dataset)
        is_successul = True
        wrong_word = ''
        for key in result:
            if result[key][0] == 'NOT FOUND':
                is_successul = False
                wrong_word = key
                break
        
        print(query)
        if is_successul:
            print_transcription_query(query, result)
        else:
            similar = similar_words(wrong_word, sub_dataset, distance=2)
            print(f"<{wrong_word}> wasn't found. Did you mean...", ", ".join(similar), " ?")
    lang = input("lang>> ")
    print(f"Selected language: {lang_codes[lang]}") if lang else print("Ciao 👋🏼")




