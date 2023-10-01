# %%
from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
import json
from unidecode import unidecode
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd

# %%
def open_otomi_corpus() -> list:

    with open("corpus_otomi", "r", encoding='utf8') as file:
        lines = file.readlines()
        corpus = []
        for line in lines:
            corpus.append(json.loads(line))
        
        return corpus
    
def join_chunks(data: list) -> list:
    """Given a word with the following structure:
        [[CHUNK, GLOSS], [CHUNK, GLOSS],..., POS]
           |---------------------------------|
                        WORD
    joins the chunks while deleting the gloss

    Parameters:
    -----------
    corpus: list
        a list that contains chunks and a pos tag
    
    Returns:
    --------
    (str, str, int)
        with joined chunks, tag, number of chunks and the gloss
        vector"""
    
    num_chunks = len(data) - 1
    gloss_vector = []
    word = ""
    for element in range(0, num_chunks):
        chunk = data[element][0]
        gloss = data[element][1]
        # Converts to ASCII to avoid future problems
        word += unidecode(chunk)
        gloss_vector.append(unidecode(gloss))

    tag = data[num_chunks]
    return (word, unidecode(tag), num_chunks, gloss_vector)

def word_to_features(sentence: list, index: int) -> dict:
    """Given a sentence, gets the word at the
    index position and extracts its features"""
    
    word, _, num_chunks, gloss_vector = sentence[index]
    features = {
        'word.lower()': word.lower(),
        'word[0:3]': word[0:3],
        'word[0:2]': word[0:2],
        'gloss': gloss_vector,
        'chunks': num_chunks
    }
    
    if index > 0:
        prev_word, _, _, _ = sentence[index - 1]
        features.update({
            'prev_word.lower()': prev_word.lower(),
        })
    else:
        features['BOS'] = True
    
    return features

def sentence_to_features(sentence: list[tuple]) -> list:
    return [word_to_features(sentence, i) for i in range(len(sentence))]

def sent_to_labels(sentence: list[tuple]) -> list:
    return [tag for _, tag, _, _ in sentence]


def train_crf(corpus: list) -> tuple:
    """Creates a POS tagging model from a given corpus"""
    sentences = []
    for sentence in corpus:
        joined = [ join_chunks(word) for word in sentence ]
        sentences.append(joined)


    X = [[word_to_features(s, i) for i in range(len(s))] for s in sentences]
    y = [[tag for _, tag, _, _ in s] for s in sentences]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    crf = CRF(algorithm='lbfgs', 
              c1=0.1, 
              c2=0.1, 
              max_iterations=100, 
              all_possible_transitions=True, 
              verbose=True)
    try:
        crf.fit(X_train, y_train)
    except AttributeError as e:
        print(e)

    return crf, X_test, y_test

# %% 
corpus = open_otomi_corpus()
crf, X_test, y_test = train_crf(corpus)

# %%
y_pred = crf.predict(X_test)

y_test_flat = [label for sent_labels in y_test for label in sent_labels]
y_pred_flat = [label for sent_labels in y_pred for label in sent_labels]

print("Metrics")
print("accuracy: ",accuracy_score(y_pred_flat, y_test_flat))
print("precision: ",precision_score(y_pred_flat, y_test_flat, average="macro", zero_division=0))
print("recall: ",recall_score(y_pred_flat, y_test_flat, average="macro", zero_division=0))
print("f1: ",f1_score(y_pred_flat, y_test_flat, average="macro"))
# %%
sentence_test = [ feature["word.lower()"] for feature in X_test[3] ]
prediction = crf.predict_single(X_test[3])
original =  y_test[3]

data = { "sentence": sentence_test, "original": original, "prediction": prediction }
df = pd.DataFrame(data)
print(df)