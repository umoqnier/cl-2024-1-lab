"""
Practica 03 <<Pos Tagging>>

POS (Parts of Speech) en HÑÄHÑU (Otomí)
"""
# Paquetería
from inspect import Attribute
# NLP
import nltk 
from nltk.corpus import cess_esp
# Entrenamiento de modelos
from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
# Analisis de resultados
from sklearn.metrics import *
import pandas as pd
from json import loads
# Reproducibilidad
import numpy as np
import random
from unidecode import unidecode

def tone(word):
    for vowel in ["aa","ee","ii","oo","uu"]:
        if vowel in word:
            # Ascending tone
            return 1
    for vowel in ["á","é","í","ó","ú"]:
        if vowel in word:
            # High tone
            return 2
    return 3
def position_of_letter(chr,word):
    try:
        return word.index(chr)
    except:
        return -1

def word_to_features(word,pos):
    """
    Long de la palabra
    Termina en vocal
    tono
        - ascendente con doble vocal
        - tilde ´ tono alto
        otrherwise bajo
    Tiene glotal al inicio 
    Tiene glotal al medio --- Composición de palabras
    Tiene n o m --- Composición de palabras
    Empieza en nu o gigo --- pronominales ye
    -k,h,g,n + vocal --- terminaciones de verbo
    Inicia en hín --- negación
    """
    features = {
        'length':       str(len(word)),
        'pos':          str(pos),
        # 'tone':         tone(word),
        'glotal_pos':   str(position_of_letter("'", word)),
        'm_pos':        str(position_of_letter("m", word)),
        'n_pos':        str(position_of_letter("n", word)),
        'hiacuten_pos': str(position_of_letter("hín", word)),
        'bi_pos':       str(position_of_letter("bi", word)),
        # 'unicode_word': unidecode(word)
    }
    return features
# Código de ayudantía
def sent_to_features(sent):
    return [word_to_features(sent, i) for i in range(len(sent))]

def sent_to_labels(sent):
    return [label for token, label in sent]

def report(true, predictions):
    report = classification_report(y_true=true, y_pred=predictions)
    print(accuracy_score(true, predictions))
    print(precision_score(true, predictions, average="macro"))
    print(recall_score(true, predictions, average="macro"))
    print(f1_score(true, predictions, average="macro"))
    disp = ConfusionMatrixDisplay.from_predictions(true, predictions)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    print(report)
    
def train_crf_model(features, labels, print_results = True):
    # TODO: Corpus to Features
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
    print(X_train)
    # Initialize and train the CRF tagger: https://sklearn-crfsuite.readthedocs.io/en/latest/api.html
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True, verbose=True)
    try:
        crf.fit(X_train, y_train)
    except AttributeError as e:
        print(e)
    y_pred = crf.predict(X_test)

    # Flatten the true and predicted labels
    y_test_flat = [label for sent_labels in y_test for label in sent_labels]
    y_pred_flat = [label for sent_labels in y_pred for label in sent_labels]

    if print_results:
        # Evaluate the model
        report(y_test_flat, y_pred_flat)


        
# CLI
if __name__ == '__main__':
    corpus = []
    file_location = "./corpus_otomi.dat"
    with open(file_location) as file_object:
        for line in file_object:
            line = loads(line)
            sentence = []
            for idx, parts_of_word in enumerate(line):
                word = "".join([part[0].strip() for part in parts_of_word[0:-1]])
                # word, label, features
                sentence.append([parts_of_word[-1].strip(),word_to_features(word, idx)])
            corpus.append(sentence)
    # df = pd.DataFrame.from_dict(corpus)
    
    # # pd.set_option('display.max_columns', None)
    # # pd.set_option('display.max_rows', None)
    # print(df.head())
    # Arreglo de features
    X = [[word[1] for word in sentence] for sentence in corpus]
    # Arreglo de etiquetas
    y = [[word[0] for word in sentence] for sentence in corpus]
    # Seed
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    print(len(X_train))
    print("---")
    print("---")
    print("---")
    print(len(y_train))
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=5000, all_possible_transitions=True, verbose=True)
    try:
        crf.fit(X_train, y_train)
    except AttributeError as e:
        print(e)
    y_pred = crf.predict(X_test)

    # Flatten the true and predicted labels
    y_test_flat = [label for sent_labels in y_test for label in sent_labels]
    y_pred_flat = [label for sent_labels in y_pred for label in sent_labels]
    report(y_test_flat, y_pred_flat)
    