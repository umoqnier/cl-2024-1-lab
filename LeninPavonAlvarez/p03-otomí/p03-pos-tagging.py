"""
Practica 03 <<Pos Tagging>>

POS (Parts of Speech) en HÑÄHÑU (Otomí)
"""

# Entrenamiento de modelos
from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
# Analisis de resultados
from sklearn.metrics import *
from json import loads
# Reproducibilidad
import numpy as np
import random
from unidecode import unidecode
from rich.console import Console
from rich.table import Table

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
        'tone':         tone(word),
        'glotal_pos':   str(position_of_letter("'", word)),
        'o_pos':        str(position_of_letter("o", word)),
        'm_pos':        str(position_of_letter("m", word)),
        'n_pos':        str(position_of_letter("n", word)),
        'hiacuten_pos': str(position_of_letter("hín", word)),
        'bi_pos':       str(position_of_letter("bi", word)),
        'unicode_word': unidecode(word),
        'utf-8':        word.encode(encoding = 'UTF-8')
    }
    return features
# Código de ayudantía
def sent_to_features(sent):
    return [word_to_features(sent, i) for i in range(len(sent))]

def sent_to_labels(sent):
    return [label for token, label in sent]

def report(true, predictions):
    report = classification_report(y_true=true, y_pred=predictions, zero_division=np.nan)
    print(report)
    print("Accuracy score:", accuracy_score(true, predictions))
    print("Precision score:", precision_score(true, predictions, average="macro", zero_division=np.nan))
    print("Recall score:", recall_score(true, predictions, average="macro", zero_division=np.nan))
    print("f1 score:", f1_score(true, predictions, average="macro"))
    
    
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

    # Arreglo de features
    X = [[word[1] for word in sentence] for sentence in corpus]
    # Arreglo de etiquetas
    y = [[word[0] for word in sentence] for sentence in corpus]
    # Seed
    random_seed = 42
    np.random.seed(random_seed)
    random.seed(random_seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

    crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=1000, all_possible_transitions=True, verbose=True)
    try:
        crf.fit(X_train, y_train)
    except AttributeError as e:
        print(e)
    y_pred = crf.predict(X_test)

    # Flatten the true and predicted labels
    y_test_flat = [label for sent_labels in y_test for label in sent_labels]
    y_pred_flat = [label for sent_labels in y_pred for label in sent_labels]
    report(y_test_flat, y_pred_flat)
    
    table = Table(title="Ejemplo")
    table.add_column("Palabra", style="cyan")
    table.add_column("Real", style="magenta")
    table.add_column("Predicción", style="green")
    for idx, x in enumerate(X_test[random_seed]):
        table.add_row(x['utf-8'].decode(encoding = 'UTF-8'), y_test[random_seed][idx],y_pred[random_seed][idx])
    console = Console()
    console.print(table)
    