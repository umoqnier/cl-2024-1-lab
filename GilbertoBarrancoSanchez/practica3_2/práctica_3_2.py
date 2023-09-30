# -*- coding: utf-8 -*-

""" Práctica 3_2.ipynb """

from sklearn.model_selection import train_test_split

postags = {
    "v": "verbo",
    "det": "determinante",
    "dem": "demostrativo",
    "n": "sustantivo",
    "p.loc": "particula locativa",
    "conj.adv": "conjuncion adversativa",
    "gen": "genitivo",
    "it": "iterativo",
    "aff": "afrimativo",
    "dec": "decimal",
    "cord": "coordinacion",
    "regular/v": "verbo regular",
    "obl": "oblicuo",
    "cnj": "conjuncion",
    "unkwn": "desconocido",
    "neg": "negativo",
    "prt": "partcula",
    "dim": "diminutivo",
    "cond": "condicional",
    "lim": "limitativo",
    "loc": "locativo",
    "conj": "conjuncion",
    "cnj.adv": "conjuncion adversativa",
}

tags = postags.keys()


def get_tagged_sentences(corpus):
    # Obtains the sentences in the corpora. Every word of a sentence has their own POStag.
    tagged_sentences = []

    for line in corpus:
        tagged_words = []
        for schema in line:
            pretagged_word = [chunk[0] for chunk in schema[:-1]]
            word = "".join(pretagged_word)
            tagged_word = [word]
            tagged_word.append(schema[-1])
            tagged_words.append(tagged_word)
        tagged_sentences.append(tagged_words)
    return tagged_sentences


def map_tag(tag):
    try:
        return postags[tag]
    except:
        return "unkwn"


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.istitle()": word.istitle(),
        "postag": postag,
        "postag[:2]": postag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        features.update(
            {
                "-1:word.lower()": word1.lower(),
                "-1:word.istitle()": word1.istitle(),
                "-1:postag": postag1,
                "-1:postag[:2]": postag1[:2],
            }
        )
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        features.update(
            {
                "+1:word.lower()": word1.lower(),
                "+1:word.istitle()": word1.istitle(),
                "+1:postag": postag1,
                "+1:postag[:2]": postag1[:2],
            }
        )
    else:
        features["EOS"] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for token, label in sent]


# Comenzamos a tratar el corpus.

import ast
import re

corpus = open("corpus_otomi", "r")
corpus = corpus.read()
corpus = re.sub(r"\n", ", ", corpus)
corpus = eval(corpus)
corpus = get_tagged_sentences(corpus)

# Prepare data for CRF
X = [[word2features(sent, i) for i in range(len(sent))] for sent in corpus]
y = [[map_tag(pos) for _, pos in sent] for sent in corpus]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from inspect import Attribute
from sklearn_crfsuite import CRF

# Initialize and train the CRF tagger: https://sklearn-crfsuite.readthedocs.io/en/latest/api.html
crf = CRF(
    algorithm="lbfgs",
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True,
    verbose=True,
)
try:
    crf.fit(X_train, y_train)
except AttributeError as e:
    print(e)

from sklearn.metrics import classification_report

y_pred = crf.predict(X_test)

# Flatten the true and predicted labels
y_test_flat = [label for sent_labels in y_test for label in sent_labels]
y_pred_flat = [label for sent_labels in y_pred for label in sent_labels]

# Evaluate the model
report = classification_report(y_true=y_test_flat, y_pred=y_pred_flat)
print(report)

print("\nReporte de accurary, precision, recall y F1-score\n")

from sklearn.metrics import accuracy_score

print("\tAccuracy", accuracy_score(y_pred_flat, y_test_flat))

from sklearn.metrics import precision_score

print("\tPrecision:", precision_score(y_pred_flat, y_test_flat, average="macro"))

from sklearn.metrics import recall_score

print("\tRecall:", recall_score(y_pred_flat, y_test_flat, average="macro"))

from sklearn.metrics import f1_score

print("\tF1-score:", f1_score(y_pred_flat, y_test_flat, average="macro"))

from sklearn.metrics import ConfusionMatrixDisplay

disp = ConfusionMatrixDisplay.from_predictions(y_test_flat, y_pred_flat)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

print("\n\nEl siguiente es un ejemplo de oración etiquedada\n")
print("Tomamos un elemento del corpus, correspondiente a los datos de evaluación.\n")
# Reconstruimos la oración
sentence = []
for i in range(len(X_test[1])):
    sentence.append(X_test[1][i]["word.lower()"])

# Obtenemos las etiqutes originales
Xtags = []
for i in range(len(X_test[1])):
    Xtags.append(postags[X_test[1][i]["postag"]])

print("En este caso, consideramos la oración:\n ", " ".join(sentence), "\n")
print("Por otro lado, las etiquetas POS de nuestra oración son:\n", Xtags, "\n")
print(
    "Finalmente, las etiquetas POS calculadas por el modelo predictivo, son:\n",
    y_pred[1],
)
