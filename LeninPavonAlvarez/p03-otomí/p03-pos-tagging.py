"""
Práctica 03 <<Pos Tagging>>

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
from sklearn.metrics import *
# Análisis de resultados

# TODO: Obtener corpus
# TODO: Procesar corpus para hacerlo digerible sin pérdida de info
# TODO: Definir features
# TODO: word_to_features
def word_to_features():
    return
# Código de ayudantía
def sent_to_features(sent):
    return [word_to_features(sent, i) for i in range(len(sent))]

def sent_to_labels(sent):
    return [label for token, label in sent]

# Arreglo de features
# X = [[word_to_features(sent, i) for i in range(len(sent))] for sent in corpus]
X=[]

# Arreglo de etiquetas
# y = [[map_tag(pos) for _, pos in sent] for sent in corpus]
y=[]
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
    print("Sí jala")