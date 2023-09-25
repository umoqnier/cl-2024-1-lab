import ast
from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
from inspect import Attribute
import warnings
from sklearn_crfsuite import CRF
from sklearn.metrics import f1_score
from unidecode import unidecode
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score


encoding = "utf-8"

def get_otomi_corpus():
    try:
        sentences = []
        with open("corpus_otomi.txt", "r", encoding=encoding) as file:
            for line in file:
                try:
                    # Evaluar la línea como una expresión de Python y guardarla en una variable
                    sentences.append(ast.literal_eval(line.strip()))  
                except Exception as e:
                    print(f"Error al evaluar la línea: {line}")
                    print(e)
        return sentences
    except FileNotFoundError:
        print("El archivo 'corpus_otomi.txt' no se encontró en la carpeta actual.")
    except Exception as e:
        print("Ocurrió un error al abrir el archivo:")
        print(e)

def format_sentence(sent: list):
    data = []
    for word in sent:
        _word = {}
        _word["tag"] = word[-1]
        _word["word"] = "".join(map(lambda s: s[0],word[:-1])) 
        _word["morphemes"] = word[:-1]
        data.append(_word)  
    return data

def map_tag(tag: str, tags_map : list) -> str:
    return tags_map.get(tag.lower(), "N/F")

def word_to_features(sent, i):
    word = sent[i]
    stem = list(filter(lambda x: x[1] == "stem", word['morphemes']))
    # para las features tomé las que nos dió la profesora como sugerencia
    features = {
        'word.lower()': word["word"].lower(),
        'word[:3]': word["word"][:3],
        'word[:2]': word["word"][:2],
        'word.isdigit()': word["word"].isdigit(),
        'length': len(word["word"]),
        'position': i
    }

    if len(stem) > 0: features.update({'stem': stem[0][0]})

    return features

# Extract features and labels
def sent_to_features(sent):
    return [word_to_features(sent, i) for i in range(len(sent))]

def sent_to_labels(sent):
    return [label for token, label in sent]

corpus = get_otomi_corpus() 
data = [format_sentence(s) for s in corpus]

# Prepare data for CRF
X = [[word_to_features(sent, i) for i in range(len(sent))] for sent in data]
y = [[unidecode(word["tag"]) for word in sent] for sent in data]

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# para tener una salida limpia y que no se vean los warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

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


    # Evaluate the model
    report = classification_report(y_true=y_test_flat, y_pred=y_pred_flat)
    print("\n\n******* Evaluación del modelo *******")
    print(report)
    print("******* ACCURACY *******")
    print(accuracy_score(y_pred_flat, y_test_flat))
    print("******* PRESICION *******")
    print(precision_score(y_pred_flat, y_test_flat, average="macro"))
    print("******* RECALL *******")
    print(recall_score(y_pred_flat, y_test_flat, average="macro"))
    print("******* F1_SCORE *******")
    print(f1_score(y_pred_flat, y_test_flat, average="macro"))
    print("\n\n\n******* Ejemplo de oración etiquetada *******")
    sentence = X_test[0]
    string = " ".join([w['word.lower()'] for w in sentence])
    print(f"Oración: {string}")
    prediction = crf.predict(sentence)
    print(prediction)
    # print(f"Etiquetado obtenido: {}")
    print(f"\nEtiquetado correcto: {', '.join(y_test[0])}")
