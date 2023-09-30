# Autor: Mikel Segura Elizalde
# Versión 1, septiembre 2023

from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
from unidecode import unidecode
from sklearn_crfsuite import CRF
from random import randrange
from sklearn.metrics import classification_report

corpus_file = open("corpus_otomi", "r")
raw_corpus = corpus_file.read()
raw_sentences = (raw_corpus.replace(' ','')).split("\n")
sentences_words_unsplitted = [sentence.split("],[[") for sentence in raw_sentences]
sentences_words_splitted = [[word.split("],") for word in sentence] for sentence in sentences_words_unsplitted]
sentences_words_splitted.pop() #necesario para quitar un vacío al final

def get_word_and_pos(splitted_word):
  word = ''
  ignore = '[]"'
  for morpheme in splitted_word[:-1]:
    word += (morpheme.split('","')[0]).translate({ord(i): None for i in ignore})
  return [word, unidecode(splitted_word[-1].translate({ord(i): None for i in ignore}))]

def get_sentence(splitted_sentence):
  result = ""
  for word in splitted_sentence:
    result += get_word_and_pos(word)[0] + ' '
  return result[:-1]

corpus = [[get_word_and_pos(word) for word in sentence] for sentence in sentences_words_splitted]

def word_to_features(sent, i):
    word = sent[i][0]
    features = {
        'word': word,
        'word[:3]': word[:3], # tratando de detectar sufijos
        'word[:2]': word[:2],
        'word[-3:]': word[-3:], # tratando de detectar prefijos
        'len(word)': len(word), # importante al ser el otomí muy sintético
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        prev_word = sent[i - 1][0]
        features.update({
            'prev_word.lower()': prev_word.lower(),
            'prev_word.istitle()': prev_word.istitle(),
        })
    else:
        features['BOS'] = True  # Beginning of sentence
    return features

# Extract features and labels
def sent_to_features(sent):
    return [word_to_features(sent, i) for i in range(len(sent))]

def sent_to_labels(sent):
    return [label for token, label in sent]

# Prepare data for CRF
X = [[word_to_features(sent, i) for i in range(len(sent))] for sent in corpus]
y = [[pos for _, pos in sent] for sent in corpus]

 # Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
print(report)

# Mostrar ejemplos
stop_examples = ''
while not stop_examples:
  random_index_y_test = randrange(len(y_test)-1)
  test_sentence = []
  for word in X_test[random_index_y_test]:
    test_sentence.append(word['word'])
  print(f' '.join(test_sentence))
  test_sentence_pos_analysis = list(zip(test_sentence, y_test[random_index_y_test]))
  for word, pos in test_sentence_pos_analysis:
    print('  -  ' ,word, ' --> ', pos)
  stop_examples = input('\npresiona enter para otro ejemplo aleatorio del conjunto de pruebas\n')