import ast
from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
import codecs 
import random

#Código dado por el ayudante
def word_to_features(sent, i):
    word = sent[i][0]    
    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),                       
    }
    if i > 0:
        prev_word = sent[i - 1][0]
        features.update({            
            'prev_word.lower()': prev_word.lower(),
            'prev_word.istitle()': prev_word.istitle(),
            'prev_word.tag': sent[i - 1][1],
        })
    else:
        features['BOS'] = True  # Beginning of sentence

    if i < len(sent) - 1:
        next_word = sent[i + 1][0]
        features.update({
            'next_word.tag': sent[i+1][1],
            'next_word.lower()': next_word.lower(),         #En el paper mencionan que el lugar de la palabra es importante
            'next_word.istitle()': next_word.istitle(),
            'next_word.tag': sent[i + 1][1]
        })
    else:
        features['EOS'] = True  # Ending of sentence

    return features

# Extract features and labels
def sent_to_features(sent):
    return [word_to_features(sent, i) for i in range(len(sent))]


def getword_tag(sent):  
  '''
  Función que obtiene una sentencia y la convierte en pares de 
  [palabra,postag]

  param list:sent
  lista de sentencias

  return list:sentnence
  lista de la forma [[word,tag]]
  '''
  sentence = []
  for w in sent:
    tag = w.pop()                         #Sacamos el postag
    word = "".join([item[0] for item in w])  #Unimos los chunks
    sentence.append([word,tag])
  return sentence



#Inicio del procesamiento:

sentences = []
with codecs.open('corpus_otomi', encoding='utf-8') as f:   #leemos el corpus  
    for line in f:
        sentences.append(ast.literal_eval(line))           #Usamos ast porque lee lista de listas

postpros = [getword_tag(sent) for sent in sentences]       #procesamos el corpus para meterlo en el crf

X = [[word_to_features(sent, i) for i in range(len(sent))] for sent in postpros]
y = [[pos[1] for pos in sent] for sent in postpros]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


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


print("____________REPORTE____________")


print("Accuracy:  ",accuracy_score(y_pred_flat, y_test_flat))
print("Precision: ", precision_score(y_pred_flat, y_test_flat, average="macro"))
print("Recall:    ",recall_score(y_pred_flat, y_test_flat, average="macro"))
print("F1 score:  ",f1_score(y_pred_flat, y_test_flat, average="macro"))


print("____________Analisis de oracion ____________")
iter = random.choice(range(0, len(X_test)-1))
prueba =  X_test[iter]
tex_prueba = " ".join([x['word.lower()'] for x in prueba])
print("Oracion a analizar: ", tex_prueba)
prediction = crf.predict([prueba])
print("Etiquetas encontradas :" ,prediction[0])
print("Etiquetas reales:", y_test[iter])
