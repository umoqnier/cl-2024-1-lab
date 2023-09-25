#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import cess_esp
from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split


# In[2]:


import ast

# Nombre del archivo a leer
nombre_archivo = "./corpus_otomi"

# Lista para almacenar los datos recuperados
datos_recuperados = []

# Leer el archivo y convertir cada línea en una lista de Python
with open(nombre_archivo, 'r', encoding='utf-8') as archivo:
    for linea in archivo:
        try:
            # Utilizamos ast.literal_eval para interpretar la línea como una lista válida en Python
            lista_datos = ast.literal_eval(linea.strip())
            datos_recuperados.append(lista_datos)
        except ValueError as e:
            print(f"Error al procesar la línea: {linea}")
            print(e)

corpus = datos_recuperados


# In[3]:


len(corpus)


# In[4]:


corpus[0]


# In[5]:


from unidecode import unidecode
def get_corpus_list(corpus):
    biglst = []
    
    for sentence in corpus:
        shortlst = []
        for trozo in sentence:
            s = ""
            s.encode('utf-8')
            for chunk, gloss in trozo[:-1]:
                s += chunk
            pos = trozo[-1]
            shortlst.append((s, unidecode(pos)))
        biglst.append(shortlst)

    return biglst
cuerpo = get_corpus_list(corpus)


# In[6]:


cuerpo[0]


# In[7]:


def word_to_features(sent, i):
    word = sent[i][0]
    features = {
        'word.lower()': word.lower(),
        #'word[-3:]': word[-3:],
        'word[:3]': word[:3],
        'word.lenght': len(word),
        'word.quote': "'" in word,
        'word.isascii()' : word.isascii(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        prev_word = sent[i - 1][0]
        features.update({
            'prev_word.lower()': prev_word.lower(),
            'prev_word[:3]': prev_word[:3],
            'prev_word.istitle()': prev_word.istitle(),
        })
    else:
        features['BOS'] = True  # Beginning of sentence
    if len(sent) > i+1:
        next_word = sent[i + 1][0]
        features.update({
            'next_word.lower()': next_word.lower(),
            'next_word[:3]': next_word[:3],
            'next_word.istitle()': next_word.istitle(),
        })

    return features

# Extract features and labels
def sent_to_features(sent):
    return [word_to_features(sent, i) for i in range(len(sent))]

def sent_to_labels(sent):
    return [label for token, label in sent]


# In[8]:


X = [[word_to_features(sent, i) for i in range(len(sent))] for sent in cuerpo]
y = [[pos for _, pos in sent] for sent in cuerpo]


# In[9]:


len(X[0]) == len(y[0])


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)


# In[11]:


X_train[0]


# In[12]:


from inspect import Attribute
from sklearn_crfsuite import CRF
# Initialize and train the CRF tagger: https://sklearn-crfsuite.readthedocs.io/en/latest/api.html
crf = CRF(algorithm='lbfgs', c1=0.878054, c2=0.034124, max_iterations=100, all_possible_transitions=True, verbose=True)
try:
    crf.fit(X_train, y_train)
except AttributeError as e:
    print(e)


# In[13]:


from sklearn.metrics import classification_report
y_pred = crf.predict(X_test)

# Flatten the true and predicted labels
y_test_flat = [label for sent_labels in y_test for label in sent_labels]
y_pred_flat = [label for sent_labels in y_pred for label in sent_labels]

# Evaluate the model
report = classification_report(y_true=y_test_flat, y_pred=y_pred_flat,zero_division=0)
print(report)  


# In[14]:


from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay.from_predictions(y_test_flat, y_pred_flat, xticks_rotation='vertical')
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")


# In[15]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

print("accuracy:",accuracy_score(y_pred_flat, y_test_flat))

print("precision",precision_score(y_pred_flat, y_test_flat, average="macro",zero_division=0))
print("recall",recall_score(y_pred_flat, y_test_flat, average="macro",zero_division=0))

print("f1_score",f1_score(y_pred_flat, y_test_flat, average="macro"))


# In[ ]:


import random
user_input = ""
while True:
    print(f"\nPRUEBA EL MODELO:  \n presiona RET tecla para continuar, q para salir: {user_input}")
    user_input = input(">>")
    
    if user_input == 'q':
        break  # Exit the loop if 'q' is entered
    
    
    # Continue executing code here
    
    n = random.randint(1,len(cuerpo))

    cadena = [tupla[0] for tupla in cuerpo[n]]
    real = [tupla[1] for tupla in cuerpo[n]]

    to_pred = [[pos for _, pos in cuerpo[n]]]
    pred = crf.predict(to_pred)
    print("oración:")
    print("\t","  ".join(word for word in cadena))
    print("predecido:")
    print("\t",pred)
    print("real:")
    print("\t",[real])
    
# Code outside the loop (after 'q' is entered) will continue executing
print("Done!")


# In[ ]:





# In[ ]:




