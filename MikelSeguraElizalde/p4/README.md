Autor: Mikel Segura Elizalde
Número de cuenta: 420004231
Versión: 1; Octubre 2023

# Cómo correr el script:

Basta con abrir pos_tagging.py usando Python3.

# Qué dependencias son necesarias y cómo instalarlas.

El script requiere de tres librerías: scikit-learn, sklearn-crfsuite, nltk y
elotl.
Para instalarlas, ejecutar las siguientes líneas en la terminal:

	1. pip install scikit-learn
	2. pip install sklearn-crfsuite
	3. pip install nltk
	4. pip install elotl

Además, para generar las nubes de palabras, es necesario poder usar fuentes
TrueType. Para ello, basta con ejecutar en terminal:

	1. pip install --upgrade pip
	2. pip install --upgrade Pillow

# Respuestas a las preguntas

## ¿Obtenemos el mismo resultado? 

No, pero es parecido.

## y ¿Porqué?

Porque las stopwords de paquetería han sido más cuidadosamente seleccionadas.
Zipf sólo depende de la frecuencia, por lo que puede ser menos certero;
aunque puede considerarse decente aproximación, lo cual podemos comprobar
comparando las nubes de palabras y viendo que comparten varias palabras
capturadas.