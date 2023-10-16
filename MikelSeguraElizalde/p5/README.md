Autor: Mikel Segura Elizalde | Número de cuenta: 420004231 | Versión: 1; Octubre 2023

# Respuestas a las preguntas

## ¿Aumentó o disminuyó la entropía para los corpus?
	▪︎ Nahuatl
	Disminuyó.
	▪︎ Español
	Disminuyó.

## ¿Qué significa que la entropía aumente o disminuya en un texto?
	Que la entropía aumente en un texto significa que cada segmento (token) que lo compone es más impredecible. También significa que, entre más aumente, cada segmento tendrá consigo más información.

## ¿Como influye la tokenización en la entropía de un texto?
	Hace que disminuya; puesto que la tokenización reduce la cantidad de tokens (segmentos) distintos en los que se separa el texto, provocando que hayan menos tokens con muy baja frecuencia. Los tokens de muy baja frecuencia hacen que aumente considerablemente la entropía; razon por la cual segmentando el texto mediante sus espacios, es decir, tokenizar por palabras, no sólo nos genera un vocabulario (lista de tokens distintos) muy grande, sino que también provocará una alta entropía, debido a que hay palabras con frecuencias mucho más bajas que otras.

# Cómo correr el script:

Basta con abrir tokenization.py usando Python3.

# Qué dependencias son necesarias y cómo instalarlas.

El script requiere de tres librerías: nltk, elotl y subword-nmt.
Para instalarlas, ejecutar las siguientes líneas en la terminal:

	pip install nltk
	pip install elotl
	pip install subword-nmt