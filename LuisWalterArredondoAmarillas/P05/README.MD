# Práctica 5: Subword tokenization
**Fecha de entrega:** 15 de Octubre 11:59pm

## Descripción
En esta práctica, investigaremos cómo medir la entropía de un texto y compararemos la entropía de la Biblia en español y el texto "axolotl" antes y después de la tokenización utilizando BPE native (subword-nmt).

## Instrucciones
1. Investiga cómo medir la entropía de un texto.
2. Mide la entropía de la Biblia en español sin tokenizar.
3. Mide la entropía de la Biblia en español después de la tokenización.
4. Mide la entropía del texto "axolotl" sin tokenizar.
5. Mide la entropía del texto "axolotl" después de la tokenización.
6. Responde las siguientes preguntas:

   - ¿Aumentó o disminuyó la entropía para los corpus en español y Nahuatl?
   - ¿Qué significa que la entropía aumente o disminuya en un texto?
   - ¿Cómo influye la tokenización en la entropía de un texto?

## Bibliotecas necesarias
Asegúrate de tener las siguientes bibliotecas instaladas antes de ejecutar el código en tu notebook de Jupyter:

```bash
!pip install elotl
!pip install subword-nmt
!pip install spacy
```

Además, asegúrate de descargar el modelo de Spacy para el idioma español:

```python
import spacy
spacy.load('es_core_news_sm')
```

También, necesitarás descargar el corpus "cess_esp" de NLTK:

```python
import nltk
nltk.download("cess_esp")
```

    Asegúrate de que todas las bibliotecas estén instaladas y configuradas antes de ejecutar el código en tu notebook.
