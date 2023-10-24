# README - Programa de etiquetado de partes del habla en textos en Otomi

Este programa utiliza el aprendizaje automático para etiquetar las partes del habla en textos en otomi. El etiquetado de partes del habla es una tarea común en el procesamiento del lenguaje natural (NLP) que implica asignar una etiqueta gramatical (como sustantivo, verbo, adjetivo, etc.) a cada palabra en un texto.

## Librerías necesarias

Asegúrate de tener instaladas las siguientes librerías antes de ejecutar el programa:

- `nltk`: Librería de procesamiento del lenguaje natural para cargar el corpus y funciones auxiliares.
- `sklearn_crfsuite`: Implementa un etiquetador de partes del habla basado en CRF (Conditional Random Fields).
- `unidecode`: Utilizada para manejar caracteres Unicode en el procesamiento de texto.

Puedes instalar estas librerías usando `pip`:

```bash
pip install nltk
pip install sklearn-crfsuite
pip install unidecode
```

## Modo de uso

1. Asegúrate de tener instaladas las librerías mencionadas anteriormente.

2. Ejecuta el programa en Python con el script proporcionado:

```python
python main.py
```

3. El programa comenzará a entrenar un etiquetador de partes del habla en base al corpus proporcionado en el archivo `corpus_otomi`.

4. Una vez que el etiquetador esté entrenado, podrás probarlo ingresando oraciones en español. Presiona la tecla `Enter` para continuar con la siguiente oración o `q` y luego `Enter` para salir del programa.

5. El programa mostrará la oración ingresada, las etiquetas predichas por el etiquetador y las etiquetas reales (si están disponibles en el corpus).

6. Después de salir del programa, verás estadísticas de rendimiento, incluyendo la precisión, la recuperación y el puntaje F1 del etiquetador en el conjunto de prueba.

Nota: Asegúrate de tener un archivo llamado `corpus_otomi` en el mismo directorio que el programa, que contenga datos en un formato adecuado para entrenar el etiquetador. Asegúrate también de que los datos estén etiquetados correctamente con las partes del habla en Otomi.

¡Disfruta de tu etiquetador de partes del habla en Otomi!