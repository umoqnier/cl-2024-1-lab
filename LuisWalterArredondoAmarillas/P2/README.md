**Programa de Análisis Morfológico de Oraciones**

Este programa es un analizador morfológico de oraciones que procesa el texto de entrada en diferentes idiomas. El programa utiliza varias bibliotecas de Python para descargar, procesar y analizar el texto. A continuación, se proporciona una breve documentación sobre lo que hace el programa, las bibliotecas que requiere y cómo navegar por él:

**Librerías Requeridas:**
- `pandas` (`pd`): Para trabajar con estructuras de datos como DataFrames.
- `requests`: Para realizar solicitudes HTTP y descargar datos de una URL.
- `spaCy` (`spacy`): Para realizar el análisis morfológico y el análisis de lenguaje natural en inglés.
- `nltk` (`nlp`): Para realizar la tokenización de palabras en inglés.
- `SnowballStemmer`: Para realizar el análisis de raíces de palabras en varios idiomas.

```python
import pandas as pd
import requests
import spacy
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
```

**Funciones Principales:**
1. `get_files(lang: str, track: str = "word") -> list[str]`: Genera una lista de nombres de archivo basados en el idioma y la pista del shared task de donde vienen los datos.

2. `get_raw_corpus(files: list) -> list`: Descarga y concatena los datos de los archivos tsv desde una URL base.

3. `raw_corpus_to_dataframe(corpus_list: list, lang: str) -> pd.DataFrame`: Convierte una lista de datos de corpus en un DataFrame de pandas que contiene los datos del corpus procesados.

4. `raw_corpus_to_dataframe_sentences(corpus_list: list, lang: str) -> pd.DataFrame`: Convierte una lista de datos de corpus en un DataFrame de pandas que contiene las oraciones del corpus procesadas.

5. `get_corpora() -> dict`: Obtiene y almacena los datos del corpus para varios idiomas en un diccionario.

6. `get_sentence_list(corpora: dict, lang: str, count: int) -> [str]`: Obtiene una lista de oraciones de prueba para un idioma específico desde los datos del corpus.

7. `get_stems(lang: str, sentence: str) -> [str]`: Realiza el análisis de raíces de palabras (stems) en el idioma especificado para una oración dada.

8. `get_lemmas(lang: str, text: str) -> [str]`: Obtiene los lemas de las palabras en una oración en el idioma especificado.

9. `get_analysis(lang: str, sentence: str) -> []`: Realiza un análisis morfológico de una oración en el idioma especificado.

**Navegación del Programa:**

- El programa permite al usuario seleccionar un idioma para el análisis morfológico. Actualmente, solo admite el idioma inglés ("eng").

- Luego, muestra una lista de oraciones disponibles para analizar y permite al usuario seleccionar una de ellas para su procesamiento.

- Una vez seleccionada una oración, el programa realiza el análisis morfológico y muestra la raíz y el lema de cada palabra en la oración.

- Después de analizar una oración, el programa permite al usuario continuar seleccionando otra oración o cambiar el idioma.

- Para salir del programa, simplemente presione Enter sin seleccionar un idioma.

**Uso del Programa:**

1. Ejecute el programa y seleccione el idioma ingresando "eng" cuando se le solicite.
2. Elija una oración para el análisis.
3. El programa mostrará el análisis morfológico de la oración, incluyendo la raíz y el lema de cada palabra.
4. Puede continuar analizando más oraciones o cambiar el idioma.
5. Para salir del programa, simplemente presione Enter sin seleccionar un idioma.

