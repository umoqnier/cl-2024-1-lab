# Instrucciones para la Práctica 4 de Lingüística Computacional

Este archivo README proporciona instrucciones sobre las bibliotecas de Python que deben instalarse para ejecutar correctamente el notebook de Python relacionado con la Práctica 4 de Lingüística Computacional. A continuación, se detalla cómo instalar estas bibliotecas. Además se incluye un archivo estático del notebook en PDF para comparar la versión local y la que se produjo durante la creación de este notebook.

## Bibliotecas Requeridas

Asegúrate de tener instaladas las siguientes bibliotecas de Python antes de ejecutar el notebook:

1. **collections** (Colecciones):
   - Utilizado para contar elementos en una secuencia.
   - No es necesario instalar, ya que es una biblioteca estándar de Python.

2. **matplotlib**:
   - Utilizado para crear gráficos y visualizaciones.
   - Instalación: `pip install matplotlib`

3. **re** (Expresiones Regulares):
   - Utilizado para el procesamiento de texto basado en expresiones regulares.
   - No es necesario instalar, ya que es una biblioteca estándar de Python.

4. **numpy**:
   - Utilizado para operaciones numéricas y manipulación de matrices.
   - Instalación: `pip install numpy`

5. **string**:
   - Utilizado para operaciones relacionadas con cadenas de texto.
   - No es necesario instalar, ya que es una biblioteca estándar de Python.

6. **elotl**:
   - Utilizado para trabajar con el corpus "elotl".
   - Instalación: `pip install elotl`

7. **wordcloud**:
   - Utilizado para generar nubes de palabras.
   - Instalación: `pip install wordcloud`

8. **nltk** (Natural Language Toolkit):
   - Utilizado para el procesamiento del lenguaje natural.
   - Instalación: `pip install nltk`

9. **unidecode**:
   - Utilizado para convertir caracteres Unicode a sus equivalentes ASCII.
   - Instalación: `pip install unidecode`

## Descarga de Datos Adicionales

Para ejecutar el notebook correctamente, también es necesario descargar algunos datos adicionales utilizando NLTK. Ejecuta las siguientes líneas de código en tu notebook para descargar estos recursos:

```python
import nltk
nltk.download('brown')
nltk.download('cess_esp')
nltk.download('stopwords')
```

## Contenido del Notebook

El notebook contiene código relacionado con la verificación de la Ley de Zipf en diferentes aspectos, incluyendo:

1. Verificación de la Ley de Zipf en Etiquetas POS.
2. Verificación de la Ley de Zipf en Caracteres.
3. Verificación de la Ley de Zipf en n-gramas.

También se explora el uso de la Ley de Zipf para obtener stopwords en el procesamiento de lenguaje natural.

