# Práctica 02 - Morfología
## Presentación
Usando el corpus en inglés de SIGMORPHON 2022 track: sentences, lo almacena en un dataframe, y selecciona al azar 10 oraciones. Posteriormente, se obtiene la siguiente información de cada una de sus palabras componentes:
1. Stem
2. Lemma
3. Información morfológica
## Instalación y ejecución
1. Instalar los siguientes paquetes en el entorno de Python donde se vaya a ejecutar el programa
	1. nltk (Procesamiento del lenguaje)
	2. ssl (Permite descargar punkt)
	3. spacy (Procesamiento del lenguaje)
	4. requests (Permite descargar corpus)
	5. pandas (Procesamiento de datos)
	6. rich (Visualización en terminal)
2. Ejecutar `p02-morphology.py`
3. En caso de no tener punkt instalado se descargará, a veces saldrá una pestaña. En cuyo caso seleccionar `Models>>punkt` y luego descargar. 
## Notas
De los 3 lenguajes de SIGMORPHON, solamente el inglés está disponible para nltk y spacy. Por lo que el código solo usa el corpus en inglés.

En caso de que los paquetes se instalen en Python y VS Code no los reconozca, reinstalarlos usando la siguiente línea de código
```python
python3 -m pip install <pkg> 
```
o, en su defecto
```python
python -m pip install <pkg>
```
