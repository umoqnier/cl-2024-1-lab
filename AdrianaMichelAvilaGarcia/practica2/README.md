
# Práctica 2
Adriana Michel Ávila García

### ¿Cómo correr el script?
Ejecutar `python Practica2.py` en esta carpeta.

### Dependencias necesarias para correr el script:
El archivo Practica2.py requiere que estén instalados los siguientes paquetes:
- requests
- pandas
- nltk
- spacy
- tabulate
- (y random)

Pueden ser instalados con conda o pip.

Tabulate se instala con conda corriendo lo siguiente:
`conda install -c conda-forge tabulate`

Y nltk corriendo lo siguiente:
`conda install -c anaconda nltk`

Además, para que funcione, se sebe tener descargado el modelo **en_core_web_sm** de spacy. Lo puedes descargar ejecutando lo siguiente (en el notebook debe ejecutarse en raw):
`!python -m spacy download en_core_web_sm`

### Notas:
El script **sólo hace el análisis de las oraciones en inglés**, ya que es el único idioma que se encontraba en la intersección de las bibliotecas utilizadas, y los idiomas de la shared task que tenían oraciones disponibles.

En el script se toman 10 oraciones al azar, y para cada una de ellas se imprime una tabla como la siguiente:
```
+----------+--------+----------+--------------------------------------+
| word     | stem   | lemma    | morphological_info                   |
|----------+--------+----------+--------------------------------------|
| Well     | well   | well     | {}                                   |
| kept     | kept   | keep     | {'Tense': 'Past', 'VerbForm': 'Fin'} |
| facility | facil  | facility | {'Number': 'Sing'}                   |
| with     | with   | with     | {}                                   |
| friendly | friend | friendly | {'Degree': 'Pos'}                    |
| staff    | staff  | staff    | {'Number': 'Sing'}                   |
| .        | .      | .        | {'PunctType': 'Peri'}                |
+----------+--------+----------+--------------------------------------+
```