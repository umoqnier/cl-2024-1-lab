# Práctica 2: Morfología y análisis morfógico

Elaborado por: Alejandro Axel Rodríguez Sánchez  
Correo: [ahexo@ciencias.unam.mx](mailto:ahexo@ciencias.unam.mx)  
Github: [@Ahexo](https://github.com/Ahexo/)  
Número de Cuenta: 315247697  
Institución: Facultad de Ciencias UNAM  
Asignatura: Lingüística computacional  
Grupo: 7014  
Semestre: 2024-1

## Para ejecutar el script

Se requiere Python 3.3+ para ejecutar esta práctica.

1. Se recomienda generar un nuevo entorno virtual de Python, esto se hace por medio del comando:
```sh
virtualenv venv
```
En el directorio donde está ubicado el script, seguido de:
```sh
source /bin/activate
```
el cual activará el entorno.

2. Instalar las dependencias, estas se tratan de los paquetes [requests](https://pypi.org/project/requests/), [pandas](https://pandas.pydata.org), [nltk](https://www.nltk.org), [spacy](https://spacy.io) y [tabulate](https://pypi.org/project/tabulate/), los cuales ya se encuentran especificados en el archivo `requierements.txt`, así que bastará con ejecutar el siguiente comando para instalarlos: 
```sh
pip install -r requirements.txt
```

3. El paquete spacy necesita de dependencias extra para ejecutar este ejercicio, así que se precisa ejecutar el comando:
```sh
python -m spacy download en_core_web_sm
```

4. Solo queda correr el script `practica02.py`.

Adicionalmente, se adjunta el notebook de Jupyter original por si se opta por correrlo en [Google Colaboratory](https://colab.research.google.com/).
