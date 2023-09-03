# Antes de ejecutar el proyecto

Es necesario crear un entorno virtual y ahí descargar los elementos necesarios para poder ejecutar 'p2.py'

Desde '/DuranVillanuevaJean/practica2/', comenzamos creando el entorno virtual:

> python3 -m venv env

Entramos a este

> source env/bin/activate

Ahora descargamos todo lo necesario con los siguientes comandos

> pip install nltk

> pip install spacy

> pip install pandas

> python -m spacy download en_core_web_sm


# Para ejecutarlo:

Desde '/DuranVillanuevaJean/practica2/'.

Si no se ha activado el entorno virtual, entonces:

> source env/bin/activate

Ya en el entorno virtual, escribimos 

> python3 p2.py

# Para salir del entorno virtual:

> deactivate