Autor: Mikel Segura Elizalde
Número de cuenta: 420004231
Versión: 1; Septiembre 3, 2023

# Cómo correr el script:

Basta con abrir 2_morphology.py usando Python3.

# Qué dependencias son necesarias y cómo instalarlas.

El script requiere de dos librerías: ntlk y spacey; así como un
paquete de modelo de lenguaje que utilia spacey en el script.

Para instalar ntlk, en la terminal, ejecutar la siguiente línea:

pip install --user -U nltk


Para instalar spacey, así como el paquete del modelo de lenguaje, 
en la terminal, ejecutar las siguientes líneas:

pip install spacy
python3 -m spacy download en_core_web_sm
python3 -m spacy download en
