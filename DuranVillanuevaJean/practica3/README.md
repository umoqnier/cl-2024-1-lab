# PREGUNTAS -------------------------

## ¿Qué diferencias encuentran entre trabajar con textos en español y en Otomí?

Que la estructura del idioma es bastante distinta a lo que había notado con otros idiomas más parecidos: italiano, español, ingles.

Además, que supuso todo un reto manejar estos datos por mi corta experiencia, pues el modelo no podía codificar los datos sin un tratamiento previo.

## ¿Se obtuvieron buenos o malos resultados? ¿Porqué?

Creo que los resultados fueron favorables si los comparo con los de mis compañeras que obtuvieron resultados similares, aun cuando ellas me comentan que agregaron más atributos a los datos de entrenamiento (en X en particular). En general, creo además que tuve resultados aceptables, sobre todo con Accuracy (que refleja que tan bien, donde 0 es el peor resultado y 1 es el mejor, se predijo correctamente a los datos). Tuve flaqueza con lás demás metricas (precision, recall, f1-score), aunque se mantuvieron constantes entre 0.73 y 0.75.

Creo además que no podría emitir un juicio de alto valor, pues desconozco los resultados de las metricas en las que se puede considerar como un modelo que obtuve "buenos resultados". Más aún por la circunstancia en la que nos encontramos como lo es los bajos recursos para abstraer la informacion, y probablemente pocos resultados previos publicos sobre el otomí (y en especifico esta variante).  

# EJECUCION -------------------------

# Antes de ejecutar el proyecto

Es necesario crear un entorno virtual y ahí descargar los elementos necesarios para poder ejecutar 'p3.py'

Desde '/DuranVillanuevaJean/practica3/', comenzamos creando el entorno virtual:

> python3 -m venv env

Entramos a este

> source env/bin/activate

Ahora descargamos todo lo necesario con los siguientes comandos

> pip install unidecode

> pip install sklearn_crfsuite

> pip install scikit-learn

# Para ejecutarlo:

Desde '/DuranVillanuevaJean/practica3/'.

Si no se ha activado el entorno virtual, entonces:

> source env/bin/activate

Ya en el entorno virtual, escribimos 

> python3 p3.py

# Para salir del entorno virtual:

> deactivate