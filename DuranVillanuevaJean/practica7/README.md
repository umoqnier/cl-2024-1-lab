# PREGUNTAS
## Perplejidad de un modelo del lenguaje 

La perplejidad es una métrica usada para evaluar el rendimiento de un modelo de lenguaje. Mide que tan bien un modelo del lenguaje predice una muestra del texto. Lo mide en base a un corpus. La métrica tiene como objetivo medir que tan bien un modelo predice a un corpus, por lo que se utilizan las mismas frases de un corpus sobre diferentes modelos de lenguaje para saber cual predice de mejor manera el mismo corpus.

El resultado de la formula sobre una frase, mientras sea mas bajo, el modelo es mejor prediciendo la siguiente palabra. El resultado de de 2, por ejemplo, es equivalente a lanzar una moneda, mientras que resultados mayores a este numeros nos hablan de una muy baja certeza de cual será el resultado.

La formula de la perplejidad es la siguiente

$\text{Perplejidad} = -\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i | w_1, w_2, \ldots, w_{i-1})$

## Evaluar los modelos y reportar la perplejidad de cada modelo. Comparar los resultados entre los diferentes modelos del lenguaje (bigramas, trigramas)

¿Cual fue el modelo mejor evaluado? ¿Porqué?

Los resultados en el modelo de trigramas oscila entre los valores 200 y 220, mientras que para el modelo de bigramas oscila entre los valores 91 y 99. 

El modelo mejor evualuado fue el de trigramas, pues tiene valores más bajos. Lo que significa que predice de mejor menra que el de bigramas. Según lo investigado, el valor que regresa refleja entre cuantas opciones el modelo puede escoger y entre menor el valor significa que tiene menores opciones, y una probabilidad más alta en acertar al momento de predecir. 

# EJECUCION

## Antes de ejecutar el proyecto

Es necesario crear un entorno virtual y ahí descargar los elementos necesarios para poder ejecutar 'p7.py'

Desde '/DuranVillanuevaJean/practica7/', comenzamos creando el entorno virtual:

> python3 -m venv env

Entramos a este

> source env/bin/activate

Ahora descargamos todo lo necesario con los siguientes comandos

> pip install requests

> pip install scikit-learn

## Para ejecutarlo:

Desde '/DuranVillanuevaJean/practica7/'.

Si no se ha activado el entorno virtual, entonces:

> source env/bin/activate

Ya en el entorno virtual, escribimos 

> python3 p7.py

## Para salir del entorno virtual:

> deactivate