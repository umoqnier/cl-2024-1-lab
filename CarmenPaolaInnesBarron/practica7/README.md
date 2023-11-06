# Práctica 7: Evaluación de modelos de lenguaje

Para correr la práctica es necesario tener instalados los paquetes: 

- requests
- re
- sklearn
- collections
- numpy
- itertools 

Todos se pueden obtener con pip install

Y se corre con

> $ python3 Practica7

Estando dentro de la carpeta practica7

## Perplejidad
La perplejidad es una medida con la que podemos evaluar la calidad y el rendimiento de un modelo de lenguaje probabilístico, se utiliza para calificar qué tan bien puede un modelo predecir una secuencia de palabras en función de la probabilidad que asigna a la secuencia.

Una perplejidad baja indica que el modelo es capaz de predecir bien la secuencia de palabras, por lo que es más "coherente" con los datos. Si lo vemos como una medida de sorpresa, podemos decir que el modelo está menos sorprendido de ver esa seceuncia de palabras, al contrario de una perplejidad alta que indica que el modelo tiene dificultades para predecir la secuencia de palabras, o sea que está más sorprendido de verla.

Se puede calcular de la siguiente manera:

$H(W) = - \frac{1}{N} \sum_{i=1}^{N} \log (P(w_i|w_1, \ldots, w_{i-1}))$


La perplejidad se utiliza para evaluar la capacidad de un modelo de lenguaje para predecir secuencias de palabras y se busca minimizarla. Cuanto menor sea la perplejidad, mejor será el modelo.

## Resultados

Después de evaluar el modelo con la fórmula que vimos en clase la diferencia entre ambos era muy pequeña, de décimas, preguntándole a CHATPGT encontré que había una versión de la fórmula en la que se eleva 2 a la H, y usando esa versión la diferencia era más notoria, así que en los resultados dejé la que eleva con exponencial. El modelo de trigramas fue mucho mejor porque con los trigramas tenemos más contexto de las oraciones y gracias a esto se evalúa mejor :)