# Práctica 7 - Modelos de Lenguaje
## Presentación
Usando el corpus *Don Quijote* del proyecto Gutenberg se crean modelos de bigramas y trigramas que se evalúan con perplejidad.


Como se ve en el capítulo 3 del Jurafsky, mientras que el estándar para evaluar modelos de lenduaje es a través de una evaluación extrìnseca que muestra sus capacidades in vivo, es más sencillo llevar a cabo una evaluación intrínseca del modelo. Mas, la probabilidad de una cadena por sí misma no nos brinda tanta información. 

La perplejidad es una medida enriquecida de la probabilidad de una cadena W. Si W tiene longitud n, la perplejidad de W con respecto a un modelo con cierto vocabulario es la raíz n-ésima del inverso de la probabilidad conjunta de cada w\_i de W en el modelo. Si el programa se queda _perplejo_ ante la nueva cadena, esta va a tener su baja probabilidad. En consecuencia la perplejidad va a crecer, pero se normaliza con la raíz n-ésima. Para evaluar al modelo, calcularemos la probabilidad conjunta del conjunto de prueba
## Instalación y ejecución
1. Se requiere instalar los siguientes paquetes y dependencias en el entorno de Python donde se vaya a ejecutar el programa. 
	1. numpy
	2. re
    3. requests
    4. itertools
    5. collections
    6. sklearn


3. Ejecutar `p07-lang-models.py`.

## Resultados
La evaluación debería salir como que el modelo de trigramas debería tener perplejidad que el de los bigramas. Sin embargo, esto puede ser consecuencia de las limitaciones computacionales del entrenamiento. Por la ley de Heap, hay un incremento considerable en los primeros tokens del corpus, y como mi computadora está limitada al 1.32615442% de las oraciones del corpus total, problemente se requiere de mayor poder computacional para llegar al punto en el que el incremento de tipos sea menor al de tokens. El aumento de trigramas es mayor al de bigramas conforme aumentan los tipos, y esto aunado a la interpolación de Laplace, puede hacer que el modelo sea más burdo pese a tener mayor memoria. 


Evaluando con la oración "de la jamás vista ni oída aventura que con más libros de con otras cosas dignas de su" nos resulta la tabla

Oraciones | Log-perplejidad (n=2) | Log-perplejidad (n=3)
---|---|---
50 | 0.26897549800290677 | 0.16357994328702447 
100 | 1.8285182934263078 | 0.9758350613526244
150 | 3.63958431340518 | 3.419594543816276
200 | 3.848509495212295 | 3.6162274492746893
250 | 4.444155821641341 | 4.588050231147675
300 | 4.595662395725735 | 4.7390811625607805
350 | 4.830820399752612 | 5.047611981634018
400 | 4.648390284655992 | 4.872677207799188
450 | 4.739394887507609 | 4.976875406718325
500 | 4.811185268462868 | 5.0859693573077935


Evaluando con todo el corpus de prueba los resultados de perplejidad son

Oraciones | Log-perplejidad (n=2) | Log-perplejidad (n=3)
---|---|---
50 | 1.845826690498331 | 1.845826690498331
100 | 4.273092770173895 | 4.364481813225362
150 | 4.551137926712067 | 4.820143287032925
200 | 4.822925078020575 | 5.084923169914585
250 | 4.822279826733306 | 5.1339988770008285
300 | 4.957165691715251 | 5.332486362522087
350 | 5.114254544631499 | 5.4972209773779745
400 | 5.162810349228414 | 5.599142148039801
450 | 5.236482688119362 | 5.713556917053857
500 | 5.327623896651925 | 5.812170905963278
