# Práctica 3: Implementación de un etiquetador POS usando CRFs

## Para correr la práctica es necesario tener instalados los paquetes:
  - scikit-learn
  - sklearn-crfsuite
  - unidecode
  
Se pueden instalar con pip install como vimos en el laboratorio

> pip install scikit-learn
> 
> pip install -U sklearn-crfsuite
> 
> pip install unidecode

## Ejecución

Para ejecutar la prática sólo es necesario correr el comando:
  > python3 Practica3.py

Se debe correr dentro de la carpeta practica3, y se motrará la información solicitada :)

## Comparación y reporte de resultados 
En mi modelo podemos ver que obtuve 
- **Accuracy:** 0.94
- **Presicion:** 0.77
- **Recall:** 0.78
- **F1-score:** 0.94

Comparado con los resultados que obtuvimos en el laboratorio creo que me fue bastante bien considerando que estamos trabajando con bajos recursos, el tamaño del corpus que trabajamos en el laboratorio es más grande del de la práctica y creo que si tuviéramos un corpus más grande se podrían obtener resultados mucho mejores, pues como hemos visto en las clases mientras más datos le demos al modelo este trabajará mucho mejor y va a predecir de mejor manera.

## Análisis extra
- Hacer un análisis breve de los resultados
    - ¿Qué diferencias encuentran entre trabajar con textos en español y en Otomí?
  
      Las principales diferencias que noté es que el etiquetado del otomí puede ser más complejo (al menos el que usamos en esta práctica), y que el trabajar con un idioma que no es tan hablado/estudiado nos vamos a enfrentar a no tener un corpus grande para entrenar a nuestro modelo. Otra cosa es que sin la ayuda de la profesora que nos dio algunos hints para la feature list me hubiera costado mucho trabajo analizar el idioma para saber cuáles eran las características más importantes en el otomí, pues es un idioma desconociod para mí. 
    - ¿Se obtuvieron buenos o malos resultados? ¿Porqué?
    
      Considero que para no saber nada de otomí y tener un corpus relativamente pequeño, pude llegar a un buen accuracy y en general creo que tener casi el 80% de las demás métricas es un buen resultado a mi parecer, basándome en las métricas creo que si hubieron buenos resultados :) me gustaría probar el modelo con más datos para comprobar si sí es realmente bueno  