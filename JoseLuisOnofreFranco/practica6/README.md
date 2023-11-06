# Modelos de lenguaje

## Perplejidad

La perplejidad sirve para tener una métrica sobre qué tan bueno es nuestro modelo prediciendo palabras. Se puede ver como un factor de sorpresa, que entre más bajo, la sorpresa es menor, pues dada nuestra oración a probar, no se sorprende tanta en verla. En cambio si se sorprende, o sea es mayor la perplejidad, quiere decir que el modelo no es tan bueno prediciendo. 

Para calcular la perplexidad, se usa:

$H(W) = - \frac{1}{N} \sum_{i=1}^{N} \log (P(w_i|w_1, \ldots, w_{i-1}))$

que es usando log probabilidades.

## Evaluación de los modelos

Comparando el modelo de bigramas y trigramas, este último resultó tener la perplejidad más baja. En este caso, se obtuvo:

- Bigramas:  57.0
- Trigramas:  25.0

La razón es el modelo de trigramas tiene más información sobre el contexto, por ende, le hace más sentido las oraciones que se le evaluan. Por ejemplo, tener un trigrama (manza, está, madura) a tener (manza, está) y (está, roja) da más contexto, porque también se pueden tener oraciones como "la manzaná está podrida", "su cara está roja", que puede afectar en la predictibilidad de un modelo de bigramas. Sin embargo, los recursos para computar un modelo de bigramas es menor que uno de trigramas.

## Ejecución

### Dependencias para correr la aplicación

- [jupytext](https://github.com/mwouts/jupytext)
- [numpy](https://numpy.org/install/)
- [scikit-learn](https://scikit-learn.org/stable/install.html)

### Cómo ejecutar el archivo

El archivo principal es `lab.py`, el cual se puede leer como un archivo `.ipynb` (gracias al usar el formato `# %%`) de la siguiente manera:

- Con Visual Studio Code.
- Con jupyter usando `jupytext`, que permite leer de manera correcta los textos en Markdown.
