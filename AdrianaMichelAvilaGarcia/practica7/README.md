# Práctica 7
**Adriana Michel Ávila García**

## Notas sobre la práctica 
Para correr el notebook necesitas tener instalado:
- nltk
- numpy
- pip (instala subword-nmt)

Para correr el script:
- nltl
- subword-nmt

Se incluye el notebook creado. No incluí un script porque no salía usar subword-nmt sin el notebook.
Para ver detalles acerca de los modelos, el preprocesamiento y todo, puedes ver el notebook.

Nota: se usó tokenización por subpalabras, como se vio en la práctica 5.

## Cálculo de la perplejidad
En el libro de Jurafsky (y en clase) se define la perplejidad como:
$$ perplexity(W) = \sqrt[N]{ \prod_{i=1}^N \frac{1}{p(w_i|w_{1} ... w_{i-1})} } $$

Donde $W$ es una cadena de prueba, $N$ es la cantidad de tokens en $W$, y $w_i$ es el *i-ésimo* token en $W$. En clase, la profesora nos explicó que $W$ se puede obtener concatenando las cadenas del conjunto de prueba, o que podíamos calcular la perplejidad de cada cadena del conjunto de prueba, y luego obtener una media ponderada. En esta práctica decidí realizar la primera opción, y concatené las cadenas del conjunto de prueba. 

Para **bigramas**, en el libro de Jurafsky se nos muetsra que la perplejidad se calcula como:
$$ perplexity(W) = \sqrt[N]{ \prod_{i=1}^N \frac{1}{p(w_i|w_{i-1})} } $$

De acuerdo a Jurafsky, la perplejidad (a veces abreviada como PPL) de un modelo lenguaje en un conjunto de prueba la probabilidad inversa del conjunto de prueba, normalizada por el número de palabras.

En las diapositivas de la profesora se menciona la perplejidad logarítmica (log perplexity), definida como el logaritmo de la perplejidad, de la siguiente forma:

$$ \log(Perplexity(W)) = -\frac{1}{N} \sum_{i=1}^N \log_2{p(w_i|w_{1} ... w_{i-1})}  $$

De donde podemos obtener que la perplejidad logarítmica es:
$$logPerplexity(W) = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i | w_1, w_2, \ldots, w_{i-1})}$$

Esta definición es consistente con la encontrada en el sitio de HuggingFace (https://huggingface.co/docs/transformers/perplexity), en donde se define la perplejidad como:
$$PPL(W) = exp ( -\frac{1}{t} \sum_{i=1}^{t} \log P(x_i | x_{<i}) )$$

## Evaluación de los dos modelos creados

La perplejidad obtenida para el modelo de bigramas fue: 119.75132838700031

La perplejidad obtenida para el modelo de trigramas fue: 356.80767789670006

El modelo mejor evaluado fue el de bigramas. Creo que es porque hay muchos más ejemplares de los bigramas que de los trigramas (es decir, cada bigrama aparece más frecuentemente que cada trigrama).