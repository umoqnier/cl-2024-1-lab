# Práctica 8
**Adriana Michel Ávila García**

## Notas importantes

El corpus elegido fue el de Náhuatl.

El modelo traduce de Náhuatl a Español (Náhuatl es la lengua fuente).

Se agrega la carpeta *nmt-practica8*, que contiene el notebook *nmt-practica8.ipynb*. Este notebook está pensado para ejecutarse en Google Colab, dentro de una carpeta de drive del mismo nombre. La carpeta de drive debería tener la siguiente estructura para ejecutar el notebook:
```
    nmt-practica8
        |-- built_vocab
        |-- models
        +-- nmt-practica8.ipynb
```

(donde *built_vocab* y *models* son carpetas vacías)

La carpeta *nmt-practica8* que incluimos tiene los archivos `translated` que se generan al correr el notebook (`test-practice.translated` y `test-practice.translated.desubword`), y los archivos test ya preprocesados (`test.es-filtered.es.subword` y `test.nah-filtered.nah.subword`).

En el siguiente drive está la carpeta en la que yo ejecuté el notebook (tiene todos los archivos generados): https://drive.google.com/drive/folders/1qYlO81zN1hE3zpRNeyK-MLjxI9LiKzRW?usp=sharing

El modelo entrenado se encuentra en la carpeta *models* del drive.

**Es importante que el notebook se ejecute en un ambiente con GPU**.

## Requerimientos
Todos los paquetes necesitados se instalan en el notebook.

## Evaluación del modelo
Las puntuaciones BLEU y ChrF del modelo se obtuvieron con la biblioteca *sacrebleu*, que es la utilizada en el script `evaluate.py` del repositorio de *AmericasNLP 2021* (https://github.com/AmericasNLP/americasnlp2021/blob/main/evaluate.py).

BLEU y ChrF obtenidos:
- BLEU: 0.77
- ChrF: 13.92

BLEU y ChrF baseline (https://github.com/AmericasNLP/americasnlp2021/tree/main/baseline_system#baseline-results):
- BLEU: 0.33
- ChrF: 0.182

Tanto el BLEU y el ChrF superaron la baseline.

## Extra
Investigar porque se propuso la medida ChrF en el Shared Task

ChrF es una métrica recomendada para lenguajes morfológicamente ricos, ya que toma en cuenta n-gramas a nivel caracter, en vez de nivel palabra. Esto permite que se le pueda dar una calificación aceptable a una traducción de una palabra aún cuando la traducción no es idéntica a la esperada.

Es posible que se haya decidido usar esta métrica porque los lenguajes originarios de América son en su mayoría polisintéticos.

- ¿Como se diferencia de BLEU?

    *BLEU* genera su puntuación basándose en n-gramas de palabras, y no toma en cuenta variaciones morfológicas, mientras que *chr-F* calcula la similitud basado en n-gramas de caracteres.

- ¿Porqué es relevante utilizar otras medidas de evaluación además de BLEU?

    Porque BLEU es muy rígido, y no toma en cuenta variaciones morfológicas, sinónimos, ni la semántica de la oración. Por lo tanto, hay casos en los que otras medidas tienen un desempeño mucho mejor.