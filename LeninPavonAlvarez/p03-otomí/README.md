# Práctica 02 - Morfología
## Presentación
Usando el corpus en otomí de Mijangos, lo procesa, y entrena un modelo POS. Posteriormente, se reportan sus métricas.
## Instalación y ejecución
1. Instalar los siguientes paquetes en el entorno de Python donde se vaya a ejecutar el programa
	1. sklearn (Entrenamiento del modelo)
	2. numpy (Permite descargar punkt)
	3. unidecode (Procesamiento del lenguaje)
	4. rich (Visualización en terminal)

2. Ejecutar `p03-pos-tagging.py` en la misma carpeta donde se encuentre el corpus.
## Comparación de resultados
Notemos que a diferencia del modelo visto en clase que tenía buenas metricas a lo largo de todas las categorías, nuestro modelo tiene picos en los cuales es muy precisa, y momentos en los cuales comete muchos errores. Comparando métricas
|Métrica| Clase| Modelo|
|-|-|-|
|Accuracy|0.9719|0.8967|
|Precision|0.9298|0.8380|
|Recall|0.9633|0.7476|
|F1-score|0.9447|0.7247|

## Notas
El Corpus de Mijangos contenía siertas variaciones en las etiquetas que levantaban un error al momento de entrenar al modelo. Y otras etiquetas, no generaban un error pero disminuían la precisión del modelo. Estas fueron las correcciones:

| Original | Cambio |
|-|-|
|San|unkwn|
|Andrés|n|
|mexico|n|
|chalma|n|
|chente|n|
|juan|n|
|toluca|n|