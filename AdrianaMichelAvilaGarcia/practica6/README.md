# Práctica 6

**Adriana Michel Ávila García**

### Requerimientos:
Paquetes de python que se instalan al correr el notebook:
- numpy 1.24.4
- gensim

Paquetes de python que se deben tener instalados:
- scikit-learn
- matplotlib

### Notas importantes para correr el notebook:
Elegí el modelo eswiki-large-vs500-w6-SKIP_GRAM. Para que el notebook funcione necesita que estén los tres archivos del modelo en la carpeta modelos:
- eswiki-large-vs500-w6-SKIP_GRAM.model
- eswiki-large-vs500-w6-SKIP_GRAM.model.syn1neg.npy
- eswiki-large-vs500-w6-SKIP_GRAM.model.wv.vectors.npy

Github no me dejó subirlos porque estaban muy pesados :c ¿podrías por favor ponerlos en la carpeta?

### Comparación de las topologías que se generan con cada algoritmo:
Después de realizar la reducción de la dimensionalidad con diferentes conjuntos de vectores al azar, notamos que usando el algoritmo t-SNE, los vectores están más separado, y distribuidos más equitativamente en todo el espacio. Mientras que usando SVD y PCA están más juntos y concentrados en una pequeña sección del espacio.