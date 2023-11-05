# Práctica 5 

Para correr la práctica es necesario tener instalados los paquetes: 

- requests
- elotl
- subword-nmt
- collections
- math
- os
Todos se pueden obtener con pip install

Y se corre con

> $ python3 Practica5 

Estando dentro de la carpeta practica5

## ¿Cómo medir la entropía de un texto? 
Como vimos en la clase, en un texto T con un vocabulario de palabras 
$$ V = \{t_1, t_2,..., t_n\}$$ 
 
La entropía está definida como: 
 
$$ 
H(T) = -\sum_{i=1}^{n}  p(t_i) \log_2p(t_i) 
$$

## Preguntas 
- ¿Aumentó o disminuyó la entropía para los corpus...
    - Nahuatl?
      - Disminuyó, en esta lengua puse cantidad de símbolos = 200 y me pareció que disminuyó bastante, por lo que dejé ese número
    - Español?
      - Aunque también disminuyó, tuve que hacer varias pruebas disminuyendo el número de símbolos, empecé con 300 y la disminución de la entropía era de decimas, por lo que al final dejé 80 símbolos porque me parecía que con esos disminuyó lo suficiente
- ¿Qué significa que la entropía aumente o disminuya en un texto?
  - Valores altos de entropía significan que el texto es más complejo, menos predecible, pues hay mayor diversidad de tipos. Las lenguas ricas en morfología tienen una entropía alta.
- ¿Cómo influye la tokenizacion en la entropía de un texto?
  - Al tokenizar el texto hacemos que haya menos diversidad de tipos, pues estamos separando las palabras (en caso de bpe) en las secuencias de caracteres que más aparecen, esto ocasiona que los tipos de nuestro texto sean menos y por lo tanto vamos a tener una menor diversidad de tipos, que ocasionará una entropía baja y por lo tanto un texto más predecible