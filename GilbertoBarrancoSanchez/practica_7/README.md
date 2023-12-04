# Perplejidad 

## ¿Cómo calcular la perplejidad de un modelo?

Si tenemos un corpus de evaluación(W) de un modelo de n-gramas, que se compone de N oraciones, podemos calcular la perplejidad de nuestro modelo mediante la probabilidad P(w1, ... ,  wN), donde cada wi es una oración correspondiente al corpus de entrenamiento. 

Denotamos P := P(w1, ... , wN ). Entonces la perplejidad de nuestro modelo es: 

perplexity(W) = (P)^{-1/N}

Notemos que la expresión anterior puede tomar distintas formas, dependiendo de n (en nuestro modelo de n-gramas), pues podemos apoyarnos de esto y el Teorema de Bayes para obtener el valor de P.

## En la práctica 

¿Cuál fue el modelo mejor evaluado?¿Por qué?

En nuestro caso, obtuvimos una perplejidad de infinito en ambos casos :(. 

Lo anterior fue debido a que nuestro vocabulario generado era muy pequeño. Por esta razón, varias probabilidades de ciertas palabras se consideraban como 0 y el parámetro P, mencionado anteriormente, resultaba muy pequeño. Por otra parte, también el valor 1/N resultaba ser casi nulo y hacía creer al programa que se trataba de una división entre cero. Desafortunadamente, no logré corregir ese error :(. 

