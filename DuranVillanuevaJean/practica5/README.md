# PREGUNTAS

- Responder las preguntas:
    - ¿Aumento o disminuyó la entropia para los corpus?
        Disminuyó para ambas.
        - Nahuatl: De 11.98 a 3.50, con el parametro num_symbols=400 
        - Español: De 8.34 a 3.50, con el parametro num_symbols=250
    - ¿Qué significa que la entropia aumente o disminuya en un texto?
        - La entropía se refiere a la medida de la incertidumbre o desorden en un corpus. Y el hecho de que el valor sea alto significa que el texto es mas complejo, variado, menos predecible para modelos que busquen abstraer este. Con un valor bajo, nos indica que el valor tiene menor variedad, lo que lo vuelve mas predecible y un mejor candidato para los modelos que abstraigan este.  
    - ¿Como influye la tokenizacion en la entropía de un texto?
        Este concierte un corpus de texto legible a un texto segmentado (muchas veces este ya no es legible). Existen diferentes metodos en que se hace esta segmentación, pero todas buscan descomponer el texto con la finalidad de dismunuir su entropia y ser mejor candidatos para los modelos que los abstraigan, pues la tokenizacion afecta la forma en que las unidades de texto se representan y organizan.  


# EJECUCION

## Antes de ejecutar el proyecto

Es necesario crear un entorno virtual y ahí descargar los elementos necesarios para poder ejecutar 'p5.py'

Desde '/DuranVillanuevaJean/practica5/', comenzamos creando el entorno virtual:

> python3 -m venv env

Entramos a este

> source env/bin/activate

Ahora descargamos todo lo necesario con los siguientes comandos

> pip install elotl

> pip install requests

> pip install subword_nmt

## Para ejecutarlo:

Desde '/DuranVillanuevaJean/practica5/'.

Si no se ha activado el entorno virtual, entonces:

> source env/bin/activate

Ya en el entorno virtual, escribimos 

> python3 p3.py

## Para salir del entorno virtual:

> deactivate