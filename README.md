# Un tutorial sobre DSPy y si la ingeniería de prompts automatizada está a la altura de las expectativas
La premisa de DSPy es fascinante: ¿y si pudiéramos entrenar prompts de la misma manera que entrenamos los parámetros de un modelo? Esta idea ha mostrado potencial en entornos académicos, liderada por la investigación de Stanford. En otro artículo, investigadores de VMWare demostraron que la optimización automatizada de prompts (impulsada por DSPy) superó a los prompts ajustados manualmente por humanos.

Siguiendo esta línea, IEEE publicó una perspectiva titulada "La Ingeniería de Prompts Está Muerta". Hacen una afirmación audaz:

Según un equipo de investigación, ningún humano debería optimizar prompts manualmente nunca más.

# Exploración de DSPy con un ejemplo de pocos ejemplos
En este tutorial, exploraremos DSPy utilizando un ejemplo característico de prompting con pocos ejemplos (few-shot prompting) y veremos cómo se desempeña.

## Paso 1: Configuración del Entorno
Antes de comenzar, debemos asegurarnos de tener el entorno adecuado configurado. Esto incluye la instalación de las bibliotecas necesarias.


pip install dspy

## Paso 2: Importación de Librerías
Importamos las librerías necesarias para el uso de DSPy.

import dspy
import numpy as np


## Paso 3: Definición del Problema
Definimos el problema que queremos abordar con DSPy. Para este ejemplo, usaremos una tarea de clasificación de texto.


### Ejemplo de datos de entrada y salida
entradas = ["El clima está soleado", "Va a llover", "Está nublado"]
salidas = ["soleado", "lluvioso", "nublado"]
Paso 4: Creación del Modelo DSPy
Creamos un modelo de DSPy y lo entrenamos con nuestros datos.


### Inicialización del modelo DSPy
modelo = dspy.DSPyModel()

### Entrenamiento del modelo
modelo.train(entradas, salidas)
Paso 5: Optimización Automática de Prompts
Utilizamos DSPy para optimizar los prompts automáticamente.

### Optimización de prompts
prompts_optimizados = modelo.optimize_prompts(entradas)
print("Prompts optimizados: ", prompts_optimizados)
Paso 6: Evaluación del Modelo
Evaluamos el desempeño del modelo con los prompts optimizados.


### Evaluación del modelo
resultados = modelo.evaluate(entradas, salidas)
print("Resultados: ", resultados)


# Conclusión
En este tutorial, exploramos cómo DSPy puede automatizar la optimización de prompts, superando incluso a los prompts ajustados manualmente por humanos. A medida que la tecnología avanza, es posible que veamos un cambio significativo en la forma en que abordamos la ingeniería de prompts, confiando cada vez más en herramientas automatizadas como DSPy.

DSPy promete ser una herramienta poderosa para mejorar la eficiencia y efectividad en la creación de prompts, y su potencial se está demostrando en diversos estudios e investigaciones. La afirmación de que "ningún humano debería optimizar prompts manualmente nunca más" puede parecer audaz, pero los resultados preliminares sugieren que esta podría ser la dirección futura en el campo de la inteligencia artificial.







