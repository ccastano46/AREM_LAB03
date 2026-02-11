# Laboratorio 3: Clasificación de Imágenes con Redes Neuronales en CIFAR-10

**Autor:** Manus AI (basado en el notebook de AREM_LAB03)
**Fecha:** 10 de febrero de 2026

## 1. Descripción del Problema

Este laboratorio aborda un problema de **clasificación de imágenes multi-clase** utilizando el conjunto de datos CIFAR-10. El objetivo principal es comparar el rendimiento de dos arquitecturas de redes neuronales fundamentalmente diferentes:

1.  Una **Red Neuronal Densa (MLP)**, que sirve como modelo base no convolucional.
2.  Una **Red Neuronal Convolucional (CNN)**, diseñada específicamente para tareas de visión por computadora.

El propósito es demostrar empíricamente por qué las arquitecturas convolucionales son superiores para tareas que involucran datos con estructura espacial, como las imágenes, y analizar las limitaciones inherentes de los modelos densos en este contexto.

## 2. Descripción del Dataset

-   **Nombre**: CIFAR-10
-   **Fuente**: Cargado a través de `tensorflow_datasets`.
-   **Contenido**: 60,000 imágenes a color de baja resolución (32×32 píxeles).
-   **Clases**: 10 clases mutuamente excluyentes: `airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.
-   **Distribución**: El dataset está perfectamente balanceado, lo que significa que cada clase tiene el mismo número de muestras:
    -   **Conjunto de Entrenamiento**: 50,000 imágenes (5,000 por clase).
    -   **Conjunto de Prueba**: 10,000 imágenes (1,000 por clase).
-   **Preprocesamiento**: Para los experimentos, las imágenes se redimensionaron a 64×64 píxeles y sus valores de píxeles se normalizaron al rango `[0, 1]`.

## 3. Arquitecturas Implementadas

Se implementaron y evaluaron dos modelos con arquitecturas distintas.

### 3.1. Modelo 1: Red Neuronal Densa (MLP) - Baseline

Este modelo sirve como punto de referencia. Su diseño aplana la imagen de entrada en un único vector, perdiendo toda la información espacial.

![Arquitectura MLP](mlp_architecture.png)

-   **Capa de Entrada**: Las imágenes de 64×64×3 se aplanan a un vector de 12,288 características.
-   **Capas Ocultas**: Tres capas densas con activación ReLU (512, 256 y 128 neuronas) para aprender relaciones no lineales.
-   **Capa de Salida**: Una capa densa con 10 neuronas y activación Softmax para producir una distribución de probabilidad sobre las 10 clases.
-   **Parámetros Totales**: 6,457,482.

### 3.2. Modelo 2: Red Neuronal Convolucional (CNN)

Esta arquitectura está diseñada para preservar y explotar la estructura espacial de las imágenes mediante el uso de capas convolucionales y de pooling.

![Arquitectura CNN](cnn_architecture.png)

-   **Capa Convolucional**: Una capa `Conv2D` con 32 filtros de 3×3 y activación ReLU. Aprende a detectar características locales como bordes y texturas.
-   **Capa de Pooling**: Una capa `MaxPooling2D` que reduce la dimensionalidad espacial (de 62×62 a 31×31), haciendo la representación más manejable y robusta a pequeñas traslaciones.
-   **Capa de Aplanamiento (Flatten)**: Convierte la salida del bloque convolucional en un vector para las capas densas.
-   **Capas Densas**: Una capa oculta de 128 neuronas (ReLU) y la capa de salida Softmax de 10 neuronas.
-   **Parámetros Totales**: 3,938,570 (significativamente menos que el MLP).

## 4. Resultados de los Experimentos

Ambos modelos se entrenaron durante 20 épocas. A continuación se presenta una tabla comparativa con los resultados finales.

| Métrica | Modelo MLP (Baseline) | Modelo CNN | Interpretación |
| :--- | :--- | :--- | :--- |
| **Accuracy (Entrenamiento)** | 55.70% | **89.57%** | La CNN aprende mucho mejor las características del conjunto de entrenamiento. |
| **Accuracy (Validación/Test)** | 47.79% | **~57.02%** | La CNN generaliza mejor a datos no vistos, superando al MLP en ~10%. |
| **Loss (Entrenamiento)** | 1.2404 | **0.3170** | El error de la CNN en el entrenamiento es mucho menor, indicando un mejor ajuste. |
| **Loss (Test)** | 1.4834 | 1.8955 | Aunque la CNN tiene mayor loss en test, su accuracy es superior. |
| **Overfitting** | Moderado (~8% de brecha) | **Severo (~32% de brecha)** | La CNN, al ser más potente, memoriza los datos de entrenamiento si no se regulariza. |
| **Parámetros** | 6.5 Millones | **3.9 Millones** | La CNN es mucho más eficiente en parámetros. |

## 5. Interpretación y Conclusiones

1.  **Limitación Fundamental del MLP**: El modelo MLP, a pesar de tener más parámetros, se estanca en un rendimiento inferior (~48% de accuracy). Esto se debe a que la capa `Flatten` destruye la estructura espacial de la imagen, tratando los píxeles como características independientes. El modelo no puede aprender conceptos como "un ojo está al lado de una nariz".

2.  **Ventaja de la CNN**: La CNN, aunque simple, supera al MLP con un ~57% de accuracy en validación. Su capa `Conv2D` actúa como un detector de características que se desliza sobre la imagen, preservando las relaciones espaciales. La capa `MaxPooling2D` ayuda a que estas características sean más robustas.

3.  **Eficiencia y Overfitting**: La CNN logra un mejor rendimiento con un 40% menos de parámetros. Esto demuestra la eficiencia de la compartición de pesos en las capas convolucionales. Sin embargo, su mayor capacidad de aprendizaje la hace propensa a un sobreajuste severo, como lo demuestra la gran diferencia entre la precisión de entrenamiento (89%) y la de validación (57%).

**Conclusión Final**: Los experimentos demuestran claramente que las arquitecturas convolucionales son indispensables para tareas de visión por computadora. Un MLP no es una herramienta adecuada para este tipo de problemas. Aunque la CNN simple implementada es superior, sufre de un fuerte sobreajuste, lo que indica que los siguientes pasos lógicos serían mejorar su arquitectura (añadiendo más capas) e introducir técnicas de regularización como **Dropout** y **aumento de datos (data augmentation)** para mejorar su capacidad de generalización.
