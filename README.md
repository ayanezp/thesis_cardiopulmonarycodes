# Trabajo de titulación previo a la obtención del título de Ingeniero en Biomédicina
DISEÑO E IMPLEMENTACIÓN DE UN SISTEMA BIOMÉDICO PARA LA CLASIFICACIÓN DE ENFERMEDADES CARDIOPULMONARES MEDIANTE EL ANÁLISIS DE SEÑALES ACÚSTICAS USANDO MACHINE LEARNING

Este trabajo utiliza señales acústicas de pacientes extranjeros para desarrollar un sistema biomédico avanzado, aplicando técnicas de machine learning. Se busca mejorar la precisión en el diagnóstico de enfermedades cardiopulmonares mediante un enfoque integral que combina conjuntos de datos específicos, un detallado preprocesamiento de señales y la extracción de características relevantes. La fase de clasificación, respaldada por modelos de aprendizaje automático optimizados, promete contribuir significativamente a la detección temprana y tratamiento efectivo de estas afecciones.
Adquisición de Datos:
Este estudio tiene como objetivo el uso de técnicas de machine learning para clasificar enfermedades cardio pulmonares a partir de conjuntos de datos correspondientes a sonidos cardíacos [pie de pagina] y pulmonares [pie de pagina]. Kaggle, una de las plataformas de Data Science más reconocidas a nivel mundial, fue utilizada para la obtención de los datos, debido a que no existen registros de sonidos pulmonares y cardiacos en el Ecuador. Los conjuntos de datos incluyen grabaciones de alta calidad de pacientes saludables y no saludables de origen extranjero y fueron extraídos con estetoscopios electrónicos. El conjunto de datos contiene grabaciones de audio del examen de la pared torácica en varios puntos estratégicos. La colocación del estetoscopio en el sujeto fue determinada por el médico especialista que realizó el diagnóstico. Cada grabación fue replicada tres veces correspondientes a varios filtros de frecuencia que enfatizan ciertos sonidos corporales.
para clasificar enfermedades cardio pulmonares en pacientes de origen extranjero, esto debido a la falta de datos acústicos de enfermedades cardio pulmonares en el Ecuador (los detalles específicos sobre el país de origen se encuentran en Kaggle). Las señales fueron extraídas mediante el uso de estetoscopios electrónicos y manuales, y se emplearon dos conjuntos de datos distintos para abordar de manera complementaria el problema de estudio: la clasificación de enfermedades cardiopulmonares mediante el análisis de señales acústicas.

El primer conjunto de datos se centra en la clasificación de sonidos cardíacos, incluyendo categorías como sístoles, extrasístoles, y otros sonidos relacionados con el funcionamiento del corazón. Utilizó el conjunto de datos proporcionado por Fraiwana et al. [1], que se enfoca en sonidos pulmonares grabados desde la pared torácica utilizando un estetoscopio electrónico. Este conjunto de datos presenta grabaciones de 112 sujetos, abarcando diversas condiciones pulmonares, como asma, insuficiencia cardíaca, neumonía, entre otras. Cada grabación se replicó tres veces con diferentes filtros de frecuencia, permitiendo un análisis detallado de las características acústicas específicas de cada condición. La disponibilidad pública de este conjunto de datos en Mendeley Data [1] facilita su acceso para investigadores y profesionales interesados en el desarrollo de algoritmos de detección de enfermedades pulmonares mediante el análisis de sonidos respiratorios.

Información sobre el primer conjunto de datos:

Nombre del Conjunto de Datos: Conjunto de Datos de Sonidos Pulmonares de Fraiwana et al.
Origen: Jordan University of Science and Technology, Jordan.
Tipo de Señales: Sonidos Pulmonares.
Número de Sujetos: 112 sujetos (35 saludables y 77 con diversas condiciones pulmonares).
Filtros de Frecuencia: Tres filtros aplicados en cada grabación para resaltar diferentes características.
Disponibilidad: Público en Mendeley Data [1].
Por otro lado, el segundo conjunto de datos se centra en la clasificación de sonidos pulmonares, abarcando condiciones como neumonías, taquicardias, bradicardias, entre otras. Este conjunto de datos, originalmente creado para un desafío de aprendizaje automático destinado a clasificar sonidos cardíacos, fue recopilado a partir de dos fuentes: (A) del público en general a través de la aplicación iStethoscope Pro para iPhone y (B) de un ensayo clínico en hospitales utilizando el estetoscopio digital DigiScope. Las dos tareas asociadas con este conjunto de datos incluyen la segmentación de sonidos cardíacos normales (S1 y S2) en archivos de audio y la clasificación de los sonidos cardíacos en una de cuatro categorías distintas [2]. Este recurso invaluable proporciona etiquetas y metadatos en los archivos set_a.csv y set_b.csv, así como información de temporización de referencia en set_a_timing.csv. Los archivos de audio, con longitudes variables entre 1 segundo y 30 segundos, son cruciales para el desarrollo y evaluación de métodos de clasificación de sonidos cardíacos [2].

Información sobre el segundo conjunto de datos:

Nombre del Conjunto de Datos: Conjunto de Datos de Sonidos Cardíacos para Clasificación.
Origen: Kaggle, recopilado a través de la aplicación iStethoscope Pro y ensayo clínico con DigiScope.
Tipo de Señales: Sonidos Cardíacos.
Tareas Asociadas: Segmentación de S1 y S2, Clasificación de Sonidos Cardíacos.
Número de Categorías: Cuatro categorías distintas de sonidos cardíacos.
Disponibilidad: Público en Kaggle [2].
La combinación de estos dos conjuntos de datos permite un análisis exhaustivo de las enfermedades cardiopulmonares desde dos perspectivas complementarias. Mientras que uno proporciona una visión profunda de la salud cardíaca, el otro ofrece una perspectiva igualmente importante sobre la salud pulmonar. Juntos, estos conjuntos de datos forman una base sólida para un sistema de clasificación integral que puede identificar una gama más amplia de condiciones cardio pulmonares, mejorando así la precisión del diagnóstico y la eficacia del tratamiento.


Preprocesamiento de Señales

Código 1:

Filtrado de Datos por Duración: En este código, se realiza la limpieza y filtrado de ruido mediante la aplicación de técnicas de procesamiento de señales. El código carga archivos de audio, descarta aquellos con una duración menor a 3 segundos y luego segmenta la señal en intervalos de tiempo específicos, aplicando técnicas de aumento de datos para mejorar la robustez del modelo ante variaciones en tono y velocidad.

Normalización de Señales: La duración de las señales se ajusta para asegurar que tengan una longitud consistente y se utilizan técnicas de aumento de datos para diversificar el conjunto de entrenamiento.

Segmentación y Ventaneo: La señal se segmenta en intervalos de 3 segundos con un desplazamiento adecuado para garantizar la cobertura total. La técnica de ventaneo se aplica durante la extracción de características para dividir la señal en tramas solapadas.

Reducción de Dimensionalidad: La dimensión de las características extraídas se mantiene constante y se ajusta mediante relleno o truncamiento.

Aumento de Datos (Data Augmentation): El primer código destaca la aplicación de técnicas de aumento de datos para mejorar la diversidad y robustez del conjunto de entrenamiento. Se realiza un cambio de tono y velocidad en las señales, lo que contribuye a mejorar la generalización del modelo ante variaciones en las condiciones de grabación.

Ponderación de Clases: Se aplica ponderación de clases para abordar posibles desequilibrios en la distribución de clases en el conjunto de datos. Esto asegura que el modelo se entrene de manera equitativa en todas las clases, mejorando así su capacidad para reconocer diferentes categorías.

División de Conjuntos de Datos: El conjunto de datos se divide en conjuntos de entrenamiento y prueba, permitiendo evaluar la capacidad de generalización del modelo en datos no vistos.

Codificación de Etiquetas: Se utiliza la codificación de etiquetas (LabelEncoder) para convertir las etiquetas de clase en valores numéricos, facilitando la tarea de clasificación para el modelo.

Código 2:

Limpieza y Filtrado de Ruido: En este segundo código, se realiza la limpieza y filtrado de ruido a través de la creación de espectrogramas a partir de archivos de audio. La función create_spectrogram utiliza la transformada de Fourier para calcular el espectrograma, que visualiza cómo varía la intensidad de frecuencia en función del tiempo.

Normalización de Señales: Se utiliza la función adjust_spectrogram_size que se encarga de ajustar el tamaño de los espectrogramas para que todos tengan la misma longitud.

Reducción del Tamaño de Espectrogramas: Se generan los espectrogramas para la señal completa. La función adjust_spectrogram_size realiza ventaneo para ajustar el tamaño de los espectrogramas.

Información Extraída de Nombres de Archivos: El código extrae información relevante de los nombres de archivos, como el diagnóstico, tipo de sonido, ubicación, edad y género. Esta información se utiliza para etiquetar y asociar cada archivo de audio con datos específicos.

Extracción de Características Espectrales: La extracción de características se realiza indirectamente mediante la creación de espectrogramas. Estos proporcionan información detallada sobre la variación de la energía de la señal con respecto al tiempo y la frecuencia.

Aumento de Datos (Data Augmentation):  Se procede con la creación de espectrogramas proporciona una representación diversificada de las señales de audio.

Extracción de Características

Para ambos códigos:

Transformada de Fourier de Corto Tiempo (STFT): La transformada de Fourier de corto tiempo se utiliza implícitamente en la extracción de características al calcular los espectrogramas de Mel y otros descriptores espectrales.

Espectrogramas de Mel: Se calculan los espectrogramas de Mel utilizando la transformada de Mel, proporcionando una representación visual de cómo varía la energía de la señal con el tiempo en diferentes frecuencias.

Extracción de Características Espectrales: Las características espectrales se extraen utilizando la transformada de Mel y se calculan los espectrogramas de Mel, los coeficientes espectrales en las frecuencias de Mel (MFCC), el cromagrama y el contraste espectral.

Coeficientes Cepstrales en las Frecuencias de Mel (MFCC): Los MFCCs se obtienen a partir de los espectrogramas de Mel, capturando las características más relevantes del espectro de frecuencia de la señal de audio.

Cromagrama: El cromagrama se extrae para resaltar la información tonal en la señal, mostrando la distribución de energía en diferentes notas musicales a lo largo del tiempo.

Contraste Espectral: El contraste espectral se calcula para medir la diferencia de amplitud entre picos espectrales en diferentes bandas de frecuencia.

Características Temporales: Además de las características espectrales, se incorporan características temporales mediante la segmentación de la señal y la extracción de características en intervalos de tiempo específicos.

Normalización de Características: Se ajustan las características en longitud para asegurar consistencia en la entrada del modelo.

Selección de Características: Se asegura que todas las características tengan la misma longitud mediante relleno o truncamiento.

Selección de Características:

Código 1:

Análisis de Características: Se resalta la importancia de las características espectrales y temporales extraídas, como MFCCs, espectrogramas y contraste espectral, para capturar patrones relevantes en las señales de audio.

Optimización de Características: Se enfatiza la relevancia de las características seleccionadas en la calidad de la clasificación.

Código 2:

Análisis de Características: Se destaca la importancia de las representaciones espectrales y temporales capturadas mediante la creación de espectrogramas y el uso de capas LSTM.


Entrenamiento del Modelo de Machine Learning para la Clasificación de Señales

Para ambos códigos:
Validacion cruzada: Se dividen los conjuntos de datos en 2 partes, una destinada netamente para el entrenamiento de los modelos de machine learning, y la otra que se usará en la etapa de validacion.
Configuración de Hiper Parámetros: Los hiper parámetros del modelo, como la tasa de aprendizaje y los parámetros de las capas, son configurados antes del entrenamiento para ajustarse a la naturaleza del problema.
Callbacks: Se utilizan callbacks como Early Stopping y ReduceLROnPlateau durante el entrenamiento del modelo. Early Stopping ayuda a prevenir el sobreajuste al detener el entrenamiento cuando la métrica de validación deja de mejorar, mientras que ReduceLROnPlateau ajusta dinámicamente la tasa de aprendizaje para mejorar la convergencia.
Monitoreo del Rendimiento: Después del entrenamiento, se realiza un análisis detallado del rendimiento del modelo mediante la visualización de curvas de pérdida y precisión en conjuntos de entrenamiento y validación. Esto proporciona información sobre el proceso de aprendizaje y la capacidad de generalización del modelo.
Codigo 1
Regularización: Se incorpora regularización en la capa densa final del modelo mediante el uso de la regularización L1 y L2. Esto ayuda a prevenir el sobreajuste al penalizar coeficientes grandes y favorecer modelos más simples.
Entrenamiento de Modelos: El modelo de clasificación se construye utilizando una arquitectura de red neuronal convolucional (CNN) con capas de convolución, normalización por lotes, capas de reducción y capas densas. La clasificación se realiza mediante una capa de salida softmax.

Código 2:

Entrenamiento de Modelos: Se utiliza una arquitectura de red neuronal que incluye capas convolucionales, de normalización por lotes, una capa LSTM y capas densas para realizar la clasificación. La función de activación softmax se utiliza en la capa de salida para la clasificación multiclase.



Integración y Pruebas Complementarias
Para ambos códigos:
Integración de Modelos: Una vez entrenados, los modelos se integran en un sistema unificado que puede procesar y clasificar señales tanto cardíacas como pulmonares.

Pruebas Complementarias: Se realizan pruebas para evaluar cómo los modelos trabajan en conjunto. Esto incluye la clasificación de señales mixtas y la evaluación de la precisión del diagnóstico combinado.

Ajustes Finales: Basándose en los resultados de las pruebas, se realizan ajustes finales en los modelos para mejorar su precisión y eficacia en la clasificación de enfermedades cardiopulmonares.
