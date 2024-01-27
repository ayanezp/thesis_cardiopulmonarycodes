### Trabajo de titulación previo a la obtención del título de Ingeniero en Biomédicina
## DISEÑO E IMPLEMENTACIÓN DE UN SISTEMA BIOMÉDICO PARA LA CLASIFICACIÓN DE ENFERMEDADES CARDIOPULMONARES MEDIANTE EL ANÁLISIS DE SEÑALES ACÚSTICAS USANDO MACHINE LEARNING


Se recomienda crear un nuevo entorno local con la versión 3.11.0 de Python, ya que esta versión de Python es la que se utilizó para el desarrollo de este proyecto.

## Características del equipo utilizado como entorno local:
* Procesador: 13th Gen Intel(R) Core(TM) i5-13600KF
* Memoria Ram: 32,0 GB DDR5 a 6000 MHz
* Tarjeta grafica: GPU NVIDIA GeForce RTX 3070 8 GB
* Disco duro: SPCC M.2 PCIe SSD de 2 TB 

En el nuevo entorno, para la ejecucion del primer codigo:
1) Ejecutar `pip install -r Requerimientos_Primer_Codigo.txt` para instalar las bibliotecas de Python necesarias.
2) Cambiar la ruta donde esta ubicada tanto el set_a como el set_b de datos en la linea 25 del primer codigo, considerar siempre utizar la estructura que esta proporcionada como ejemplo a manera de comentario.
3) Ejecutar `Primer_Codigo.py`
4) Cada módulo está comentado correctamente. Si el ejemplo anterior se ejecuta correctamente, se puede comenzar a usarlo y extenderlo según sea necesario.
5) Base de datos usada en el primer codigo: https://www.kaggle.com/datasets/kinguistics/heartbeat-sounds/download?datasetVersionNumber=1

En otro entorno, para la ejecucion del segundo codigo:
1) Ejecutar `pip install -r Requerimientos_Segundo_Codigo.txt` para instalar las bibliotecas de Python necesarias.
2) Cambiar la ruta donde esta ubicada tanto el directorio de los audios como la base de datos tanto en la linea 36 como 37 del segundo codigo.
3) Ejecutar `Segundo_Codigo.py`
4) Cada módulo está comentado correctamente. Si el ejemplo anterior se ejecuta correctamente, puede comenzar a usarlo y extenderlo según sea necesario.
5) Base de datos usada en el segundo codigo: https://www.kaggle.com/datasets/arashnic/lung-dataset/download?datasetVersionNumber=1
