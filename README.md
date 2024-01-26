### Trabajo de titulación previo a la obtención del título de Ingeniero en Biomédicina
## DISEÑO E IMPLEMENTACIÓN DE UN SISTEMA BIOMÉDICO PARA LA CLASIFICACIÓN DE ENFERMEDADES CARDIOPULMONARES MEDIANTE EL ANÁLISIS DE SEÑALES ACÚSTICAS USANDO MACHINE LEARNING

Running the simulation
It is recommended to create a new virtual environment with python version 3.11.0 as this version of Python was used for development.

Create a virtual environment with

conda (recommended): https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

venv: https://docs.python.org/3/library/venv.html

In the new environment navigate to the cloned project:
run ´pip install -r requirements.txt´ to install the required Python libraries
set desired simulation parameters in Configs/base.yaml. You can use a different file by editing env_vars.yaml (all configs files must be in the Cofigs directory)
run python blockchain.py
Each module is extensively described with docstring comments. If the above example executes correctly you can start using it and extending it as necessary for your work.
