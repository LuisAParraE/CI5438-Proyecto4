# CI5438-Proyecto4

## Implementación

Se implementaron redes neuronales convolucionales para realizar la predicción de caciones de acuerdo a su espectrograma de la canción.

## Ejecución

### Dependencias

Se requieren instalar las dependencias del proyecto, para ello se puede usar el `requirements.txt` con pip3:

```bash
pip3 install -r requirements.txt
```

Se recomienda crear un virtualenv de Python para el manejo de las dependencias.

### Configuración

El archivo `trainNeuralNetwork.py` puede ser ejecutado con o sin ayuda del procesamiento de GPU. Se recomienda usar GPU para acelerar las ejecuciones, sin embargo, esto no es limitativo.

Dentro de este archivo, se pueden modificar los siguientes parámetros:

- Número de batches: esto es la cantidad de lotes en los cuales se proesa la data. Es posible ajustar el batch tanto para pruebas como entrenamiento. Esto se cambia en las variables `tfDataset_train` y  `tfDataset_test`
- Densidad de la red: la red neuronal convolucional puede cambiar su densidad de neuronas en los parámetros `units` del método `Dense` en la variable `modelo`.
- Cantidad de épocas: en la variable `epochs` es posible definir la cantidad de épocas a ejecutar.

### Corrida

Para correr el programa una vez instaladas las dependencias, se ejecuta:

```python
python3 trainNeuralNetwork.py
```
