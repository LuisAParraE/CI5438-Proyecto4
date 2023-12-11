# CI5438-Proyecto4

## Implementación

Se implementaron redes neuronales convolucionales para realizar la predicción del genero de canciones de acuerdo a su diagrama de ondas.

![diagrama de ondas](/data/0.png)

## Ejecución

### Dependencias

Se requieren instalar las dependencias del proyecto, para ello se puede usar el `requirements.txt` con pip3:

```bash
pip3 install -r requirements.txt
```

Se recomienda crear un virtualenv de Python para el manejo de las dependencias.

### Manejo de los Datos

Existen multiples arhivos aca la descripción de cada uno:

El archivo `formatData.py` se encarga de generar todas las graficas que se le pasaran a la red neuronal para su entrenamiento. Estos diagramas son obtenidos por medio de la librería librosa de python. Para poder sacar los diagramas, es necesario tener un carpeta llamada `songs`, que a su vez, internamente esté dividida en generos, cada canción debe estar dentro de su genero correspndiente. `formatData.py` leera dicha carpeta y colocará los diagramas en la carpeta `data`, ya vienen con diagramas para poder realizar pruebas, además generara un .csv llamado `imageData.csv` donde se encuentra la información de donde esta el diagrama, y que tipo de genero musical es, este csv será usado después para entrenar la red neuronal.

### Entrenamiento de la Red

El archivo `trainNeuralNetwork.py` puede ser ejecutado con o sin ayuda del procesamiento de GPU. Se recomienda usar GPU para acelerar las ejecuciones, sin embargo, esto no es limitativo.

Dentro de este archivo, se pueden modificar los siguientes parámetros:

- Número de batches: esto es la cantidad de lotes en los cuales se proesa la data. Es posible ajustar el batch tanto para pruebas como entrenamiento. Esto se cambia en las variables `tfDataset_train` y  `tfDataset_test`
- Modelo de la Red: Será posible modificar la Red, si se quiere experimentar con diversos cambios en la Red.
- Cantidad de épocas: en la variable `epochs` es posible definir la cantidad de épocas a ejecutar.

Para entrenar la red una vez instaladas las dependencias y seleecionada la data usando `formatData.py`, se ejecuta:

```python
python3 trainNeuralNetwork.py
```

### Clasificar una canción

Para clasificar uan cancion tenemos el archivo `checkSong.py`. En este archivo se realiza lo necesario para que se le provea una sola canción, y con un modelo ya entrenado proceda a clasificarla. Es importante saber que hay que pasarle tanto la dirección donde se encuentra la canción, y la dirección de un modelo de tensorflow ya entrenado.

```python
python3 checkSong.py <path_de_la_cancion> <path_del_modelo_de_la_redNueronal>
```
