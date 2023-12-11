from pyexpat import model
from traceback import print_tb
import librosa
from PIL import Image
import librosa.display
import os
import sys
import tensorflow as tf
from keras import  Sequential, datasets, losses
from keras.layers import  Conv2D, Dense, Flatten,MaxPooling2D, Dropout
import numpy as np
import matplotlib.pyplot as plt

listOfGenres = ["country","edm","jazz","pop","regueton","rock","salsa"]

def main(songPath,modelPath):
    y, sr = librosa.load(songPath)
    y_harm, y_perc = librosa.effects.hpss(y)
    plt.figure(figsize=(15,6))
    librosa.display.waveshow(y_harm, sr=sr, alpha=0.5)
    librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5)
    plt.grid()

    savePath = "temp/check.png"
    plt.savefig(savePath)
    plt.close()
    
    img = Image.open(savePath).convert('RGB').resize((1500, 600))
    img = np.array(img)

    savedModel = tf.keras.models.load_model(modelPath)

    value = savedModel.predict(img[None,:,:],verbose=0)
    maxIndex = 0
    maxValue = -1
    i = 0
    for element in value[0]:
        if element > maxValue:
            maxValue = element
            maxIndex = i
        i = i+1

    print(f"La cancion es de Genero: {listOfGenres[maxIndex]}")
    os.remove(savePath)

if __name__ == "__main__":
    if len(sys.argv)== 3:
        main(sys.argv[1],sys.argv[2])
    else:
        print("Usage: python3 checkSong <songPath> <NNmodelPath>")