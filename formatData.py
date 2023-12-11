import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

#Carpeta de genero de canciones
songsPath = "songs"
#Carpeta donde se colocaran los analisis de ondas
dataPath = "data"

#Extraemos la lista de generos
listOfGenres = [f for f in os.listdir(songsPath) if os.path.isdir(songsPath+"\\"+f)]
print(f"path,genre",file=open(f"imageData.csv", 'x'))
i = 0
for genre in listOfGenres:
    for song in os.listdir(songsPath+"\\"+genre):
        if i >191:
            y, sr = librosa.load(songsPath+"\\"+genre+"\\"+song)
            y_harm, y_perc = librosa.effects.hpss(y)
            plt.figure(figsize=(15,6))
            librosa.display.waveshow(y_harm, sr=sr, alpha=0.5)
            librosa.display.waveshow(y_perc, sr=sr, color='r', alpha=0.5)
            plt.grid()

            savePath = dataPath + "/"+ str(i) + ".png"
            plt.savefig(savePath)
            plt.close()
            print(f"{savePath},{(listOfGenres.index(genre))}",file=open(f"imageData.csv", 'a'))
        i = i+1


