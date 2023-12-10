from cgi import test
import pandas
import os
import tensorflow as tf
from tensorflow import keras
from keras import  Sequential, datasets, losses
from keras.layers import  Conv2D, Dense, Flatten,MaxPooling2D
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

#funciones
def readImage(image_path,genre):
    raw = tf.io.read_file(image_path)
    image = tf.image.decode_png(raw,channels=3)
    image = tf.image.resize(image,(600,1500))
    return image,genre

allDataInfo = pandas.read_csv("imageData.csv")

filePath = allDataInfo["path"].values
labels = allDataInfo["genre"].values

DataTraining,DataTest,AnswerTraining,AnswerTest = train_test_split(
            filePath, labels,test_size= 0.2,shuffle=True
        )

tfDataset_train = tf.data.Dataset.from_tensor_slices((DataTraining,AnswerTraining))
tfDataset_test = tf.data.Dataset.from_tensor_slices((DataTest,AnswerTest))

tfDataset_train = tfDataset_train.map(readImage).batch(2)
tfDataset_test = tfDataset_test.map(readImage).batch(2)

modelo =  Sequential([
    Conv2D(32,(3,3), input_shape=(600,1500,3),activation="relu"),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(units=20, activation="relu"),
    Dense(units=20, activation="relu"),
    Dense(units=7, activation="softmax")
])


modelo.compile(optimizer='adam',loss =losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["accuracy"] )
saveFilepath = 'nnModels/'
folder_name = 'graphs'
epochs = 2
trainLoss = []
trainAcc = []
testLoss = []
testAcc = []
for i in range(0,epochs):
    print(f"---Epoca {i+1}/{epochs}---")
    historialTrain = modelo.fit(tfDataset_train,epochs=1)
    lossTest,accTest = modelo.evaluate(tfDataset_test)
    modelo.save(saveFilepath+str(i)+"/modelo.keras")

    aux1 = pandas.DataFrame(historialTrain.history)
    aux2 = aux1['loss'].values
    aux3 = aux1['accuracy'].values

    trainLoss.append(aux2[0])
    trainAcc.append(aux3[0])
    testLoss.append(lossTest)
    testAcc.append(accTest)


plt.plot(trainLoss)
plt.title(f"Train loss with epochs:{epochs}")
plt.grid()
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig(f'{folder_name}/Train_Loss_epochs_{epochs}.png')
plt.close()

plt.plot(trainAcc)
plt.title(f"Train Accuracy with epochs:{epochs}")
plt.grid()
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.savefig(f'{folder_name}/Train_Accuracy_epochs_{epochs}.png')
plt.close()


plt.plot(testLoss)
plt.title(f"Test loss with epochs:{epochs}")
plt.grid()
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.savefig(f'{folder_name}/Test_Loss_epochs_{epochs}.png')
plt.close()

plt.plot(testAcc)
plt.title(f"Test Accuracy with epochs:{epochs}")
plt.grid()
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.savefig(f'{folder_name}/Test_Accuracy_epochs_{epochs}.png')
plt.close()



