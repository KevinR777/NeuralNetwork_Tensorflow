# Librerias necesarias
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import sys
from PIL import Image
import numpy as np


#Inicializamos el modelo en nulo
new_model = None


#Funcion: loadModel
#Se encarga de cargar el modelo o red neuronal especificada
def loadModel(modelName):
	global new_model
	new_model = tf.keras.models.load_model(modelName + '.h5')


def testModel(image_name):
	global new_model
	img = Image.open('data/' + str(image_name)).convert("L")
	img.show()
	img = img.resize((28,28))
	im2arr = np.array(img)
	im2arr = im2arr.reshape(1,28,28)
	# Predicting the Test set results
	predictions = new_model.predict(im2arr)
	print(np.argmax(predictions))


#Funcion: main()
#Carga el modelo y lo prueba
def main():
	modelName = sys.argv[1]
	image_name = sys.argv[2]
	loadModel(modelName)
	testModel(image_name)


main()