# Librerias necesarias
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import sys

#Importamos la base de datos de MNIST de Numeros escritos del 0-9
num_database = tf.keras.datasets.mnist

#Cargamos la informacion en train images y labels que es lo que se utilizara para aprender. Son los datos que el modelo utiliza para aprender
#El modelo luego se probara con los test images y labels
(train_images, train_labels), (test_images, test_labels) = num_database.load_data()

#Inicializamos el modelo en nulo
new_model = None


#Funcion: loadModel
#Se encarga de cargar el modelo o red neuronal especificada
def loadModel(modelName):
	global new_model
	new_model = tf.keras.models.load_model(modelName + '.h5')


#Funcion: testModel()
#Se encarga de probar el modelo, lee las imagenes de test, las muestra y hace la prediccion
def testModel():
	global new_model, test_images
	i = 0
	#For para recorrer imagenes del test set
	for i in range(8000,8010):
		#Hacemos la prediccion en base al test set de imagenes
		predictions = new_model.predict([test_images])
		#Imprimimos la prediccion
		print(np.argmax(predictions[i]))
		#Mostramos la imagen para comprobar
		plt.imshow(test_images[i], cmap=plt.cm.binary)
		plt.show()


#Funcion: main()
#Carga el modelo y lo prueba
def main():
	modelName = sys.argv[1]
	loadModel(modelName)
	testModel()


main()