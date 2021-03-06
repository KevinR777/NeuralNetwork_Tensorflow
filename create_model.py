# Librerias necesarias
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import sys



#Importamos la base de datos de MNIST de Numeros escritos del 0-9
num_database = tf.keras.datasets.mnist
#Cargamos la informacion en tipo numpy arrays,  train images y labels que es lo que se utilizara para aprender. Son los datos que el modelo utiliza para aprender
#El modelo luego se probara con los test images y labels
(train_images, train_labels), (test_images, test_labels) = num_database.load_data()
#Inicializamos el modelo en nulo
model = None



#Funcion: normalizeData
#Modificamos las imagenes de la base para que los pixeles esten dentro de 0 y 1
def normalizeData():
	global train_images, test_images
	#Normalizar o ajustar la imagen
	#De esta forma los valores de la matriz o pixeles va estar entre 0 y 1
	#Y no de 0 a 255 como lo estaria orgininalmente
	#Esto no es necesario, pero hace que la red neuronal pueda aprender de manera mas sencilla
	train_images = tf.keras.utils.normalize(train_images, axis=1) #Las de entrenar
	test_images = tf.keras.utils.normalize(test_images, axis=1) #Las de test

#Funcion: buildModel
#Se encarga de construir las distintas capas del modelo o de la red neuronal
def buildModel():
	global model
	#Vamos a construir el modelo
	model = tf.keras.models.Sequential() #Creamos un modelo secuencial
	#Transforma el formato de las imágenes de una matriz 2d (de 28 por 28 píxeles) a una matriz 1d de 28 * 28 = 784 píxeles. 
	#Esta capa no tiene parámetros para aprender; sólo reformatea los datos.
	model.add(tf.keras.layers.Flatten(input_shape = ((28, 28)))) #Capa de input, el input lo queremos ver de manera plana
	#Vamos a crear para este modelo, dos capas intermedias
	#model.add(tf.keras.layers.Dense(512,activation = tf.nn.relu)) #Capa hidden 1, 128 es el numero de neuronas o units, y se le pasa la funcion activation o lo que va a realizar dicha neurona si es "disparada" o "activada". Relu es la default o rectify linear unit
	model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
	model.add(tf.keras.layers.Dense(128,activation = tf.nn.relu))
	#Vamos a crear para este modelo, dos capas intermedias
	model.add(tf.keras.layers.Dense(10,activation = tf.nn.softmax)) #Capa output, utilizamos softmax por que queremos visualizar el output como una distribucion de porcentajes
	#Fin del modelo


def as_keras_metric(method):
    import functools
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.python.keras import backend as K
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

#Funcion: trainModel
#Se encarga de compilar o entrenar el modelo
def trainModel():
	global model, train_images, train_labels
	#Argumentos o metodos para entrenar nuestro modelo
	#Se le pasa el optimizador (adam) es que el se usa por defecto y la metrica de perdida o el porcentaje de error. 
	#Las redes neuronales siempre buscan disminuir la perdida de precision.
	#Y las metricas que queremos visualizar o seguir a lo largo del proceso, en este caso la precision
	precision = as_keras_metric(tf.metrics.precision)
	recall = as_keras_metric(tf.metrics.recall)
	model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = [precision,'accuracy',recall])

	#Entrenamos el modelo, epochs viene siendo la cantidad de ciclos que se va entrenar la red neuronal
	model.fit(train_images, train_labels, epochs = 3)

	#Calculamos los valores de perdida y de precision
	val_loss, val_prec, val_acc, val_recall = model.evaluate(train_images, train_labels)
	print("-------------------------------------------------------------------------------------------------------------------")
	print("El valor o porcentaje de perdida durante entrenamiento es del: " + str(val_loss))
	print("El valor o porcentaje de precision durante entrenamiento es del: " + str(val_prec))
	print("El valor o porcentaje de exactitud durante entrenamiento es del: " + str(val_acc))
	print("El valor o porcentaje de exhaustividad durante entrenamiento es del: " + str(val_recall))
	print("-------------------------------------------------------------------------------------------------------------------")
	val_loss_test, val_prec_test, val_acc_test, val_recall_test = model.evaluate(test_images, test_labels)
	print("El valor o porcentaje de perdida con el conjunto de prueba es del: " + str(val_loss_test))
	print("El valor o porcentaje de precision con el conjunto de prueba es del: " + str(val_prec_test))
	print("El valor o porcentaje de exactitud con el conjunto de prueba es del: " + str(val_acc_test))
	print("El valor o porcentaje de exhaustividad con el conjunto de prueba es del: " + str(val_recall_test))
	print("-------------------------------------------------------------------------------------------------------------------")

#Funcion: saveModel
#Guardamos el modelo
def saveModel(modelName):
	global model
	model.save(modelName + '.h5')


#Funcion: main
#Normaliza la data, construye el modelo, lo entrena y finalmente lo guarda
def main():
	modelName = sys.argv[1]
	normalizeData()
	buildModel()
	trainModel()
	saveModel(modelName)


main()

#Nos muestra como se ve esa imagen en forma de matriz, donde cada campo representa los pixeles
#A esto se le llama tensor
#print(train_images[0])