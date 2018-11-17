# Importing the Keras libraries and packages
from keras.models import load_model
from PIL import Image
import numpy as np
import sys

def testModel(modelName , directory):

	model = load_model(modelName)
	for index in range(10):
	    img = Image.open(directory+'/' + str(index) + '.png').convert("L")
	    img = img.resize((28,28))
	    im2arr = np.array(img)
	    im2arr = im2arr.reshape(1,28,28,1)
	    # Predicting the Test set results
	    y_pred = model.predict(im2arr)
	    print(y_pred)

def main():
	modelName = sys.argv[1]
	directory = sys.argv[2]
	testModel(modelName, directory)

main()