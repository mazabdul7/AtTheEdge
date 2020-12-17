#Test accuracy of TRT models
#import libraries
import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import signature_constants, tag_constants
from tensorflow.python.framework import convert_to_constants

image_height = 288
image_width = 384

#load model
print("Loading Model")
saved_model_loaded = tf.saved_model.load(
    'B02TRT_FP16', tags=[tag_constants.SERVING])

#intantiate graphs from runtime engine
print("Instantiating graph")
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

#convert graph variables to constants
print("Freezing graph")
graph_func = convert_to_constants.convert_variables_to_constants_v2(
    graph_func)

#load test set
print("Commencing Test")
labels = ["cardboard", "glass", "metal", "paper", "plastic"]
print("Loading images")
img = tf.keras.preprocessing.image.DirectoryIterator('path to test set', tf.keras.preprocessing.image.ImageDataGenerator(), target_size=(image_height, image_width), batch_size=1)

#run accuracy test
acc = 0
for i in range(len(img)):
	x = tf.constant(img[i][0])
	predictions = graph_func(x)
	for j in img[i][1]:
		if np.argmax(predictions[0].numpy()[0]) == j.tolist().index(1):
			acc += 1
print("The accuracy against the test set was found to be: " + str((acc/len(img))*100) + "%")
