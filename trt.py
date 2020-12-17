import tensorflow as tf
import numpy as np
##from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import signature_constants, tag_constants
from tensorflow.python.framework import convert_to_constants
#from tensorflow.keras.mixed_precision import experimental as mixed_precision
import time

#with tf.device('/GPU:0'):
	#tf.config.optimizer.set_experimental_options({'auto_mixed_precision':True})
	#print('setting policy')
	#policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
	#mixed_precision.set_policy(policy)

print("loading")
saved_model_loaded = tf.saved_model.load(
    'B02TRT_FP16', tags=[tag_constants.SERVING])
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
graph_func = convert_to_constants.convert_variables_to_constants_v2(graph_func)


print("INFERENCE")
while(1):
    img = tf.keras.preprocessing.image.load_img('/home/maz/Desktop/tests/test/cardboard/cardboard5.jpg', target_size=(288,384))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.constant(x)
    start =  time.time()
    output = graph_func(x)
    print(output)
    end = time.time()
    speed = end - start
    print("Took " + str(speed) + " FPS: " + str(1/speed))	
