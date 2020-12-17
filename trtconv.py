#converts TF model to TRT runtime engine
#import required libraries
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np

#initiate parameters
batch_size = 1
img_height = 288
img_width = 384

#set conversion parameters such as quantisation, max memory allowance, batch size
#check TensorRT documentation for more parameters
print('setting params')
conversion_params =trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(max_workspace_size_bytes=1<<25, precision_mode='FP16', max_batch_size = 1)

#load TF model and conversion parameters
print('converting')
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='B0Small',
    conversion_params=conversion_params)

#generate graphs
converter.convert()

#by feeding the builder sample data representative of the input data to be fed
#at inference we can build the runtime engine ahead of time to save computation and memory
print('building')
#define a generator that generates random data equal to the shape of the test data
def my_input_fn():
	numr = 10
	for _ in range(numr):
		# Input for a single inference call, for a network that has one input tensors:
		inp1 = np.random.random([batch_size, img_height, img_width, 3]).astype(np.float32) * 255
		yield [inp1]

#build runtime engines from subgraphs and save
converter.build(input_fn=my_input_fn)
converter.save('TRT MODEL')
