'''Manual model acceleration script'''
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np

model_name = 'B0'
model_batchsize = 1
model_height = 384
model_width = 512
model_channels = 3

input_tensor = [model_batchsize, model_height, model_width, model_channels]

print('*Model Acceleration Script*')
print('Setting params')
#Set desired configuration
conversion_params =trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(max_workspace_size_bytes=1<<25, precision_mode='FP16', max_batch_size = 1)

print('Converting')
#Convert model to graph format
converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=model_name,
    conversion_params=conversion_params)

converter.convert()

print('Building')
#Pass samples to pre-build engine
def my_input_fn():
	numr = 10
	for _ in range(numr):
		inp1 = np.random.random(input_tensor).astype(np.float32) * 255
		yield [inp1]

converter.build(input_fn=my_input_fn)
converter.save('B0TRT')
