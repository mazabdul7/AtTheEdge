import hashlib
import socket
import requests
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import numpy as np

def h5toSavedModel():
	print("Converting .h5 model to SavedModel format...")
	model = tf.keras.models.load_model(filename, compile=False)
	model.save("B0", include_optimizer = False)
	print("Model is ready for acceleration")

def accelerateModel(name, bs, height, width, channels, outname):
	model_name = name
	model_batchsize = bs
	model_height = height
	model_width = width
	model_channels = channels

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
	converter.save(outname)
	print("Finished... Model has been accelerated")

#Grab local ip (Use mac address if possible)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
localip = s.getsockname()[0]
s.close()

#Hash ip
m = hashlib.md5()
m.update(localip.encode())
key = m.hexdigest()

#Form request URL
url = "http://192.168.0.11:5000/download/"
username = "jetson"

full_url = url + username + "." + key

print("Contacting: " + full_url)

print("Looking for new update")
req = requests.get(full_url, allow_redirects=True)

if len(req.content) > 10:
	print("Found new update: Downloading...")
	filename = "EfficientNetB0-Small.h5"
	with open(filename, "wb") as file:
		file.write(req.content)
	print("Download successful!")

	#Convert h5 model to SavedModel format to be accelerated
	h5toSavedModel()
	#Accelerate model
	accelerateModel(filename, 1, 288, 384, 3, 'B0_Accelerated')

else:
	print("No new update")
