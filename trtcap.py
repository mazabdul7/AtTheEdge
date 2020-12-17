#Main code to run real-time inference
#import libraries
import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import signature_constants, tag_constants
from tensorflow.python.framework import convert_to_constants
import time
import cv2

#load model
print("Loading Model")
saved_model_loaded = tf.saved_model.load(
    'B02TRT_FP16N', tags=[tag_constants.SERVING])

#intantiate graphs from runtime engine
print("Instantiating graph")
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

#convert graph variables to constants
print("Freezing graph")
graph_func = convert_to_constants.convert_variables_to_constants_v2(
    graph_func)

#create pipeline to grab feed from CSI camera
print("Initialising Camera")
def gstreamer_pipeline(
    capture_width=384,
    capture_height=288,
    display_width=384,
    display_height=288,
    framerate=25,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

#instantiate camera object
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

#main inference code
print("Commencing Inference")
labels = ["cardboard", "glass", "metal", "paper", "plastic"]
while(1):
	start =  time.time() #start timer
	ret, frame = cap.read() #capture input from CSI camera stream
	cv2.imshow('frame', frame) #display image to user
	x = tf.constant(np.expand_dims(frame, axis=0).astype(np.float32)) #convert input to tensor and f32
	
	pred = graph_func(x) #run inference on input
	output = np.argmax(pred[0].numpy()[0]) #find top prediction
	end = time.time() #end timer
    #print 'Predicted Label, Confidence, frame rate'
	print("%s   Confidence:   %.2f  FPS:  %.2f" % (labels[output], pred[0].numpy()[0][output]*100, 1/(end-start)))

cap.release() #release capture pipeline

