'''Main code to run entire model'''
import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import signature_constants, tag_constants
from tensorflow.python.framework import convert_to_constants
import time
import cv2
from adafruit_servokit import ServoKit
import board
import busio

model_name = 'B0_Accelerated'

#Load model
print("Loading Model")
saved_model_loaded = tf.saved_model.load(
    model_name, tags=[tag_constants.SERVING])

#Intantiate graphs from runtime engine
print("Instantiating graph")
graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

#Convert graph variables to constants
print("Freezing graph")
graph_func = convert_to_constants.convert_variables_to_constants_v2(
    graph_func)

#Create pipeline to grab feed from CSI camera
print("Initialising Camera")
def gstreamer_pipeline(
	capture_width=1920,
	capture_height=1080,
	display_width=384,
	display_height=288,
	framerate=30,
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

def capture_input():
	'''Capture input from camera'''
	ret, frame = cap.read()
	return frame

def inference(input):
	'''Run inference on input'''
	x = tf.constant(np.expand_dims(input, axis=0).astype(np.float32))
	pred = graph_func(x) 

	return pred[0].numpy()[0]

def moveServo(port):
	'''Send signal to move servo'''
	sweep = range(0,180)
	sweep2 = range(180, 0, -1)
	for deg in sweep:
		kit.servo[port].angle=deg
	for deg in sweep2:
		kit.servo[port].angle=deg

print("Instantiating camera object")
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
cv2.namedWindow("Cam Detect", cv2.WINDOW_AUTOSIZE)

print("Initializing Servos")
i2c_bus0=(busio.I2C(board.SCL_1, board.SDA_1)) #Set i2c to bus0
print("Initializing ServoKit")
kit = ServoKit(channels=16, i2c=i2c_bus0)
kit.servo[0].set_pulse_width_range(400, 2600)
kit.servo[1].set_pulse_width_range(400, 2600)
kit.servo[2].set_pulse_width_range(400, 2600)

print("Commencing Inference")
labels = ["cardboard", "glass", "metal", "paper", "plastic"]
iflg = 0
previousOut = None
queue = [0]
N = 10 #Queue length

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,30)
fontScale              = 0.5
fontColor              = (255,0,0)
lineType               = 2

while(1):
	print("Inference off, type 1 to start or 0 to exit...")
	choice = int(input())

	if(choice == 1):
		iflg = 1
	elif(choice == 0):
		break

	if(iflg):
		try:
			while(1):
				start =  time.time()
				frame = capture_input()
				prediction = inference(frame)
				output = np.argmax(prediction)

				if len(queue) < N:
					queue.append(output)

				if len(queue) == N:
    					queue.pop(0)
    					queue.append(output)

				output = int(round(sum(queue)/len(queue))) #Averaging output

				if output==0 and previousOut!=output and prediction[output]*100 > 70:
    					moveServo(0)
				elif output==1 and previousOut!=output and prediction[output]*100 > 70:
    					moveServo(1)
				elif output==2 and previousOut!=output and prediction[output]*100 > 70:
    					moveServo(2)
				elif output==3 and previousOut!=output and prediction[output]*100 > 70:
    					moveServo(0)
				if output==4 and previousOut!=output and prediction[output]*100 > 70:
    					moveServo(2)

				if prediction[output]*100 > 60:
					text = "%s Confidence: %.2f FPS: %.2f" % (labels[output], prediction[output]*100, 1/(time.time()-start))
					cv2.putText(frame, text, 
								bottomLeftCornerOfText, 
								font, 
								fontScale,
								fontColor,
								lineType)
					cv2.imshow('Cam Detect', frame)
				else:
					text = "Other FPS: " + str(1/(time.time()-start))
					cv2.putText(frame, text, 
								bottomLeftCornerOfText, 
								font, 
								fontScale,
								fontColor,
								lineType)
					cv2.imshow('Cam Detect', frame) 

				if cv2.waitKey(30) & 0xFF == ord("q"): 
				# Stop the program on the ESC key
					iflg = 0
					break
				
				if prediction[output]*100 > 60:
						previousOut = output
				else:
    					previousOut = None
				time.sleep(0.01)
		except KeyboardInterrupt:
			iflg = 0

cap.release()
cv2.destroyAllWindows()
