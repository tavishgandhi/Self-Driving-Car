#  This file is to create a connection between our model and simulator
import socketio
import eventlet
from flask import Flask 
from keras.models import load_model
import base64
import matplotlib.image as mpimg
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

sio = socketio.Server()

app = Flask(__name__)
speed_limit = 15

def img_preprocessing(img):
  img = img[60:135 , :, :] #Our image has got trees and car hood which is uneccessary,also an image is array of height width and depth,so we can crop image
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  img = cv2.GaussianBlur(img, (3, 3) , 0)
  img = cv2.resize(img, (200, 66))
  img = img/255
  return img

@sio.on('telemetry')
def telemetry(sid, data):
	speed = float(data['speed'])
	image = Image.open(BytesIO(base64.b64decode(data['image'])))
	image = np.asarray(image)
	image = img_preprocessing(image)
	image = np.array([image])
	steering_angle = float(model.predict(image))
	throttle = 1.0 - speed/speed_limit
	print('{} {} {}'.format(steering_angle, throttle, speed))
	send_control(steering_angle, throttle)


@sio.on('connect') 
def connect(sid, tensorflow):
	print('connected')
	send_control(0, 0)

def send_control(steering_angle, throttle):
	sio.emit('steer', data = {
		'steering_angle' : steering_angle.__str__(), 
		'throttle': throttle.__str__()

		})

if __name__ == '__main__':
	model = load_model('model.h5')
	app = socketio.Middleware(sio, app)
	eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
