#author Sefik Ilkin Serengil
#you can find the documentation of this code from the following link: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

import numpy as np
import cv2
import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

from os import listdir
#-----------------------

color = (67,67,67)
#color = (0,0,0)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

#face_cascade = cv2.CascadeClassifier('C:/Users/IS96273/AppData\Local/Continuum/anaconda3/pkgs/opencv-3.3.1-py35h20b85fd_1/Library/etc/haarcascades/haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('C:\\Users\\USER\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    #preprocess_input normalizes input in scale of [-1, +1]. You must apply same normalization in prediction.
    #Ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py (Line 45)
    img = preprocess_input(img)
    return img

def loadVggFaceModel():
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Convolution2D(4096, (7, 7), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(4096, (1, 1), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(2622, (1, 1)))
	model.add(Flatten())
	model.add(Activation('softmax'))
	
	#you can download pretrained weights from https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
	from keras.models import model_from_json
	model.load_weights('weights/vgg_face_weights.h5')
	
	vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
	
	return vgg_face_descriptor

model = loadVggFaceModel()

#------------------------

#put your employee pictures in this path as name_of_employee.jpg
employee_pictures = "img1/"

employees = dict()

#2622?????? ??????
for file in listdir(employee_pictures):
	employee, extension = file.split(".")
	print(employee,","+extension)
	employees[employee] = model.predict(preprocess_image('img1/%s.jpg' % (employee)))[0,:]
	print(employees[employee])
	result=sum(employees[employee])
	print("sum : ", result)
	print("?????? : ", result/len(employees[employee]))
	avg=np.mean(employees[employee])
	print("????????? ?????? : ", avg)
	print("--------------------------------------------------------------------")

print("employee representations retrieved successfully")

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

#------------------------

cap = cv2.VideoCapture(0) #webcam
#cap = cv2.VideoCapture('C:/Users/IS96273/Desktop/zuckerberg.mp4') #video


j=1
o=0
X=0
while(True):
	ret, img = cap.read()
	#img = cv2.resize(img, (640, 360))
	faces = face_cascade.detectMultiScale(img, 1.3, 5)

	for (x,y,w,h) in faces:
		if w > 130:
			cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3) #draw rectangle to main image
			
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.resize(detected_face, (224, 224)) #resize to 224x224
			
			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			#img_pixels /= 255
			#employee dictionary is using preprocess_image and it normalizes in scale of [-1, +1]
			img_pixels /= 127.5
			img_pixels -= 1
			
			captured_representation = model.predict(img_pixels)[0,:]
			
			found = 0
			for i in employees:
				employee_name = i
				representation = employees[i]
				
				similarity = findCosineSimilarity(representation, captured_representation)
				if(similarity < 0.21): #0.30
					cv2.putText(img, employee_name, (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
					print("%d ??????, ?????? ??????!!----->"%j, employee_name, similarity)
					o=o+1
					found = 1
					break
				#else:
					#print("?????? ??????!!----->", employee_name, similarity)

			#connect face and text
			cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),color,1)
			cv2.line(img,(x+w,y-20),(x+w+10,y-20),color,1)
		
			if(found == 0): #if found image is not in employee database
				cv2.putText(img, 'unknown', (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
				print("%d ?????? unkown"%j)
				X=X+1

			j=j+1
	"""
	for (x, y, w, h) in faces:

		# cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #draw rectangle to main image

		detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
		detected_face = cv2.resize(detected_face, (224, 224))  # resize to 224x224

		img_pixels = image.img_to_array(detected_face)
		img_pixels = np.expand_dims(img_pixels, axis=0)
			# img_pixels /= 255
			# employee dictionary is using preprocess_image and it normalizes in scale of [-1, +1]
		img_pixels /= 127.5
		img_pixels -= 1

		captured_representation = model.predict(img_pixels)[0, :]

		found = 0
		for i in employees:
			employee_name = i
			representation = employees[i]

			similarity = findCosineSimilarity(representation, captured_representation)
			if (similarity < 0.21):  # 0.30
				cv2.putText(img, employee_name, (int(x + w + 15), int(y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
								2)
				print("%d ??????, ?????? ??????!!----->" % j, employee_name, similarity)
				o = o + 1
				found = 1
				break
			# else:
			# print("?????? ??????!!----->", employee_name, similarity)

			# connect face and text
		cv2.line(img, (int((x + x + w) / 2), y + 15), (x + w, y - 20), color, 1)
		cv2.line(img, (x + w, y - 20), (x + w + 10, y - 20), color, 1)

		if (found == 0):  # if found image is not in employee database
			cv2.putText(img, 'unknown', (int(x + w + 15), int(y - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
			print("%d ?????? unkown" % j)
			X = X + 1

		j = j + 1
	"""
	cv2.imshow('img',img)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break
	
#kill open cv things
pre=o/j*100
print("O:%d,  X: %d  ????????? : %d"%(o,X,pre))
cap.release()
cv2.destroyAllWindows()
