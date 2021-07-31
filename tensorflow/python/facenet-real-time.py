#Face Recognition with Google's Facenet Model
#Author Sefik Ilkin Serengil (sefiks.com)

#You can find the documentation of this code from the following link: 
#https://sefiks.com/2018/09/03/face-recognition-with-facenet-in-keras/

#Tested for TensorFlow 1.9.0, Keras 2.2.0 and Python 3.5.5

#-----------------------

import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json
from os import listdir
from inception_resnet_v1 import *

#-----------------------
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    print(e)

#face_cascade = cv2.CascadeClassifier('D:\opencv\opencv\build\etc\haarcascades\haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier('C:\\Users\\USER\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(160, 160))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    
    #preprocess_input normalizes input in scale of [-1, +1]. You must apply same normalization in prediction.
    #Ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py (Line 45)
    img = preprocess_input(img)
    return img

#------------------------

#https://github.com/serengil/tensorflow-101/blob/master/model/facenet_model.json
model = InceptionResNetV1()
print("model built")

#https://drive.google.com/file/d/1971Xk5RwedbudGgTIrGAL4F7Aifu7id1/view?usp=sharing
model.load_weights('weights/facenet_weights.h5')
print("weights loaded")

#------------------------

#유클리드
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


#코사인
def findCosineSimilarity(source_representation, test_representation):
 a = np.matmul(np.transpose(source_representation), test_representation)
 b = np.sum(np.multiply(source_representation, source_representation))
 c = np.sum(np.multiply(test_representation, test_representation))
 return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def l2_normalize(x):   #L2정규화
	return x / np.sqrt(np.sum(np.multiply(x, x)))

#------------------------

threshold = 21#0.9#1.5 #21 #tuned threshold for l2 disabled euclidean distance

#------------------------	

#put your employee pictures in this path as name_of_employee.jpg
employee_pictures = "img1/"

employees = dict()

#128차원 출력
for file in listdir(employee_pictures):
    employee, extension = file.split(".")
    print(employee, "," + extension)
    employees[employee] = model.predict(preprocess_image('img1/%s.jpg' % (employee)))[0, :]
    print(employees[employee])
    result = sum(employees[employee])
    print("sum : ", result)
    print("평균 : ", result / len(employees[employee]))
    avg = np.mean(employees[employee])
    print("넘파이 평균 : ", avg)
    print("--------------------------------------------------------------------")
	
print("employee representations retrieved successfully")

#------------------------
cap = cv2.VideoCapture(0) #webcam

#cap.set(3, 1920)
#cap.set(4, 1080)

while(True):
	ret, img = cap.read()
	faces = face_cascade.detectMultiScale(img, 1.3, 5)
	
	for (x,y,w,h) in faces:
		if w > 130: #discard small detected faces
			cv2.rectangle(img, (x,y), (x+w,y+h), (67, 67, 67), 1) #draw rectangle to main image
			
			detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
			detected_face = cv2.resize(detected_face, (160, 160)) #resize to 224x224
			
			img_pixels = image.img_to_array(detected_face)
			img_pixels = np.expand_dims(img_pixels, axis = 0)
			#employee dictionary is using preprocess_image and it normalizes in scale of [-1, +1]
			img_pixels /= 127.5
			img_pixels -= 1
			
			captured_representation = model.predict(img_pixels)[0,:]
			
			distances = []
			
			for i in employees:
				employee_name = i
				source_representation = employees[i]
				
				distance = findEuclideanDistance(captured_representation, source_representation)
				
				#print(employee_name,": ",distance)
				distances.append(distance)
			print("------------------------------------------------------------------------------")
			label_name = 'unknown'
			index = 0
			for i in employees:
				employee_name = i
				if index == np.argmin(distances):
					if distances[index] <= threshold:
						print("detected: ",employee_name+"   distances: ",distances[index])
						
						#label_name = "%s (distance: %s)" % (employee_name, str(round(distance,2)))
						similarity = 100 + (20 - distance)
						if similarity > 99.99: similarity = 99.99
						
						label_name = "%s (%s%s)" % (employee_name, str(round(similarity,2)), '%')
						
						break
					else:
						print("distances: ",distances[index])

				index = index + 1
			
			cv2.putText(img, label_name, (int(x+w+15), int(y-64)), cv2.FONT_HERSHEY_SIMPLEX, 1, (67,67,67), 2)
					
			#connect face and text
			cv2.line(img,(x+w, y-64),(x+w-25, y-64),(67,67,67),1)
			cv2.line(img,(int(x+w/2),y),(x+w-25,y-64),(67,67,67),1)
			
	cv2.imshow('img',img)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
		break
	
#kill open cv things		
cap.release()
cv2.destroyAllWindows()
