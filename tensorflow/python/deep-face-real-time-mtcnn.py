# author Sefik Ilkin Serengil
# you can find the documentation of this code from the following link: https://sefiks.com/2018/08/06/deep-face-recognition-with-keras/

import numpy as np
import cv2
import tensorflow as tf
from mtcnn import MTCNN

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, \
    Activation
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

from os import listdir

# -----------------------
color = (0, 255, 0)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

detector = MTCNN()

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # preprocess_input normalizes input in scale of [-1, +1]. You must apply same normalization in prediction.
    # Ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py (Line 45)
    img = preprocess_input(img)
    return img


def loadVggFaceModel():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    # you can download pretrained weights from https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
    from keras.models import model_from_json
    model.load_weights('weights/vgg_face_weights.h5')

    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    return vgg_face_descriptor


model = loadVggFaceModel()

# ------------------------

# put your employee pictures in this path as name_of_employee.jpg
employee_pictures = "img1/"

employees = dict()

# 2622차원 출력
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


def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


# ------------------------

cap = cv2.VideoCapture(0)  # webcam
cap.set(3,1024) #3은 가로
cap.set(4,768)  #4는 세로
j = 1
o = 0
X = 0
while (True):
    ret, img = cap.read()
    faces = detector.detect_faces(img)

    if not ret:
        print("not ret error")
        continue

    for face in faces:
        bounding_box = face['box']
        keypoints = face['keypoints']

        cv2.rectangle(img,
                      (bounding_box[0], bounding_box[1]),
                      (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                      color,3)

        #cv2.circle(img, (keypoints['left_eye']), 2, (0, 155, 255), 2)
        #cv2.circle(img, (keypoints['right_eye']), 2, (0, 155, 255), 2)
        #cv2.circle(img, (keypoints['nose']), 2, (0, 155, 255), 2)
        #cv2.circle(img, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
        #cv2.circle(img, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

        detected_face = img[int(bounding_box[1]):int(bounding_box[1] + bounding_box[3]),
                        int(bounding_box[0]):int(bounding_box[0] + bounding_box[2])]  # crop detected face
        try:
            detected_face = cv2.resize(detected_face, (224, 224))  # resize to 224x224
        except Exception as e:
            print(str(e))
            continue

        img_pixels = image.img_to_array(detected_face)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        # img_pixels /= 255
        # employee dictionary is using preprocess_image and it normalizes in scale of [-1, +1]
        img_pixels /= 127.5
        img_pixels -= 1

        captured_representation = model.predict(img_pixels)[0,:]

        found = 0
        for i in employees:
            employee_name = i
            representation = employees[i]

            similarity = findCosineSimilarity(representation, captured_representation)
            if (similarity < 0.21):  # 0.30,0.21
                cv2.putText(img, employee_name, (int(bounding_box[0] + bounding_box[2] + 15), int(bounding_box[1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),
                                3)
                print("%d 번째, 같은 사람!!----->" % j, employee_name, similarity)
                o = o + 1
                found = 1
                break
            # else:
            # print("다른 사람!!----->", employee_name, similarity)

            # connect face and text
        cv2.line(img, (int((bounding_box[0] + bounding_box[0] + bounding_box[2]) / 2), bounding_box[1] + 15), (bounding_box[0] + bounding_box[2], bounding_box[1] - 20), (0,0,0), 3)
        cv2.line(img, (bounding_box[0] + bounding_box[2], bounding_box[1] - 20), (bounding_box[0] + bounding_box[2] + 10, bounding_box[1] - 20), (0,0,0), 3)

        if (found == 0):  # if found image is not in employee database
            cv2.putText(img, 'unknown', (int(bounding_box[0] + bounding_box[2] + 15), int(bounding_box[1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
            print("%d 번째 unkown, similarity: %d" % (j,similarity))
            X = X + 1

        j = j + 1

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break

# kill open cv things
pre = o / j * 100
print("O:%d,  X: %d  정확도 : %d" % (o, X, pre))
cap.release()
cv2.destroyAllWindows()
