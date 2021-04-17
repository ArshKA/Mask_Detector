import cv2
import re
from os import walk
import pandas as pd
import numpy as np
import os
from os import path
import random
import shutil
import urllib.request

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import Adam

class MaskDetector:
  
  def __init__(self, mask_model_path = None):
    self.dir_path = os.path.dirname(os.path.realpath(__file__))
    if not mask_model_path:
      mask_model_path = self.dir_path + '/model_weights.keras'

    self.load_model(mask_model_path)

    prototxt_path = self.dir_path + '/deploy.prototxt'
    model_path = self.dir_path + '/res10_300x300_ssd_iter_140000.caffemodel'

    if not path.exists(prototxt_path):
      urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt', prototxt_path)
    if not path.exists(model_path):
      urllib.request.urlretrieve('https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel', model_path)


    self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

  def create_model(self):
    self.model = Sequential()
    vg = VGG16(weights = 'imagenet')

    for layer in vg.layers[:-3]:
      layer.trainable = False
      self.model.add(layer)
    self.model.add(Dense(512, activation='relu'))
    self.model.add(Dense(512, activation='relu'))
    self.model.add(Dense(256))
    self.model.add(Dense(256))
    self.model.add(Dense(256))
    self.model.add(Dense(2, activation = 'softmax'))
    self.model.compile(optimizer=Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

  def load_model(self, mask_model_path):
    self.create_model()
    self.model.load_weights(mask_model_path)

  def get_face(self, image, face_confidence):
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    self.net.setInput(blob)
    detections = self.net.forward()

    faces = np.zeros((detections.shape[2], 4))
    for i in range(0, detections.shape[2]):
      # extract the confidence (i.e., probability) associated with the prediction
      confidence = detections[0, 0, i, 2]

      if confidence > face_confidence:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        faces[i] = box
    return faces

  def detect(self, img_name, save=True, face_confidence=.2):
    # First one takes some time
    image = cv2.imread(img_name)
    write_img = np.array(image)
    faces = self.get_face(image, face_confidence)
    faces = [x for x in faces if np.sum(x) != 0]
    for face in faces:
      face = face.astype(int)
      new_img = image[face[1]:face[3], face[0]:face[2]]
      new_img = cv2.resize(new_img, (224, 224))
    #  new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
      new_img = np.reshape(new_img, (1, 224, 224, 3))
      pred = self.model.predict(new_img)[0]
      if np.argmax(pred) == 0:
        text = 'mask'
        cv2.putText(write_img, text, (face[0], face[1]), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
        cv2.rectangle(write_img, (face[0], face[1]), (face[2], face[3]), (0, 255, 0), 2)

      else:
        text = 'no_mask'
        cv2.putText(write_img, text, (face[0], face[1]), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
        cv2.rectangle(write_img, (face[0], face[1]), (face[2], face[3]), (0, 0, 255), 2)

    if save:
      cv2.imwrite(self.dir_path + '/output.png', write_img)
    return write_img
