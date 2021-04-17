import cv2
import re
from os import walk
import pandas as pd
import numpy as np
import os
from os import path
import random
import shutil
import urllib


import keras
from keras.models import load_model

class MaskDetector:
  
  def __init__(self)
    self.model = load_model('mask.keras')

    if not path.exists('deploy.prototxt'):
      urllib.urlretrieve('https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt', 'deploy.prototxt')
    if not path.exists('res10_300x300_ssd_iter_140000.caffemodel'):
      urllib.urlretrieve('https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel', 'res10_300x300_ssd_iter_140000.caffemodel')

    prototxt_path = 'deploy.prototxt'
    model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
    self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)


  def get_face(self, image):
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    self.net.setInput(blob)
    detections = self.net.forward()

    faces = np.zeros((detections.shape[2], 4))
    for i in range(0, detections.shape[2]):
      # extract the confidence (i.e., probability) associated with the prediction
      confidence = detections[0, 0, i, 2]

      if confidence > 0.2:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        faces[i] = box
    return faces

  def detect(self, img_name, save=True):
    # First one takes some time
    image = cv2.imread(img_name)
    write_img = np.array(image)
    faces = get_face(image)
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

    cv2.imwrite('output.png', write_img)
    return write_img
