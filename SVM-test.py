import os
import cv2 as cv
import numpy as np
import re
import json

sift = cv.xfeatures2d_SIFT.create()

#FUNCTIONS-----------

def load_vocabulary():
    vocabulary = np.load('vocabulary.npy')
    return vocabulary

def accuracy(correct, num):
    if num == 0:
        print("run test first")
    else:
        h = (correct/ num) * 100
        print("Correct percentage is " + str(h) + "% \n")

#---------------------

vocabulary = load_vocabulary()






# Classification
descriptor_extractor = cv.BOWImgDescriptorExtractor(sift, cv.BFMatcher(cv.NORM_L2SQR))
descriptor_extractor.setVocabulary(vocabulary)



# Load SVM
svm_motorbike = cv.ml.SVM_create()
svm_motorbike = svm_motorbike.load('svm_motorbike')

svm_bus = cv.ml.SVM_create()
svm_bus = svm_bus.load('svm_bus')

svm_bike = cv.ml.SVM_create()
svm_bike = svm_bike.load('svm_bike')

svm_airplane = cv.ml.SVM_create()
svm_airplane = svm_airplane.load('svm_airplane')

svm_car = cv.ml.SVM_create()
svm_car = svm_car.load('svm_car')
#--------

test_folders = ['imagedb_test/145.motorbikes-101','imagedb_test/178.school-bus','imagedb_test/224.touring-bike','imagedb_test/251.airplanes-101','imagedb_test/252.car-side-101']

correct = 0
num = 0

for folder in test_folders:
    files = os.listdir(folder)
    for file in files:
        path = os.path.join(folder, file)
        img = cv.imread(path)
        kp = sift.detect(img)
        bow_desc = descriptor_extractor.compute(img, kp)
        num += 1


        #Generate Responses for all SVMs
        response_motorbike = svm_motorbike.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        response_bus = svm_bus.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        response_bike = svm_bike.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        response_airplane = svm_airplane.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        response_car = svm_car.predict(bow_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)


        #Smallest Distance
        resp_array = np.array([response_motorbike[1], response_bus[1], response_bike[1], response_airplane[1], response_car[1]])
        minIndex = np.argmin(resp_array)

        if minIndex== 0:
            if re.search('.*motorbike.*', folder):
                correct += 1
        if minIndex == 1:
            if re.search('.*bus.*', folder):
                correct += 1
        if minIndex == 2:
            if re.search('.*touring.*', folder):
                correct += 1
        if minIndex == 3:
            if re.search('.*airplane.*', folder):
                correct += 1
        if minIndex == 4:
            if re.search('.*car.*', folder):
                correct += 1

accuracy(correct, num)