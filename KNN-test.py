import os
import cv2 as cv
import numpy as np
import json
import re

sift = cv.xfeatures2d_SIFT.create()

#FUNCTIONS---------------------------------

def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc

def encode_bovw_descriptor(desc, vocabulary):
    bow_desc = np.zeros((1, vocabulary.shape[0]))
    for d in range(desc.shape[0]):
        distances = np.sum((desc[d, :] - vocabulary) ** 2, axis=1)
        mini = np.argmin(distances)
        bow_desc[0, mini] += 1
    if np.sum(bow_desc) > 0:
        bow_desc = bow_desc / np.sum(bow_desc)
    return bow_desc

def load_vocabulary():
    vocabulary = np.load('vocabulary.npy')
    return vocabulary

def load_index():
    # Load Index
    bow_descs = np.load('index.npy')
    with open('index_paths.txt', mode='r') as file:
        img_paths = json.load(file)
    return img_paths, bow_descs

def kNN(query_img_path, N):
    global correct
    frequency = np.zeros(5)
    desc = extract_local_features(query_img_path)
    bow_desc = encode_bovw_descriptor(desc, vocabulary)
    distances = np.sum((bow_desc - bow_descs) ** 2, axis=1)
    ids = np.argsort(distances)
    ids = ids[0:N]
    for id in ids.tolist():
        if re.search('.*motorbike.*', img_paths[id]):
            frequency[0] += 1
        elif re.search('.*bus.*', img_paths[id]):
            frequency[1] += 1
        elif re.search('.*touring.*', img_paths[id]):
            frequency[2] += 1
        elif re.search('.*airplanes.*', img_paths[id]):
            frequency[3] += 1
        elif re.search('.*car.*', img_paths[id]):
            frequency[4] += 1

    index_max = np.argmax(frequency)
    if index_max == 0:
        if re.search('.*motorbike.*', folder):
            correct += 1
    if index_max == 1:
        if re.search('.*bus.*', folder):
            correct += 1
    if index_max == 2:
        if re.search('.*touring.*', folder):
            correct += 1
    if index_max == 3:
        if re.search('.*airplane.*', folder):
            correct+= 1
    if index_max == 4:
        if re.search('.*car.*', folder):
            correct += 1


    pass

def accuracy(correct, num):
    if num == 0:
        print("run test first!")
    else:
        h = (correct / num) * 100
        print("Correct percentage is " + str(h) + "% \n")

#---------------------------

vocabulary = load_vocabulary()
img_paths, bow_descs = load_index()

test_folders = ['imagedb_test/145.motorbikes-101','imagedb_test/178.school-bus','imagedb_test/224.touring-bike','imagedb_test/251.airplanes-101','imagedb_test/252.car-side-101']

for i in range(1, 60, 5):
    correct = 0
    num = 0
    for folder in test_folders:
        files = os.listdir(folder)
        for file in files:
            path = os.path.join(folder, file)
            kNN(path,i)
            num += 1
    print("For i = " + str(i)+ " nearest neighboors")
    accuracy(correct,num)