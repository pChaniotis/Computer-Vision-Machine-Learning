import os
import cv2 as cv
import numpy as np
import json
import re

train_folders = ['imagedb/145.motorbikes-101','imagedb/178.school-bus','imagedb/224.touring-bike','imagedb/251.airplanes-101','imagedb/252.car-side-101']

sift = cv.xfeatures2d_SIFT.create()

#FUNCTIONS---------------

def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc

def create_vocabulary(train_folders):
    print('Extracting features...')
    train_descs = np.zeros((0, 128))
    for folder in train_folders:
        files = os.listdir(folder)
        for file in files:
            path = os.path.join(folder, file)
            desc = extract_local_features(path)
            if desc is None: # For Thumbs.db
                continue
            train_descs = np.concatenate((train_descs, desc), axis=0)

    # Create vocabulary
    print('Creating vocabulary...')
    term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
    vocabulary = cv.kmeans(train_descs.astype(np.float32), 50, None, term_crit, 1, 0)[2]
    np.save('vocabulary.npy', vocabulary)
    return vocabulary

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

def create_index(train_folders, vocabulary):
    # Create Index
    print('Creating index...')
    img_paths = []
    bow_descs = np.zeros((0, vocabulary.shape[0]))
    for folder in train_folders:
        files = os.listdir(folder)
        for file in files:
            path = os.path.join(folder, file)
            desc = extract_local_features(path)
            if desc is None:
                continue
            bow_desc = encode_bovw_descriptor(desc, vocabulary)

            img_paths.append(path)
            bow_descs = np.concatenate((bow_descs, bow_desc), axis=0)

    np.save('index.npy', bow_descs)
    with open('index_paths.txt', mode='w+') as file:
        json.dump(img_paths, file)
    return img_paths, bow_descs

def load_index():
    # Load Index
    bow_descs = np.load('index.npy')
    with open('index_paths.txt', mode='r') as file:
        img_paths = json.load(file)
    return img_paths, bow_descs

#-----------------

if os.path.isfile('vocabulary.npy'):
    vocabulary = load_vocabulary()
    img_paths, bow_descs = load_index()
else:
    vocabulary = create_vocabulary(train_folders)
    img_paths, bow_descs = create_index(train_folders, vocabulary)


#Train SVMs

labels = []
for p in img_paths:
    labels.append('145.motorbikes-101' in p)
labels = np.array(labels, np.int32)

print('Training Motorbike SVM...')
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_RBF)
svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))
svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels)
svm.save('svm_motorbike')

labels = []
for p in img_paths:
    labels.append('178.school-bus' in p)
labels = np.array(labels, np.int32)

# Train SVM
print('Training Bus SVM...')
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_RBF)
svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))
svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels)
svm.save('svm_bus')

labels = []
for p in img_paths:
    labels.append('224.touring-bike' in p)
labels = np.array(labels, np.int32)

# Train SVM
print('Training Bike SVM...')
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_RBF)
svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))
svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels)
svm.save('svm_bike')

labels = []
for p in img_paths:
    labels.append('251.airplanes-101' in p)
labels = np.array(labels, np.int32)

# Train SVM
print('Training Airplane SVM...')
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_RBF)
svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))
svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels)
svm.save('svm_airplane')

labels = []
for p in img_paths:
    labels.append('252.car-side-101' in p)
labels = np.array(labels, np.int32)

# Train SVM
print('Training Car SVM...')
svm = cv.ml.SVM_create()
svm.setType(cv.ml.SVM_C_SVC)
svm.setKernel(cv.ml.SVM_RBF)
svm.setTermCriteria((cv.TERM_CRITERIA_COUNT, 100, 1.e-06))
svm.trainAuto(bow_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels)
svm.save('svm_car')
