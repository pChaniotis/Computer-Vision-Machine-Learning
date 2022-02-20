import cv2 as cv
import numpy as np


#Configurations  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
choice = 'xanthiegg'#yardhouse - xanthiegg
algorithm = 'sift' #sift - surf

#Functions - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def match(desc1, desc2):

    desc1_n = desc1.shape[0]

    matches = []
    for i in range(desc1_n):
    
        fv1 = desc1[i, :]
        diff1 = desc2 - fv
        diff1 = np.abs(diff)
        distances1 = np.sum(diff, axis=1)

           
        i2 = np.argmin(distances1)  # Image 2 to Image 1
        
        mindist2 = distances1[i2]

        fv2 = desc2[i2, :]
        diff2 = desc1 - fv2
        diff2 = np.abs(diff2)
        distances2 = np.sum(diff2, axis=1)

        i1 = np.argmin(distances2)  # Image 1 to Image 2

        if (i1 == i):
            matches.append(cv.DMatch(i, i2, mindist2))

    return matches

def filename(num,choice,output=False):
    if(output==False):
        if (choice == 'yardhouse'):
            return 'images/yard-house-0'+str(num)+'.png'
        else:
            return 'images/xanthi-egg-0'+str(num)+'.png'
    if(output==True):
        if (choice == 'yardhouse'):
            return 'results/yard-house-panorama-'+str(num)+'.png'
        else:
            return 'results/xanthi-egg-panorama-'+str(num)+'.png'
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


image1 = cv.imread(filename(5,choice))


if (algorithm == 'sift'):
    sift = cv.xfeatures2d_SIFT.create(1000)
else:
    sift = cv.xfeatures2d_SURF.create(100)

for a in [4,3,2,1]:

    image2 = cv.imread(filename(a,choice))


    keyPoints1 = sift.detect(image1)
    descriptor1 = sift.compute(image1, keyPoints1)

    keyPoints2 = sift.detect(image2)
    descriptor2 = sift.compute(image2, keyPoints2)

    dimgKeypoint = cv.drawKeypoints(image2, keyPoints2, None)
    cv.namedWindow('keypoints 2',cv.WINDOW_NORMAL)
    cv.imshow('keypoints 2',dimgKeypoint)


    matches = match(descriptor1[1],descriptor2[1])


    dimgMatch = cv.drawMatches(image1, descriptor1[0], image2, descriptor2[0], matches, None)
    cv.namedWindow('matches', cv.WINDOW_NORMAL)
    cv.imshow('matches',dimgMatch)


    point_list1 = []
    point_list2 = []
    for x in matches:
        point_list1.append(keyPoints1[x.queryIdx].pt)
        point_list2.append(keyPoints2[x.trainIdx].pt)
    point_list1 = np.array(point_list1)
    point_list2 = np.array(point_list2)


    M, mask = cv.findHomography(point_list2, point_list1 , cv.RANSAC)
                                            #500 100 for yardhouse / 3000 1000 for xanthiegg
    image_union = cv.warpPerspective(image2, M, (image1.shape[1]+3000, image1.shape[0]+1000))
    image_union[0: image1.shape[0], 0: image1.shape[1]] = image1


    cv.namedWindow('main',cv.WINDOW_NORMAL)
    cv.imshow('main', image_union)
    cv.imwrite(filename(a,choice,output=True),image_union)
    image1 = image_union
    cv.waitKey(0)
    cv.destroyAllWindows()

    del point_list1
    del point_list2




