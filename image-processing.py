import cv2
import numpy as np


types=['original','noise']

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# noise_type='noise' #change betweeen 0 for 'original' and 1 for 'noise'
noise_type =types[1]
# if manual = False  the denoise process will be done with OpenCV tools.
manual = False;
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


for a in range(1,6):
    #import photo
    filename = 'images/'+str(a)+'_'+noise_type+'.png'
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    # remove salt and peper noise
    if(noise_type=='noise'):

        if(not manual):

            #CLOSING WITH OPEN CV
            den_ker = 3
            strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (den_ker,den_ker))
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, strel)
        else:
            # CLOSING MANUAL ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            img_temp = img
            img2 = img
            # dilation kernel size
            den_ker = 3
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (den_ker, den_ker))
            H, W = img.shape
            pad = int((kernel.shape[1] - 1) / 2)
            img_temp = cv2.copyMakeBorder(img_temp, pad, pad, pad, pad, cv2.BORDER_DEFAULT)

            # threshold
            img_temp = cv2.threshold(img_temp, 210, 255, cv2.THRESH_BINARY)[1]

            # DILATION
            for x in np.arange(pad, H + pad):
                for y in np.arange(pad, W + pad):
                    matrix = img_temp[x - pad:x + pad + 1, y - pad:y + pad + 1]
                    sum = np.sum(matrix * kernel)
                    if sum == 0:
                        pixel = 0
                    else:
                        pixel = 255
                    img2[x - pad, y - pad] = pixel

            # padding
            # erosion kernel size
            den_ker = 5
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (den_ker, den_ker))
            kernel_sum = np.sum(kernel)
            img3 = img2
            pad = int((kernel.shape[1] - 1) / 2)
            img2 = cv2.copyMakeBorder(img2, pad, pad, pad, pad, cv2.BORDER_DEFAULT)

            # EROSION
            for x in np.arange(pad, H + pad):
                for y in np.arange(pad, W + pad):
                    matrix = img2[x - pad:x + pad + 1, y - pad:y + pad + 1]
                    sum = np.sum(matrix * kernel) / kernel_sum
                    if sum == 255:
                        pixel = 255
                    else:
                        pixel = 0
                    img[x - pad, y - pad] = pixel

         # CLOSING MANUAL END ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




    #return to rgb to have colored bounding rectangle
    img_color = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)



    #make one cluster
    par_ker = 29+6
    strel = cv2.getStructuringElement(cv2.MORPH_RECT,(par_ker,par_ker))
    img_blob = cv2.morphologyEx(img,cv2.MORPH_ERODE,strel)

    #make binary % invert
    thresh = 210
    binary = cv2.threshold(img_blob, thresh, 255, cv2.THRESH_BINARY_INV)[1] #this function outputs 2 values, [0] is the threshold set and [1] is the binary image.
    cv2.imwrite('binaries/binary' + str(a) + '.jpg', binary)



    #find contours
    contours = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    #remove small controus

    min_area = 2500
    length = len(contours)
    i=0
    while(i<length):
        x,y,w, h = cv2.boundingRect(contours[i])
        area= w*h
        if area <= min_area:
            del contours[i]
            i=i-1
            length = length -1

        i = i+1

    #draw box
    img_par = [] #store paragraphs here
    for i in range(0,len(contours)):
        x,y,w,h = cv2.boundingRect(contours[i])

        #cut into different images
        img_temp = img[y:y + h, x:x + w]
        # cut into paragraphs
        img_par.append(img_temp)
        cv2.imwrite('img_temp_'+str(a)+'/temp' + str(i) + '.jpg', img_temp)

        #draw boxes
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0,0,255), 5)

        #write numbers
        img_color = cv2.putText(img_color, str(len(contours) - i), (x+10,y+30), cv2.FONT_HERSHEY_SIMPLEX,1, (255,0,0),5)

    #draw paragraph boxes
    cv2.imwrite('end_result/bounding_box'+str(a)+'.jpg',img_color)




    #count words, area and mean gray-level value in bounding box
    print("========== Image " + str(a) + " ==========")
    img_par.reverse();
    for i in range(0,len(img_par)):

        # make one cluster
        word_ker = 11
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (word_ker, word_ker))
        img_blob = cv2.morphologyEx(img_par[i], cv2.MORPH_ERODE, strel)

        # make binary % invert
        thresh = 210
        img_bin = cv2.threshold(img_blob, thresh, 255, cv2.THRESH_BINARY_INV)[1]

        # find contours
        contours = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
        print("-----REGION " + str(i + 1)+" -----")
        print("Word count #"+str(len(contours)))
        area = img_par[i].shape[1]*img_par[i].shape[0]
        print("AREA is " + str(area) +" pixels")
        print("MEAN GRAY LEVEL is " + str(np.sum(img_par[i])/area))








import time
print('time for execution '+str(time.time()))

