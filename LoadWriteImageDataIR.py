import glob
import os
import cv2
import numpy as np
import dlib
from collections import OrderedDict
import imutils
from cv2 import IMREAD_UNCHANGED
from PIL import Image, ImageEnhance

class LoadWriteIRROI:

    timestamp = []

    def LoadandCropFiles(self,img_dir,savepath):

        timestamp = []
        # Load Color images (test : only 10 frames)
        #img_dir = r"../StudyData/irimageMin"  # Directory containing all images
        data_path = os.path.join(img_dir, '*g')
        files = glob.glob(data_path)

        dataImages = []
        dataImagesUnchanged = []
        for f1 in files:
            filenamearr = f1.split('\\')
            filename = filenamearr[6]
            filename = filename.replace('.png', '')
            filenameList = filename.split('-')
            hr = filenameList[1]
            min = filenameList[2]
            sec = filenameList[3]
            mili = filenameList[4]

            timstampImg = str(hr) + '-' + str(min) + '-' + str(sec) + '-' + str(mili)

            imgUnchanged = cv2.imread(f1, IMREAD_UNCHANGED)
            img = cv2.imread(f1)
            dataImagesUnchanged.append(imgUnchanged)
            dataImages.append(img)
            timestamp.append(str(timstampImg))

        print('Images Loaded...')

        lips = []
        leftcheek = []
        rightcheek = []
        forehead = []

        regionOfInterests = 4

        print('Initialising detector...')
        print('Initialising predictor...')

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

        roiItems = OrderedDict([("lip", (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59)),
                                ("right_cheek", (11, 12, 13, 14, 15, 35, 53, 54)),
                                ("left_cheek", (1, 2, 3, 4, 5, 48, 49, 31)),
                                ("forehead", (17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 79, 80, 71, 70, 76, 75))
        ]
        )

        print('Initialising cropping ROIs...')
        previousdetection = None
        isprevious =False
        for key in roiItems:
            ROI_IDXS = OrderedDict([(key, roiItems[key])])

            imagecount = 0

            for f1 in dataImages:
                # read the image
                original = dataImagesUnchanged[imagecount]
                overlay = f1.copy()
                gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
                gray = imutils.resize(gray, width=600)
                # Enhance image
                gray = cv2.equalizeHist(gray)

                detections = detector(gray, 0)
                if(len(detections) == 0):
                    isprevious = True
                    detections =previousdetection
                else:
                    isprevious = False
                    previousdetection = detections

                IRwidth, IRheight = original.shape[:2]
                graywidth, grayheight = gray.shape[:2]

                width_ratio = (graywidth / IRwidth)
                height_ratio = (grayheight / IRheight)

                imagecount = imagecount + 1

                for k, d in enumerate(detections):
                    shape = predictor(gray, d)
                    for (_, name) in enumerate(ROI_IDXS.keys()):
                        pts = np.zeros((len(ROI_IDXS[name]), 2), np.int32)
                        for i, j in enumerate(ROI_IDXS[name]):
                            pts[i] = [shape.part(j).x, shape.part(j).y]

                        (x, y, w, h) = cv2.boundingRect(pts)

                        roi = gray[y:y + h, x:x + w]

                        # if(isprevious):
                        #     cv2.imshow("Image", roi)
                        #     cv2.waitKey(0)


                        xIr = int(x / width_ratio)
                        yIr = int(y / height_ratio)
                        wIr = int(w / width_ratio)
                        hIr = int(h / height_ratio)

                        roiIR = original[yIr:yIr + hIr, xIr:xIr + wIr]

                        # cv2.imshow("Image", roi)
                        # cv2.waitKey(0)

                        # grayIr = (roiIR / 256).astype('uint8')
                        # grayIr = imutils.resize(grayIr, width=600)
                        # # Enhance image
                        # grayIr = cv2.equalizeHist(grayIr)
                        # #Display cropped image
                        # # cv2.imshow("Image", grayIr)
                        # # cv2.waitKey(0)
                        #
                        # # Select ROI
                        # ROI0 = cv2.selectROI(roi, 0, 0)
                        # # # Crop image
                        # imCrop0 = roi[int(ROI0[1]):int(ROI0[1] + ROI0[3]), int(ROI0[0]):int(ROI0[0] + ROI0[2])]
                        # print(ROI0)
                        # var1=str(ROI0[1])
                        # var2=str(int(ROI0[1] + ROI0[3]))
                        # var3=str(ROI0[0])
                        # var4=str(int(ROI0[0] + ROI0[2]))
                        # print(var1 + ":" + var2 + ", "+ var3 + ":" + var4)
                        #
                        # # # Display cropped image
                        # cv2.imshow("Image", imCrop0)
                        # cv2.waitKey(0)

                        # crop_img = crop_img[0:y + h, 0:x + w]
                        # var1 = str(15)
                        # var2 = str(int(10 +(h/2)))
                        # var3 = str(int((w/2)-25))
                        # var4 = str(int(w-29))
                        # imCrop0 = crop_img[int(15):int(10 +(h/2)), int((w/2)-25):int(w-29)] #crop_img[10:int(10 + (h/2)), int(10):int(10 + ((w/2)+10)))]
                        # print(var1 + ":" + var2 + ", "+ var3 + ":" + var4)


                        #if key == "lip":
                            #remove for full roi
                            #x = x + 10
                            #w = int(w / 2)  # half the width
                            #w = w + 20

                        var1 = 0
                        var2 = 0
                        var3 = 0
                        var4 = 0

                        var1Ir =0
                        var2Ir =0
                        var3Ir =0
                        var4Ir =0


                        if key == "forehead":

                            var1 = int(20)
                            var2 = int(h /2)+20
                            var3 = int(70)
                            var4 = int(w-58)
                            #(87, 35, 167, 56)
                            #35:91, 87:254

                            roi = roi[var1:int(var2), var3:var4]#y:y + h, x:x + w

                            # print(var1)
                            # print(var2)
                            # print(var3)
                            # print(var4)
                            # y = y + 15
                            # x = x + 90
                            # h = int(h / 2)
                            # w = int(w / 2)
                            # h = h - 20
                            # w = w - 35

                        elif key == "right_cheek":
                            var1 = 10
                            var2 = int(var1 + (h / 2))
                            var3 = 10
                            var4 = int(var1 + ((w / 2) + 5))
                            other = int(5)
                            roi = roi[var1:int(var2), var3:int(var1 + ((w / 2) + other))]

                            #print(str(var1)+ ":" + str(var2) + ", "+ str(var3)+ ":" + str(var4))
                            # y= y+60
                            # x = x+40
                            # h=int(h/2)
                            # w=int(w/2)
                            # h = h-5
                            # w = w-4

                        elif key == "left_cheek":

                            var1 = 10
                            var2 = int(15 + ((h / 2) + 5))
                            var3 = 15
                            var4 = int(15 + ((w / 2) + 5))
                            roi = roi[var1:var2, var3:var4]

                            if (var2 > 0):
                                var2Ir = int(var2 / height_ratio)   # int(var1Ir + (hIr / 2))


                            # y = y + 60
                            # x = x + 40
                            # h = int(h / 2)
                            # w = int(w / 2)
                            # h = h - 5
                            # w = w - 4

                        # Map resized image cords and get cords for oringal image
                        # left cheek
                        # if key == "left_cheek":
                        #     xIr = int(x / width_ratio) + 2
                        #     yIr = int(y / height_ratio)
                        #     wIr = int(w / width_ratio) - 5
                        #     hIr = int(h / height_ratio) - 5
                        #
                        # else:
                            # right cheek, forehead , lips

                        h, w = roi.shape
                        xIr = 0#int(x / width_ratio)
                        yIr = 0#int(y / height_ratio)
                        wIr = int(w / width_ratio)
                        hIr = int(h / height_ratio)

                        if(var1>0):
                            var1Ir = int(var1 / height_ratio)

                        if (var2 > 0):
                            var2Ir = int(var2 / height_ratio)   # int(var1Ir + (hIr / 2))

                        if (var3 > 0):
                            var3Ir = int(var3/width_ratio)

                        if (var4 > 0):
                            var4Ir = int(var4/width_ratio)#int(var1Ir + ((wIr / 2) + otherIr))


                        #roi = roi[var1Ir:var2Ir, var3Ir:var4Ir]

                        #print(str(var1) + ":" + str(var2) + ", " + str(var3) + ":" + str(var4))

                        if (var2Ir > 0 and var4Ir >0):

                            if(key =="right_cheek"):
                                var2Ir =var2Ir+3
                                var1Ir=0
                                var3Ir=0

                            if(key =="left_cheek"):
                                var2Ir =var2Ir+2
                                var3Ir =var3Ir+ 2


                            if(key =="forehead"):
                                var2Ir =var2Ir+2
                                var3Ir =var3Ir+ 2
                                var1Ir =var1Ir +2

                            roiIR = roiIR[var1Ir:var2Ir, var3Ir:var4Ir]
                        else:
                            roiIR = roiIR[yIr:yIr + hIr, xIr:xIr + wIr]  # crop orignal

                        # print(str(x)+','+str(y)+','+str(w)+','+str(h))

                        # store roi
                        if key == "lip":
                            lips.append(roiIR)

                            # # Display cropped image
                            # grayIr = (roiIR / 256).astype('uint8')
                            # grayIr = imutils.resize(grayIr, width=600)
                            # grayIr = cv2.equalizeHist(grayIr)
                            # cv2.imshow("Image", grayIr)
                            # cv2.waitKey(0)
                        elif key == "right_cheek":
                            rightcheek.append(roiIR)
                        elif key == "left_cheek":
                            leftcheek.append(roiIR)
                        elif key == "forehead":
                            forehead.append(roiIR)

        print('Storing ROIs on disk...')

        directorylips = savepath + r'/lips/'
        if not os.path.exists(directorylips):
            os.makedirs(directorylips)

        directoryforehead = savepath + r'/forehead/'
        if not os.path.exists(directoryforehead):
            os.makedirs(directoryforehead)

        directoryrightcheek = savepath + r'/rightcheek/'
        if not os.path.exists(directoryrightcheek):
            os.makedirs(directoryrightcheek)

        directoryleftcheek = savepath + r'/leftcheek/'
        if not os.path.exists(directoryleftcheek):
            os.makedirs(directoryleftcheek)

        count = 0
        for lps in lips:
            tamp = str(timestamp[count])
            # save cropped image
            Imgpath = directorylips + "cropped-" + tamp + ".png"

            cv2.imwrite(Imgpath, lps)
            count += 1

        print('Lips Completed...')

        count = 0
        for frheadImg in forehead:
            # timestamp
            tamp = str(timestamp[count])

            # save cropped image
            Imgpath = directoryforehead + "cropped-" + tamp + ".png"
            cv2.imwrite(Imgpath, frheadImg)
            count += 1

        print('Forehead Completed...')

        count = 0
        for leftcheekImg in leftcheek:

            if (leftcheekImg.size > 1):
                # timestamp
                tamp = str(timestamp[count])
                # save cropped image
                Imgpath = directoryleftcheek+"cropped-" + tamp + ".png"

                cv2.imwrite(Imgpath, leftcheekImg)

            count += 1

        print('leftcheeks Completed...')

        count = 0
        for rightcheek in rightcheek:
            # timestamp
            tamp = str(timestamp[count])
            # save cropped image
            Imgpath = directoryrightcheek+"cropped-" + tamp + ".png"
            cv2.imwrite(Imgpath, rightcheek)
            count += 1

        print('Right Cheek Completed...')
