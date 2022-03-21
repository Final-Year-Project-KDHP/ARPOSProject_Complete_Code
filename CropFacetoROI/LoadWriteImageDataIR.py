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
            filename = filenamearr[len(filenamearr)-1]
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
        cheeksCombined = []

        IsManual = False
        regionOfInterests = 4

        print('Initialising detector...')
        print('Initialising predictor...')

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

        roiItems = OrderedDict([
            ("cheeksCombined", (1, 2, 3, 50, 52, 13, 14, 15, 28)),
            ("lip", (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59)),
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
                gray = cv2.equalizeHist(gray)
                # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                # sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
                # alpha = 1.5  # Contrast control (1.0-3.0)
                # beta = 0  # Brightness control (0-100)
                # adjusted = cv2.convertScaleAbs(sharpen, alpha=alpha, beta=beta)
                # gray = adjusted

                detections = detector(gray, 0)

                RoiCrop = None

                IRwidth, IRheight = original.shape[:2]
                graywidth, grayheight = gray.shape[:2]

                width_ratio = (graywidth / IRwidth)
                height_ratio = (grayheight / IRheight)

                if(len(detections) == 0):
                    isprevious = True
                    IsManual = False
                    detections =previousdetection
                    if(previousdetection is None):
                        print('Manual input needed')
                        IsManual =True
                        print('SELECT FOR KEY: ' + key )
                        xIr, yIr, wIr, hIr = self.LoadandCropFilesMannually(gray,IRwidth,IRheight,graywidth,grayheight,width_ratio,height_ratio)
                        RoiCrop = original[yIr:yIr + hIr, xIr:xIr + wIr]

                else:
                    isprevious = False
                    IsManual = False
                    previousdetection = detections

                imagecount = imagecount + 1
                if(IsManual):
                    skip =0
                    # store roi
                    if key == "lip":
                        if (len(lips) < imagecount):
                            lips.append(RoiCrop)
                    elif key == "right_cheek":
                        if (len(rightcheek) < imagecount):
                            rightcheek.append(RoiCrop)
                    elif key == "left_cheek":
                        if (len(leftcheek) < imagecount):
                            leftcheek.append(RoiCrop)
                    elif key == "forehead":
                        if (len(forehead) < imagecount):
                            forehead.append(RoiCrop)
                    elif key == "cheeksCombined":
                        if (len(cheeksCombined) < imagecount):
                            cheeksCombined.append(RoiCrop)
                else:

                    for k, d in enumerate(detections):
                        shape = predictor(gray, d)
                        for (_, name) in enumerate(ROI_IDXS.keys()):
                            pts = np.zeros((len(ROI_IDXS[name]), 2), np.int32)
                            for i, j in enumerate(ROI_IDXS[name]):
                                pts[i] = [shape.part(j).x, shape.part(j).y]

                            (x, y, w, h) = cv2.boundingRect(pts)
                            isXnegative =False
                            if(x<0):
                                x= 0
                                isXnegative =True

                            roi = gray[y:y + h, x:x + w]

                            xIr = int(x / width_ratio)
                            yIr = int(y / height_ratio)
                            wIr = int(w / width_ratio)
                            hIr = int(h / height_ratio)

                            roiIR = original[yIr:yIr + hIr, xIr:xIr + wIr]

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

                            #
                            # if key == "forehead":
                            #
                            #     var1 = int(20)
                            #     var2 = int(h /2)+20
                            #     var3 = int(70)
                            #     var4 = int(w-58)
                            #     #(87, 35, 167, 56)
                            #     #35:91, 87:254
                            #
                            #     if(isXnegative):
                            #         var3=0
                            #
                            #     roi = roi[var1:int(var2), var3:var4]#y:y + h, x:x + w
                            #
                            #     # print(var1)
                            #     # print(var2)
                            #     # print(var3)
                            #     # print(var4)
                            #     # y = y + 15
                            #     # x = x + 90
                            #     # h = int(h / 2)
                            #     # w = int(w / 2)
                            #     # h = h - 20
                            #     # w = w - 35
                            #
                            # elif key == "right_cheek":
                            #     var1 = 10
                            #     var2 = int(var1 + (h / 2))
                            #     var3 = 10
                            #     var4 = int(var1 + ((w / 2) + 5))
                            #     other = int(5)
                            #
                            #     if(isXnegative):
                            #         var3=0
                            #     roi = roi[var1:int(var2), var3:int(var1 + ((w / 2) + other))]
                            #
                            #     #print(str(var1)+ ":" + str(var2) + ", "+ str(var3)+ ":" + str(var4))
                            #     # y= y+60
                            #     # x = x+40
                            #     # h=int(h/2)
                            #     # w=int(w/2)
                            #     # h = h-5
                            #     # w = w-4
                            #
                            # elif key == "left_cheek":
                            #
                            #     var1 = 10
                            #     var2 = int(15 + ((h / 2) + 5))
                            #     var3 = 15
                            #     var4 = int(15 + ((w / 2) + 5))
                            #
                            #
                            #     if(isXnegative):
                            #         var3=0
                            #
                            #     roi = roi[var1:var2, var3:var4]
                            #
                            #     if (var2 > 0):
                            #         var2Ir = int(var2 / height_ratio)   # int(var1Ir + (hIr / 2))
                            #
                            #
                            #
                            #     # y = y + 60
                            #     # x = x + 40
                            #     # h = int(h / 2)
                            #     # w = int(w / 2)
                            #     # h = h - 5
                            #     # w = w - 4
                            #
                            # elif key == "cheeksCombined":
                            #     var1 = int(15)
                            #     var2 = h
                            #     var3 = int(25)
                            #     var4 = int(w - 35)
                            #     # (87, 35, 167, 56)
                            #     # 35:91, 87:254
                            #
                            #     if (isXnegative):
                            #         var3 = 0
                            #
                            #     roi = roi[var1:int(var2), var3:var4]  # y:y + h, x:x + w
                            #     # cv2.imshow("Image", roi)
                            #     # cv2.waitKey(0)

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

                            # if(imagecount >= 275 and key == 'forehead'):
                            #     cv2.imshow("Image", roi)
                            #     cv2.waitKey(0)

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
                            if (key == "lip"):  # if roi is not correctly detected and cropped
                                h, w = roiIR.shape
                                if (h <= 0 or w <= 0):
                                    print('SELECT REGION: ' + key)
                                    # Select ROI
                                    ROI0 = cv2.selectROI(original, 0, 0)
                                    # # Crop image
                                    roiIR = original[int(ROI0[1]):int(ROI0[1] + ROI0[3]),
                                            int(ROI0[0]):int(ROI0[0] + ROI0[2])]
                                    cv2.waitKey(0)
                                    cv2.destroyAllWindows()

                            # store roi
                            if key == "lip":
                                if(len(lips)<imagecount):
                                    lips.append(roiIR)
                            elif key == "right_cheek":
                                if(len(rightcheek)<imagecount):
                                    rightcheek.append(roiIR)
                            elif key == "left_cheek":
                                if(len(leftcheek)<imagecount):
                                    leftcheek.append(roiIR)
                            elif key == "forehead":
                                if(len(forehead)<imagecount):
                                    forehead.append(roiIR)
                            elif key == "cheeksCombined":
                                if(len(cheeksCombined)<imagecount):
                                    cheeksCombined.append(roiIR)
                            # # Display cropped image
                            # grayIr = (roiIR / 256).astype('uint8')
                            # grayIr = imutils.resize(grayIr, width=600)
                            # grayIr = cv2.equalizeHist(grayIr)
                            # cv2.imshow("Image", grayIr)
                            # cv2.waitKey(0)
        print('Storing ROIs on disk...')

        directorylips = savepath + r'/lips/'
        if not os.path.exists(directorylips):
            os.makedirs(directorylips)

        directoryforehead = savepath + r'/forehead/'
        if not os.path.exists(directoryforehead):
            os.makedirs(directoryforehead)

        directorycheeksCombined = savepath + r'/cheeksCombined/'
        if not os.path.exists(directorycheeksCombined):
            os.makedirs(directorycheeksCombined)

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
        for cheeksCombinedImg in cheeksCombined:
            # timestamp
            tamp = str(timestamp[count])

            # save cropped image
            Imgpath = directorycheeksCombined + "cropped-" + tamp + ".png"
            cv2.imwrite(Imgpath, cheeksCombinedImg)
            count += 1

        print('cheeksCombined Completed...')

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

    def LoadandCropFilesMannually(self,imgModified,IRwidth,IRheight,graywidth,grayheight,width_ratio,height_ratio):
        # read the image
        gray = imgModified.copy()

        # # Select ROI and Crop RoiCrop
        ROI0 = cv2.selectROI(gray, 0, 0)
        RoiCrop = gray[int(ROI0[1]):int(ROI0[1] + ROI0[3]), int(ROI0[0]):int(ROI0[0] + ROI0[2])]
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        ####RESCALE ROI
        h, w = RoiCrop.shape
        x = int(ROI0[0])
        y = int(ROI0[1])
        xIr = int(x / width_ratio)
        yIr = int(y / height_ratio)
        wIr = int(w / width_ratio)
        hIr = int(h / height_ratio)

        return xIr, yIr, wIr, hIr

    def ONLYLoadandCropFilesMannually(self,img_dir,savepath):

        lips = []
        leftcheek = []
        rightcheek = []
        forehead = []
        cheekscombined = []

        prevtimstampImg = ''
        PrevLipsCheekCrop = None
        PrevRightCheekCrop= None
        PrevLeftCheekCrop= None
        PrevForeheadCrop= None
        PrevCheeksCombinedCrop= None

        imagecount = 0
        timestamp = []
        data_path = os.path.join(img_dir, '*g')
        files = glob.glob(data_path)

        dataImages = []
        dataImagesUnchanged = []
        for f1 in files:
            filenamearr = f1.split('\\')
            filename = filenamearr[len(filenamearr) - 1]
            filename = filename.replace('.png', '')
            filenameList = filename.split('-')
            hr = filenameList[1]
            min = filenameList[2]
            sec = filenameList[3]
            mili = filenameList[4]

            timstampImg = str(hr) + '-' + str(min) + '-' + str(sec) + '-' + str(mili)
            timstampImgShort = str(hr) + '-' + str(min) + '-' + str(sec)
            dataImagesUnchanged = cv2.imread(f1, IMREAD_UNCHANGED)
            img = cv2.imread(f1)
            timestamp.append(str(timstampImg))

            if(prevtimstampImg == '' or prevtimstampImg != timstampImgShort):
                prevtimstampImg= timstampImgShort
                # read the image
                original = dataImagesUnchanged
                overlay = img.copy()
                gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
                gray = imutils.resize(gray, width=600)
                # Enhance image
                gray = cv2.equalizeHist(gray)

                imagecount = imagecount + 1

                IRwidth, IRheight = original.shape[:2]
                graywidth, grayheight = gray.shape[:2]

                width_ratio = (graywidth / IRwidth)
                height_ratio = (grayheight / IRheight)

                print('Select ForeheadCrop')
                # # Select ROI and Crop ForeheadCrop
                ROI0 = cv2.selectROI(gray, 0, 0)
                ForeheadCrop = gray[int(ROI0[1]):int(ROI0[1] + ROI0[3]), int(ROI0[0]):int(ROI0[0] + ROI0[2])]
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                ####RESCALE ForeheadCrop
                h, w = ForeheadCrop.shape
                x= int(ROI0[0])
                y= int(ROI0[1])
                xIr = int(x / width_ratio)
                yIr = int(y / height_ratio)
                wIr = int(w / width_ratio)
                hIr = int(h / height_ratio)
                ForeheadCrop = original[yIr:yIr + hIr, xIr:xIr + wIr]


                print('Select CheeksCombinedCrop')
                # # Select ROI and Crop CheeksCombinedCrop
                ROI0 = cv2.selectROI(gray, 0, 0)
                CheeksCombinedCrop = gray[int(ROI0[1]):int(ROI0[1] + ROI0[3]), int(ROI0[0]):int(ROI0[0] + ROI0[2])]
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                ####RESCALE CheeksCombinedCrop
                h, w = CheeksCombinedCrop.shape
                x= int(ROI0[0])
                y= int(ROI0[1])
                xIr = int(x / width_ratio)
                yIr = int(y / height_ratio)
                wIr = int(w / width_ratio)
                hIr = int(h / height_ratio)
                CheeksCombinedCrop = original[yIr:yIr + hIr, xIr:xIr + wIr]

                print('Select RightCheekCrop')
                # # Select ROI and Crop RightCheekCrop
                ROI0 = cv2.selectROI(gray, 0, 0)
                RightCheekCrop = gray[int(ROI0[1]):int(ROI0[1] + ROI0[3]), int(ROI0[0]):int(ROI0[0] + ROI0[2])]
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                ####RESCALE RightCheekCrop
                h, w = RightCheekCrop.shape
                x= int(ROI0[0])
                y= int(ROI0[1])
                xIr = int(x / width_ratio)
                yIr = int(y / height_ratio)
                wIr = int(w / width_ratio)
                hIr = int(h / height_ratio)
                RightCheekCrop = original[yIr:yIr + hIr, xIr:xIr + wIr]

                print('Select LeftCheekCrop')
                # # Select ROI and Crop LeftCheekCrop
                ROI0 = cv2.selectROI(gray, 0, 0)
                LeftCheekCrop = gray[int(ROI0[1]):int(ROI0[1] + ROI0[3]), int(ROI0[0]):int(ROI0[0] + ROI0[2])]
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                ####RESCALE LeftCheekCrop
                h, w = LeftCheekCrop.shape
                x= int(ROI0[0])
                y= int(ROI0[1])
                xIr = int(x / width_ratio)
                yIr = int(y / height_ratio)
                wIr = int(w / width_ratio)
                hIr = int(h / height_ratio)
                LeftCheekCrop = original[yIr:yIr + hIr, xIr:xIr + wIr]

                print('Select LipsCheekCrop')
                # # Select ROI and Crop LipsCrop
                ROI0 = cv2.selectROI(gray, 0, 0)
                LipsCheekCrop = gray[int(ROI0[1]):int(ROI0[1] + ROI0[3]), int(ROI0[0]):int(ROI0[0] + ROI0[2])]
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                ####RESCALE LipsCrop
                h, w = LipsCheekCrop.shape
                x= int(ROI0[0])
                y= int(ROI0[1])
                xIr = int(x / width_ratio)
                yIr = int(y / height_ratio)
                wIr = int(w / width_ratio)
                hIr = int(h / height_ratio)
                LipsCheekCrop = original[yIr:yIr + hIr, xIr:xIr + wIr]

                PrevLipsCheekCrop = LipsCheekCrop
                PrevRightCheekCrop = RightCheekCrop
                PrevLeftCheekCrop = LeftCheekCrop
                PrevForeheadCrop = ForeheadCrop
                PrevCheeksCombinedCrop = CheeksCombinedCrop

                lips.append(LipsCheekCrop)
                rightcheek.append(RightCheekCrop)
                leftcheek.append(LeftCheekCrop)
                forehead.append(ForeheadCrop)
                cheekscombined.append(CheeksCombinedCrop)

            elif(prevtimstampImg == timstampImgShort):

                lips.append(PrevLipsCheekCrop)
                rightcheek.append(PrevRightCheekCrop)
                leftcheek.append(PrevLeftCheekCrop)
                forehead.append(PrevForeheadCrop)
                cheekscombined.append(PrevCheeksCombinedCrop)

        print('Storing ROIs on disk...')

        directorylips = savepath + r'/lips/'
        if not os.path.exists(directorylips):
            os.makedirs(directorylips)

        directorycheekscombined = savepath + r'/cheeksCombined/'
        if not os.path.exists(directorycheekscombined):
            os.makedirs(directorycheekscombined)

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
        for chkheadImg in cheekscombined:
            # timestamp
            tamp = str(timestamp[count])

            # save cropped image
            Imgpath = directorycheekscombined + "cropped-" + tamp + ".png"
            cv2.imwrite(Imgpath, chkheadImg)
            count += 1

        print('cheeksCombined Completed...')

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

            # timestamp
            tamp = str(timestamp[count])
            # save cropped image
            Imgpath = directoryleftcheek + "cropped-" + tamp + ".png"

            cv2.imwrite(Imgpath, leftcheekImg)

            count += 1

        print('leftcheeks Completed...')

        count = 0
        for rightcheek in rightcheek:
            # timestamp
            tamp = str(timestamp[count])
            # save cropped image
            Imgpath = directoryrightcheek + "cropped-" + tamp + ".png"
            cv2.imwrite(Imgpath, rightcheek)
            count += 1

        print('Right Cheek Completed...')
