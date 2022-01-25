import glob
import os
import cv2
import numpy as np
import dlib
from collections import OrderedDict
from cv2 import IMREAD_UNCHANGED

"""
LoadWriteROI Class:
This class extracts region of interests (ROIS) from face image and creates lips, right and left cheek and forehead images.
Enhances IR image to detect face and extract same region of interests.
ROIS extracted are store in folder path.
"""
class LoadWriteROI:
    timestamp = []

    def LoadFiles(self,img_dir,savepath):
        # Load Color images (test : only 10 frames)
        #img_dir = r"PIS-001/Resting1/Color"  # Directory containing all images
        data_path = os.path.join(img_dir, '*g')
        files = glob.glob(data_path)
        dataImages = []
        timestamp = []
        timestamp2 = []
        # if("E:\\StudyData\\UnCompressed\\PIS-2212\\AfterExcersize\\Color\\Color-5-24-9-347-W273-H349.png" in files):
        #     t= 1
        for f1 in files:
            filenamearr = f1.split('\\')
            filename = filenamearr[len(filenamearr)-1]
            filename = filename.replace('.png', '')
            filenameList = filename.split('-')
            hr = filenameList[1]
            min = filenameList[2]
            sec = filenameList[3]
            mili = filenameList[4]
            w = filenameList[5]
            h = filenameList[6]

            timstampImg = str(hr) + '-' + str(min) + '-' + str(sec) + '-' + str(mili)
            timstampImg2 = str(hr) + '-' + str(min) + '-' + str(sec)

            img = cv2.imread(f1, IMREAD_UNCHANGED)
            dataImages.append(img)
            timestamp.append(str(timstampImg))
            timestamp2.append(str(timstampImg2))

        print('Images Loaded...')
        # Load the detector

        detector = dlib.get_frontal_face_detector()

        # Load the predictor
        predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")

        print('Initialising detector...')
        print('Initialising predictor...')

        lips = []
        leftcheek = []
        rightcheeks = []
        forehead = []

        regionOfInterests = 4

        roiItems = OrderedDict([
            ("lip", (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59)),
                               ("right_cheek", (11, 12, 13, 14, 15, 35, 53, 54)),
                                ("left_cheek", (1, 2, 3, 4, 5, 48, 49, 31)),
                                ("forehead", (17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 79, 80, 71, 70, 76, 75))
        ])

        print('Initialising cropping ROIs...')
        previousimage = None
        prevx=0
        prevy=0
        prevw=0
        prevh=0
        previousdetection = None
        isprevious = False
        getFramewithDetection =False
        CurrentIndexData = None
        # for vl in range(regionOfInterests):
        for key in roiItems:
            # if (getFramewithDetection and detections is not None):
            #     break
            # = roiItems[key]
            ROI_IDXS = OrderedDict([(key, roiItems[key])])


            for f1 in dataImages:
                #
                # if (getFramewithDetection and detections is not None):
                #     break
                overlay = f1.copy()
                gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
                detections = detector(gray, 0)
                if(len(detections) == 0):
                    isprevious = True
                    detections =previousdetection
                    if(previousdetection == None):
                        getFramewithDetection = True
                        continue
                else:
                    isprevious = False
                    previousdetection = detections
                    break;

                # for k, d in enumerate(detections):  # k=img, d=facebox
                #
                #     # predictor
                #     shape = predictor(gray, d)
                #
                #     for (_, name) in enumerate(ROI_IDXS.keys()):
                #         pts = np.zeros((len(ROI_IDXS[name]), 2), np.int32)
                #         for i, j in enumerate(ROI_IDXS[name]):
                #             pts[i] = [shape.part(j).x, shape.part(j).y]
                #
                #         if(isprevious):
                #             a=0 #Do nothing
                #         else:
                #             (x, y, w, h) = cv2.boundingRect(pts)
                #
                #             if(getFramewithDetection):
                #                 CurrentIndexData =detections#pts
                #                 break
                #
                # if (getFramewithDetection and len(detections) > 0):
                #     break

            countImg=0
            prevTime = None
            prevROI=None
            for f1 in dataImages:
                # read the image
                Original = f1.copy()
                # if Original
                # previousimage = Original

                overlay = f1.copy()

                gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
                # sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                # sharpen = cv2.filter2D(gray, -1, sharpen_kernel)
                #
                # alpha = 1.5  # Contrast control (1.0-3.0)
                # beta = 0  # Brightness control (0-100)
                #
                # adjusted = cv2.convertScaleAbs(sharpen, alpha=alpha, beta=beta)
                # gray = adjusted
                #
                # gray = cv2.addWeighted(gray, alpha, np.zeros(gray.shape, gray.dtype), 0, beta)

                # G = cv2.imguidedfilter(I, 'DegreeOfSmoothing', 0.005);
                # J = imsharpen(G, 'Amount', 2);

                detections = detector(gray, 0)

                if(len(detections) == 0):
                    isprevious = True
                    detections =previousdetection
                    # if(len(detections) == 0):
                    #     # getFramewithDetection = True
                    #     detections = CurrentIndexData
                    # elif(getFramewithDetection):
                else:
                    isprevious = False
                    previousdetection = detections

                for k, d in enumerate(detections):  # k=img, d=facebox
                    # predictor
                    shape = predictor(gray, d)

                    for (_, name) in enumerate(ROI_IDXS.keys()):
                        pts = np.zeros((len(ROI_IDXS[name]), 2), np.int32)
                        for i, j in enumerate(ROI_IDXS[name]):
                            pts[i] = [shape.part(j).x, shape.part(j).y]

                        (x, y, w, h) = cv2.boundingRect(pts)

                        if(getFramewithDetection):
                            getFramewithDetection = False
                            CurrentIndexData =pts

                        if (h<=0) or (w<=0):
                            # cv2.imshow("Image", roi)
                            # cv2.waitKey(0)
                            Original=previousimage
                            x=prevx
                            y=prevy
                            w=prevw
                            h=prevh
                            # cv2.imshow("Image", Original)
                            # cv2.waitKey(0)
                            # cv2.imshow("Image", roi)
                            # cv2.waitKey(0)

                        else:
                            previousimage=Original
                            prevx = x
                            prevy= y
                            prevw= w
                            prevh= h

                        if (x<=0):
                            x=0

                        if (y<=0):
                            y=0

                        roi = Original[y:y + h, x:x + w]

                        #crop_img = Original[y:y + h, x:x + w]
                        h, w, c =roi.shape

                        # # #Display cropped image
                        # cv2.imshow("Image", Original)
                        # cv2.waitKey(0)
                        # # #Display cropped image
                        # cv2.imshow("Image", roi)
                        # cv2.waitKey(0)

                        # if key == "lip":
                        #     #remove for full roi
                        #     x = x + 10
                        #     w = int(w / 2)  # half the width
                        #     w = w+20
                        if key == "forehead":
                            # cv2.imshow("Image", roi)
                            # cv2.waitKey(0)
                            # a = 0
                            roi = roi[int(15):int(10 + (h / 2)), int((w / 2) - 25):int(w - 29)] # for all
                            # roi = roi[int(10):int(10 + (h / 2)), int(19):int(w - 30)]# for one specific
                            # y = y + 15
                            # x = x + 90
                            # h = int(h / 2)
                            # w = int(w / 2)
                            # h = h - 20
                            # w = w - 35
                            #w = w - 50
                            # h = h - 25
                            # x = x + 30
                            # y = y - 5
                        elif key == "right_cheek":
                            roi = roi[10:int(10 + (h / 2)), 10:int(10 + ((w / 2) + 5))]
                            # if (h > 100):# for one specific
                            #     # ROI0 = cv2.selectROI(roi, 0, 0)
                            #     roi = roi[int(0):int(50), int(0):int(50)]  # for one specific
                            # else:
                            #     roi = roi[int(0):int(10 + (h / 2)), int(8):int(w - 20)]  # for one specific

                            # cv2.imshow("Image", roi)
                            # cv2.waitKey(0)
                            a=0
                            # y= y+40
                            # x = x+12
                            # h=int(h/2)
                            # w=int(w/2)
                            # h = h-5
                            # w = w-4

                        elif key == "left_cheek":

                            # if(h>100):# for one specific
                            #     roi = roi[int(5):int(5+ (h / 2)), int(26):int(w )]# for one specific
                            # else:
                            #     roi = roi[int(0):int(10 + (h / 2)), int(8):int(w - 20)]# for one specific
                            # cv2.imshow("Image", roi)
                            # cv2.waitKey(0)
                            a=0
                            roi = roi[10:int(15 + ((h / 2) + 5)), 15:int(15 + ((w / 2) + 5))] #for all
                            # y = y + 40
                            # x = x + 40
                            # h = int(h / 2)
                            # w = int(w / 2)
                            # h = h - 5
                            # w = w - 4

                        # elif key == "lip":  # for one specific
                        #     if (h > 25):
                        #         # ROI0 = cv2.selectROI(roi, 0, 0)
                        #         roi = roi[int(1):int(25), int(2):int(w)]  # for one specific

                        # cv2.imshow("Image", roi)
                        # cv2.waitKey(0)
                        #roi = Original[y:y + h, x:x + w]

                        if (h<=0 or w <=0):#if roi is not correctly detected and cropped
                            print('SELECT REGION: ' + key)
                            # Select ROI
                            ROI0 = cv2.selectROI(Original, 0, 0)
                            # # Crop image
                            roi = Original[int(ROI0[1]):int(ROI0[1] + ROI0[3]), int(ROI0[0]):int(ROI0[0] + ROI0[2])]
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

                        # if key == "lip":
                        #     if(h>23):
                        #         if(prevTime == None):#==timestamp2[countImg]
                        #             ROI0 = cv2.selectROI(Original, 0, 0)
                        #             # # Crop image
                        #             roi = Original[int(ROI0[1]):int(ROI0[1] + ROI0[3]),
                        #                   int(ROI0[0]):int(ROI0[0] + ROI0[2])]
                        #             cv2.waitKey(0)
                        #             cv2.destroyAllWindows()
                        #             prevTime =timestamp2[countImg]
                        #             prevROI = ROI0
                        #         else:
                        #             if(prevTime == timestamp2[countImg]):
                        #                 roi = prevROI
                        #             else:
                        #                 ROI0 = cv2.selectROI(Original, 0, 0)
                        #                 # # Crop image
                        #                 roi = Original[int(ROI0[1]):int(ROI0[1] + ROI0[3]),
                        #                       int(ROI0[0]):int(ROI0[0] + ROI0[2])]
                        #                 cv2.waitKey(0)
                        #                 cv2.destroyAllWindows()
                        #                 prevTime = timestamp2[countImg]
                        #                 prevROI = ROI0

                        # store roi
                        if key == "lip":
                            lips.append(roi)
                        elif key == "right_cheek":
                            rightcheeks.append(roi)
                        elif key == "left_cheek":
                            leftcheek.append(roi)
                        elif key == "forehead":
                            forehead.append(roi)

                countImg = countImg +1

        count = 0

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

        for lps in lips:
            tamp = str(timestamp[count])
            # save cropped image
            Imgpath = directorylips + "cropped-" + tamp + ".png"
            #
            # # Crop Image
            # width, height = frheadImg.shape[:2]
            #
            # # forehead
            # x = 15
            # y = 20
            # w = width - 30
            # h = height - 35
            #
            # frheadImg = frheadImg[x:x + w, y:y + h]

            cv2.imwrite(Imgpath, lps)
            count += 1

        print('Lips Completed...')

        count = 0
        for frheadImg in forehead:
            if(frheadImg is None):
                sdf =4
            else:
                # Crop Image
                width, height = frheadImg.shape[:2]

                # forehead
                # x = 15
                # y = 20
                # w = width - 30
                # h = height - 35

                #frheadImg = frheadImg[x:x + w, y:y + h]

                # timestamp
                tamp = str(timestamp[count])

                # save cropped image
                Imgpath = directoryforehead +"cropped-" + tamp + ".png"
                cv2.imwrite(Imgpath, frheadImg)
                count += 1

        print('Forehead Completed...')

        count = 0
        for leftcheekImg in leftcheek:

            # timestamp
            tamp = str(timestamp[count])
            # save cropped image
            Imgpath = directoryleftcheek + "cropped-" + tamp + ".png"

            # Crop Image
            #width, height = leftcheekImg.shape[:2]

            # if (height != 0):
            #     x = 5
            #     y = 5
            #
            #     w = width
            #     if (width > 0):
            #         w = width - 25
            #
            #     h = height
            #     if (height > 0):
            #         h = height - 25

                # leftcheekImg = leftcheekImg[y:y + h, x:x + w]

            cv2.imwrite(Imgpath, leftcheekImg)
            count += 1

            # else:
            #     print(' Failed : ' + Imgpath)


        print('leftcheeks Completed...')

        count = 0
        for rightcheek in rightcheeks:
            # Crop Image
            width, height = rightcheek.shape[:2]

            # x = 10
            # y = 5
            # w = width - 35
            # h = height - 10
            #
            # rightcheek = rightcheek[x:x + w, y:y + h]

            # timestamp
            tamp = str(timestamp[count])
            # save cropped image
            Imgpath = directoryrightcheek +"cropped-" + tamp + ".png"
            cv2.imwrite(Imgpath, rightcheek)
            count += 1

        print('Right Cheek Completed...')

    def ONLYLoadandCropFilesMannually(self,img_dir,savepath):

        lips = []
        leftcheek = []
        rightcheek = []
        forehead = []
        prevtimstampImg = ''
        PrevLipsCheekCrop = None
        PrevRightCheekCrop= None
        PrevLeftCheekCrop= None
        PrevForeheadCrop= None

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
            timestamp.append(str(timstampImg))

            if(prevtimstampImg == '' or prevtimstampImg != timstampImgShort):
                prevtimstampImg= timstampImgShort
                # read the image
                original = dataImagesUnchanged
                originalCopy = original.copy()
                # gray = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)
                # gray = imutils.resize(gray, width=600)
                # # Enhance image
                # gray = cv2.equalizeHist(gray)
                imagecount = imagecount + 1
                # # Select ROI and Crop ForeheadCrop
                ROI0 = cv2.selectROI(originalCopy, 0, 0)
                ForeheadCrop = originalCopy[int(ROI0[1]):int(ROI0[1] + ROI0[3]), int(ROI0[0]):int(ROI0[0] + ROI0[2])]
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # # Select ROI and Crop RightCheekCrop
                ROI0 = cv2.selectROI(originalCopy, 0, 0)
                RightCheekCrop = originalCopy[int(ROI0[1]):int(ROI0[1] + ROI0[3]), int(ROI0[0]):int(ROI0[0] + ROI0[2])]
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # # Select ROI and Crop LeftCheekCrop
                ROI0 = cv2.selectROI(originalCopy, 0, 0)
                LeftCheekCrop = originalCopy[int(ROI0[1]):int(ROI0[1] + ROI0[3]), int(ROI0[0]):int(ROI0[0] + ROI0[2])]
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                # # Select ROI and Crop LipsCrop
                ROI0 = cv2.selectROI(originalCopy, 0, 0)
                LipsCheekCrop = originalCopy[int(ROI0[1]):int(ROI0[1] + ROI0[3]), int(ROI0[0]):int(ROI0[0] + ROI0[2])]
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                PrevLipsCheekCrop = LipsCheekCrop
                PrevRightCheekCrop = RightCheekCrop
                PrevLeftCheekCrop = LeftCheekCrop
                PrevForeheadCrop = ForeheadCrop

                lips.append(LipsCheekCrop)
                rightcheek.append(RightCheekCrop)
                leftcheek.append(LeftCheekCrop)
                forehead.append(ForeheadCrop)

            elif(prevtimstampImg == timstampImgShort):

                lips.append(PrevLipsCheekCrop)
                rightcheek.append(PrevRightCheekCrop)
                leftcheek.append(PrevLeftCheekCrop)
                forehead.append(PrevForeheadCrop)

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