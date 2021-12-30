import cv2
import numpy as np
import pytesseract
import statistics
from PIL import Image
import os
from pylab import array, plot, show, axis, arange, figure, uint8

directory_path = os.getcwd()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
#Global
DiskPath = "E:\\ARPOS_Server_Data\\Server_Study_Data\\SouthAsian_BrownSkin_Group\\"

ParticipantNumbers= ["PIS-7180"]
hearratestatus = [ "Resting1", "Resting2", "AfterExcersize"] # , "AfterExcersize"heart rate status example resting state and after small workout

# ParticipantNumber= "PIS-4709"#"PIS-1118,PIS-2212, PIS-3252, PIS-3807, PIS-4497, PIS-5868, PIS-8308, PIS-8343, PIS-9219
# position = "Resting1"
for ParticipantNumber in ParticipantNumbers:
    print('Participant Id : ' + ParticipantNumber)

    LoadPath = DiskPath + r'\\GroundTruthData\\'+ ParticipantNumber + '\\'#r'GroundTruth/PIS-8308/Resting1/'

    for position in hearratestatus:
        print('Processing for : ' + position)

        Savepath = LoadPath + position + '\\' #r'PIS-8308/Result/GroundTruth/'
        videopath =LoadPath + position+ ".mp4"

        if not os.path.exists(Savepath):
            os.makedirs(Savepath)

        #get fps
        video = cv2.VideoCapture(videopath)
        fps = video.get(cv2.CAP_PROP_FPS)
        print(fps)

        f= open(Savepath + r"HR.txt","w+")
        f2= open(Savepath + r"SPO.txt","w+")

        # Convert all video frames to pngs
        # CommercialHR = []
        # CommercialSPO = []
        # count = 0
        framecount = 2
        # avgHRValues = []
        # avgSPOValues = []
        cap = cv2.VideoCapture(videopath)
        while not cap.isOpened():
            cap = cv2.VideoCapture(videopath)
            cv2.waitKey(1000)
            print("Wait for the header")

        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)  # p.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        endsecond =((fps *10) + (fps *60))

        TfPS =1# Target Keyframes Per Second
        hop = round(fps / TfPS) #+3

        # print(pos_frame)
        while True:
            flag, frame = cap.read()
            if flag:
                if(framecount> fps*20): ##Change here to skip initial seconds
                    if framecount % hop == 0:
                        # count = count + 1
                        # process for ocr.

                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        kernel = np.ones((1, 1), np.uint8)
                        img = cv2.dilate(img, kernel, iterations=1)
                        img = cv2.erode(img, kernel, iterations=1)

                        # #ROI
                        # fromCenter = 0
                        # showCrosshair = 0
                        # imgData = img
                        # #
                        # # # Select ROI
                        # ROI0 = cv2.selectROI(imgData, fromCenter, showCrosshair)
                        # # Crop image
                        # imCrop0 = imgData[int(ROI0[1]):int(ROI0[1] + ROI0[3]), int(ROI0[0]):int(ROI0[0] + ROI0[2])]
                        # print(imCrop0)
                        # cv2.imshow("Image", imCrop0)
                        # cv2.waitKey(0)

                        #Changes as per mobile resolution

                        Col0 = 497 #571
                        Col1 = 234 #231
                        Col2 =  131 #135
                        Col3 = 77 #79
                        crop_imgspo = img[Col1:int(Col1 + Col3), Col0:int(Col0 + Col2)]# img[311:int(311 + 126), 838:int(838 + 214)]


                        Col0 =489#584
                        Col1 = 457#534
                        Col2 = 127#114
                        Col3 = 69#72
                        crop_imghr = img[Col1:int(Col1 + Col3), Col0:int(Col0 + Col2)]#img[668:int(668 + 123), 826:int(826 + 240)]
                        # # Display cropped image
                        # cv2.imshow("Image", crop_imgspo)
                        # cv2.waitKey(0)
                        #
                        # cv2.imshow("Image", crop_imghr)
                        # cv2.waitKey(0)

                        # read text from images
                        resulthr = pytesseract.image_to_string(crop_imghr, lang='eng',
                                                               config='--psm 7')  # Image.open(saveimagepath + "1.0frame.png")
                        resultspo = pytesseract.image_to_string(crop_imgspo, lang='eng', config='--psm 7')

                        spo = resultspo.replace("\n\f", "")
                        pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)  # cap.get(1) #cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

                        hr = resulthr.replace("O", "0")
                        hr = hr.replace("\n\f", "")

                        # print(hr)
                        # print(spo)

                        # hrval = hr.isdigit()
                        # spoval = spo.isdigit()

                        # if (hrval == True and spoval == True):
                        # avgHRValues.append(int(hr))
                        # avgSPOValues.append(int(spo))

                            # if (len(avgHRValues) >= 59):
                            #     averagehr = statistics.mean(avgHRValues)
                            #     averagespo = statistics.mean(avgSPOValues)
                            #     avgHRValues = []
                            #     avgSPOValues = []
                        # CommercialHR.append(hr)
                        # CommercialSPO.append((spo)
                        # print("HR= " + str(hr) + " - SPO2= " + str(spo))
                        f.write(str(hr) + "\n")
                        f2.write(str(spo) + "\n")
                        # else:
                        #     # # Display cropped image
                        #     cv2.imshow("Image", crop_imghr)
                        #     cv2.waitKey(0)
                        #     print("NOT ADDED: HR= " + str(hr) + " - SPO2= " + str(spo))

                    # skip first 1 and last seconds (fps *1) and (fps *1) + (fps *60) end of video
                    # if (count > (fps*10) and count <= endsecond):
                    #     #paste code here for extracting with this condition
                    # elif (count > endsecond):
                    #     break;

                framecount = framecount + 1
                # cap.release()
            else:
                break
                # The next frame is not ready, so we try to read it again
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame - 1)
                print("frame is not ready")
                # It is better to wait for a while for the next frame to be ready
                cv2.waitKey(1000)

            if cv2.waitKey(10) == 27:
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break

        f.close()
        f2.close()