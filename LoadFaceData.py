import os
import glob
import cv2
import numpy as np
from cv2 import IMREAD_UNCHANGED
import datetime
import collections
import sys

from sklearn.decomposition import FastICA

from Algorithm import AlgorithmCollection

"""
LoadFaceData Class:
Paramters and functions to Load, Sort, getMean, and other pre processing loading data implemented
"""
class LoadFaceData:
    # Image r,g,b,ir,grey data arrays
    red = []
    blue = []
    green = []
    grey = []
    Irchannel = []

    #Time stamp arrays for rgb grey and ir channels
    time_list_color = []
    time_list_ir = []
    Frametime_list_ir =[]
    Frametime_list_color =[]
    timecolorCount = []
    timeirCount = []
    totalTimeinSeconds = 0

    #Depth
    distanceM = []

    #start and end time for data
    HasStartTime = 0
    StartTime = datetime.datetime.now()
    EndTime = datetime.datetime.now()

    # Ohter constatns# TODO fix for variable fps
    ColorEstimatedFPS = 0
    IREstimatedFPS = 0
    ColorfpswithTime = {}
    IRfpswithTime= {}

    def Clear(self):
        self.red = []
        self.blue = []
        self.green = []
        self.grey = []
        self.Irchannel = []
        self.time_list_color = []
        self.time_list_ir = []
        self.Frametime_list_ir = []
        self.Frametime_list_color = []
        self.timecolorCount = []
        self.timeirCount = []
        self.totalTimeinSeconds = 0
        self.distanceM = []
        self.HasStartTime = 0
        self.StartTime = datetime.datetime.now()
        self.EndTime = datetime.datetime.now()


    def getDuplicateValue(self,ini_dict):
        # finding duplicate values
        flipped = {}
        for key, value in ini_dict.items():
            if value not in flipped:
                flipped[value] = 1
            else:
                val = flipped.get(value)
                flipped[value] = val + 1
        # print(str(max(flipped.values())))
        # printing result
        # print("final_dictionary", str(flipped))
        key = [fps for fps, count in flipped.items() if count == max(flipped.values())]
        return key[0]


    """
    SortLoadedFiles:
    sorts file in time stamp order
    """
    def SortTime(self, dstimeList):
        UnsortedFiles = {}
        for k,v in dstimeList.items():
            # GET Time from filename
            # Skip first and last second
            FrameTimeStamp = k
            distance =v
            UnsortedFiles[FrameTimeStamp] = distance

        SortedFiles = collections.OrderedDict(sorted(UnsortedFiles.items()))
        return SortedFiles
    # """
    # GetEstimatedFPS:
    # Get and print Color and IR frame rate (to identify what is fps rate for data collected)
    # # """
    # def GetDistance(self,distnacepath):
    #
    #     fdistancem = open(distnacepath, "r")
    #     dstimeList= {}
    #     ##Sort first
    #     for x in fdistancem:
    #         fullline = x
    #         if (fullline.__contains__("Distance datetime")):
    #             fulllineSplited = fullline.split(" , with distance : ")
    #             dm = float(fulllineSplited[1])
    #             dt = fulllineSplited[0].replace("Distance datetime : ", "")  # 27/05/2021 06:39:10
    #             dttimesplit = dt.split(" ")
    #
    #             converteddtime = dttimesplit[1].split(":")  # datetime.datetime.strptime(dt, '%y/%m/%d %H:%M:%S')
    #             hour = int(converteddtime[0])
    #             min = int(converteddtime[1])
    #             second = int(converteddtime[2])
    #             disFrameTime = datetime.time(hour, min, second, 0)
    #             dstimeList[disFrameTime] = dm
    #
    #     SortedFiles = self.SortTime(dstimeList)
    #
    #     for key, value in SortedFiles.items():
    #         if (key == self.StartTime.time()):  # SKIP FIRST SECOND
    #             # Do nothing or add steps here if required
    #             continue
    #         elif (key == self.EndTime.time()):  # SKIP LAST SECOND
    #             continue  # Do nothing or add steps here if required
    #         else:
    #             if (self.IRfpswithTime.__contains__(key)):
    #                 currentTimeFPS = self.IRfpswithTime.get(key)
    #                 for x in range(1, currentTimeFPS):  # for constant use self.IREstimatedFPS
    #                     self.distanceM.append(float(np.abs(value)))
    #     # add distance
    #     for x in fdistancem:
    #         fullline = x
    #         if (fullline.__contains__("Distance datetime")):
    #             fulllineSplited = fullline.split(" , with distance : ")
    #             dm = float(fulllineSplited[1])
    #             dt = fulllineSplited[0].replace("Distance datetime : ", "")  # 27/05/2021 06:39:10
    #             dttimesplit = dt.split(" ")
    #
    #             converteddtime = dttimesplit[1].split(":")  # datetime.datetime.strptime(dt, '%y/%m/%d %H:%M:%S')
    #             hour = int(converteddtime[0])
    #             min = int(converteddtime[1])
    #             second = int(converteddtime[2])
    #             disFrameTime = datetime.time(hour, min, second, 0)
    #
    #             if (disFrameTime == self.StartTime.time()):  # SKIP FIRST SECOND
    #                 # Do nothing or add steps here if required
    #                 skpi = 0
    #             elif (disFrameTime == self.EndTime.time()):  # SKIP LAST SECOND
    #                 skpi = 0  # Do nothing or add steps here if required
    #             else:
    #                 if(self.IRfpswithTime.__contains__(disFrameTime)):
    #                     currentTimeFPS = self.IRfpswithTime.get(disFrameTime)
    #                     for x in range(0, currentTimeFPS): #for constant use self.IREstimatedFPS
    #                         self.distanceM.append(float(np.abs(dm)))
    #     sthop = 0

        # # REMOVING BELOW as no longer constaant for ir and color
        # if (len(self.Irchannel) > len(self.grey)):
        #
        #     differnce = len(self.time_list_ir) - len(self.time_list_color)
        #     for i in range(0, differnce):
        #         # self.time_list_ir.pop()
        #         irTime = self.Frametime_list_ir.pop()
        #         # self.timeirCount.pop()
        #         self.Irchannel.pop()
        #         # self.distanceM.pop()
        #
        # if (len(self.grey) > len(self.Irchannel)):
        #     for k, v in self.ColorfpswithTime.items():
        #         irvalue = self.IRfpswithTime.get(k)
        #         if (irvalue != v):
        #             a = 0
        #     differnce = len(self.time_list_color) - len(self.time_list_ir)
        #     for i in range(0, differnce):
        #         # self.time_list_color.pop()
        #         ColorTime = self.Frametime_list_color.pop()
        #         # self.timecolorCount.pop()
        #         self.red.pop()
        #         self.green.pop()
        #         self.blue.pop()
        #         self.grey.pop()
            # Reevaluate seconds


    """
    LoadFiles:
    Load file from path
    """
    def LoadFiles(self, filepath):
        data_path = os.path.join(filepath, '*g')
        Image_Files = glob.glob(data_path)
        return Image_Files

    """
    SortLoadedFiles:
    sorts file in time stamp order
    """
    def SortLoadedFiles(self, Image_Files):
        UnsortedFiles = {}
        for f1 in Image_Files:
            # GET Time from filename
            # Skip first and last second
            filenamearr = f1.split('\\')
            filename = filenamearr[len(filenamearr)-1]
            filename = filename.replace('.png', '')
            filenameList = filename.split('-')
            hr = filenameList[1]
            min = filenameList[2]
            sec = filenameList[3]
            mili = filenameList[4]
            FrameTimeStamp = self.GetFrameTime(hr, min, sec, mili)

            img = cv2.imread(f1, IMREAD_UNCHANGED)

            UnsortedFiles[FrameTimeStamp] = img

        SortedFiles = collections.OrderedDict(sorted(UnsortedFiles.items()))
        return SortedFiles

    """
    GetFrameTime:
    returns Frame Time Stamp in date time format
    """
    def GetFrameTime(self, hr, min, sec, mili):
        year = datetime.datetime.now().year
        month = datetime.datetime.now().month
        day = datetime.datetime.now().day
        FrameTimeStamp = datetime.datetime(year, month, day, int(hr), int(min), int(sec), int(mili))
        return FrameTimeStamp

    """
    ProcessColorImagestoArray:
    Load color region of interests, get average of b,g,r and time.
    skip first and last seconds 
    """
    def ProcessColorImagestoArray(self, filepath):

        Image_Files = self.LoadFiles(filepath)
        Image_Files = self.SortLoadedFiles(Image_Files)
        LastFileTimeStamp = list(Image_Files.keys())[-1]
        self.EndTime = self.GetFrameTime(LastFileTimeStamp.hour, LastFileTimeStamp.minute, LastFileTimeStamp.second, 0)
        ColorfpswithTime = {}
        prevFrameTimeStamp = None
        fpscountcolor=0
        count=0
        # Go through each image
        for key, value in Image_Files.items():

            FrameTimeStamp = key
            FrameTimeStampWOMili = datetime.datetime(FrameTimeStamp.year, FrameTimeStamp.month, FrameTimeStamp.day,
                                                     FrameTimeStamp.hour, FrameTimeStamp.minute, FrameTimeStamp.second,
                                                     0)

            # Get start time
            if (self.HasStartTime == 0):
                self.StartTime = FrameTimeStampWOMili
                self.HasStartTime = 1

            if (FrameTimeStampWOMili == self.StartTime):  # SKIP FIRST SECOND
                continue
                # Do nothing or add steps here if required
            elif (FrameTimeStampWOMili == self.EndTime):  # SKIP LAST SECOND
                continue
                # Do nothing or add steps here if required
            else:
                img = value

                # split channels
                b, g, r, a = cv2.split(img)

                # mean data
                BmeanValues = cv2.mean(b)
                GmeanValues = cv2.mean(g)
                RmeanValues = cv2.mean(r)
                greymeanValues = (BmeanValues[0] + GmeanValues[0] + RmeanValues[0]) / 3  # r+g+b/pixel count = grey

                # add to list
                self.blue.append(BmeanValues[0])
                self.green.append(GmeanValues[0])
                self.red.append(RmeanValues[0])
                self.grey.append(greymeanValues)

                # Add Time Stamp with miliseconds
                self.Frametime_list_color.append(FrameTimeStamp)
                self.timecolorCount.append(count)
                count = count +1

                #Color fps
                TrimmedTime = datetime.time(FrameTimeStamp.hour, FrameTimeStamp.minute, FrameTimeStamp.second)
                if(prevFrameTimeStamp == None):
                    prevFrameTimeStamp =  TrimmedTime
                    fpscountcolor = 1
                else:
                    if(prevFrameTimeStamp == TrimmedTime):
                        fpscountcolor = fpscountcolor +1
                    else:
                        ColorfpswithTime[prevFrameTimeStamp] = fpscountcolor
                        prevFrameTimeStamp = TrimmedTime
                        fpscountcolor= 1
        self.ColorEstimatedFPS = self.getDuplicateValue(ColorfpswithTime) #Only one time
        self.ColorfpswithTime = ColorfpswithTime
        # temp = self.grey
        # temp = np.array(temp).reshape((len(temp), 1))


    def getDistance(self,distnacepath):
        fdistancem = open(distnacepath, "r")
        dstimeList = {}
        ##Sort first
        for x in fdistancem:
            fullline = x
            if (fullline.__contains__("Distance datetime")):
                fulllineSplited = fullline.split(" , with distance : ")
                dm = float(fulllineSplited[1])
                dt = fulllineSplited[0].replace("Distance datetime : ", "")  # 27/05/2021 06:39:10
                dttimesplit = dt.split(" ")

                converteddtime = dttimesplit[1].split(":")  # datetime.datetime.strptime(dt, '%y/%m/%d %H:%M:%S')
                hour = int(converteddtime[0])
                min = int(converteddtime[1])
                second = int(converteddtime[2])
                disFrameTime = datetime.time(hour, min, second, 0)
                dstimeList[disFrameTime] = dm

        distanceData = self.SortTime(dstimeList)
        return distanceData
    """
    ProcessIRImagestoArray:
    Load IR region of interests, get average of b,g,r and time and distance
    also make sure color and ir data has same x and y values for processing and plotting
    skip first and last seconds 
    """
    def ProcessIRImagestoArray(self, filepath,LoadDistancePath):

        Image_Files = self.LoadFiles(filepath)
        Image_Files = self.SortLoadedFiles(Image_Files)
        IRfpswithTime = {}
        fpscountir = 0
        prevFrameTimeStamp=None
        count = 0
        countIR = 0
        distanceData = self.getDistance(LoadDistancePath)
        # Go through each image
        for key, value in Image_Files.items():


            FrameTimeStamp = key
            FrameTimeStampWOMili = datetime.datetime(FrameTimeStamp.year, FrameTimeStamp.month, FrameTimeStamp.day,
                                                     FrameTimeStamp.hour, FrameTimeStamp.minute, FrameTimeStamp.second,
                                                     0)

            if (FrameTimeStampWOMili == self.StartTime):  # SKIP FIRST SECOND
                # Do nothing or add steps here if required
                continue
            elif (FrameTimeStampWOMili == self.EndTime):  # SKIP LAST SECOND
                # Do nothing or add steps here if required
                continue
            else:
                img = value
                # dimensions = img.shape
                ImgmeanValues = cv2.mean(img)  # single channel  RmeanValue = cv2.mean(r)

                self.Irchannel.append(ImgmeanValues[0])

                TrimmedTime = datetime.time(FrameTimeStamp.hour, FrameTimeStamp.minute, FrameTimeStamp.second)

                self.Frametime_list_ir.append(FrameTimeStamp)
                self.timeirCount.append(countIR)
                #Distance
                distanceinM = distanceData.get(TrimmedTime)
                self.distanceM.append(float(np.abs(distanceinM)))
                countIR = countIR +1
                count = count+1

                #IR fps
                if(prevFrameTimeStamp == None):
                    prevFrameTimeStamp =  TrimmedTime
                    fpscountir = 1
                else:
                    if(prevFrameTimeStamp == TrimmedTime):
                        fpscountir = fpscountir +1
                    else:
                        IRfpswithTime[prevFrameTimeStamp] = fpscountir
                        prevFrameTimeStamp = TrimmedTime
                        fpscountir= 1

        # A=0
        self.IREstimatedFPS = self.getDuplicateValue(IRfpswithTime)
        self.IRfpswithTime = IRfpswithTime
        self.totalTimeinSeconds = len(self.Irchannel) / self.IREstimatedFPS

