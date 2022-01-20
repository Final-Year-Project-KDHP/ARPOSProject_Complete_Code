import os
import glob

from matplotlib import pyplot as plt
from scipy import signal
import scipy.interpolate as interp
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

    Tempred = []
    Tempblue = []
    Tempgreen = []
    Tempgrey = []
    TempIrchannel=[]
    TempFrametime_list_ir = []
    TempFrametime_list_color = []
    TemptimecolorCount = []
    TemptimeirCount = []
    TempdistanceM = []

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

    # Ohter constatns#
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

        self.Tempred = []
        self.Tempblue = []
        self.Tempgreen = []
        self.Tempgrey = []
        self.TempIrchannel = []
        self.TempdistanceM = []
        self.TempFrametime_list_ir = []
        self.TempFrametime_list_color = []
        self.TemptimecolorCount = []
        self.TemptimeirCount = []

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
        DataColorfpsTimeWise = {}
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
                count = count + 1

                #Color fps
                TrimmedTime = datetime.time(FrameTimeStamp.hour, FrameTimeStamp.minute, FrameTimeStamp.second)
                if(prevFrameTimeStamp == None):
                    prevFrameTimeStamp =  TrimmedTime
                    fpscountcolor = 1
                    self.TempFrametime_list_color.append(FrameTimeStamp)

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

                    # Temp fps wise
                    self.Tempblue.append(BmeanValues[0])
                    self.Tempgreen.append(GmeanValues[0])
                    self.Tempred.append(RmeanValues[0])
                    self.Tempgrey.append(greymeanValues)
                    self.TemptimecolorCount.append(count)

                    # Add Time Stamp with miliseconds
                    self.Frametime_list_color.append(FrameTimeStamp)
                    self.timecolorCount.append(count)

                else:
                    if(prevFrameTimeStamp == TrimmedTime):
                        fpscountcolor = fpscountcolor +1
                        self.TempFrametime_list_color.append(FrameTimeStamp)

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

                        # Temp fps wise
                        self.Tempblue.append(BmeanValues[0])
                        self.Tempgreen.append(GmeanValues[0])
                        self.Tempred.append(RmeanValues[0])
                        self.Tempgrey.append(greymeanValues)
                        self.TemptimecolorCount.append(count)

                        # Add Time Stamp with miliseconds
                        self.Frametime_list_color.append(FrameTimeStamp)
                        self.timecolorCount.append(count)
                    else:
                        ##Record FPS WISE here?
                        objChannelDataClass = ChannelDataClass(self.Tempblue,self.Tempgreen,self.Tempred,self.Tempgrey,None,fpscountcolor,self.TempFrametime_list_color,
                                                               None,self.TemptimecolorCount,None,None)
                        DataColorfpsTimeWise[prevFrameTimeStamp] =objChannelDataClass
                        self.Tempred = []
                        self.Tempblue = []
                        self.Tempgreen = []
                        self.Tempgrey = []
                        self.TempFrametime_list_color = []
                        self.TemptimecolorCount = []
                        ##UpSample here?
                        ColorfpswithTime[prevFrameTimeStamp] = fpscountcolor
                        prevFrameTimeStamp = TrimmedTime
                        fpscountcolor= 1
                        self.TempFrametime_list_color.append(FrameTimeStamp)

                        img = value

                        # split channels
                        b, g, r, a = cv2.split(img)

                        # mean data
                        BmeanValues = cv2.mean(b)
                        GmeanValues = cv2.mean(g)
                        RmeanValues = cv2.mean(r)
                        greymeanValues = (BmeanValues[0] + GmeanValues[0] + RmeanValues[
                            0]) / 3  # r+g+b/pixel count = grey

                        # add to list
                        self.blue.append(BmeanValues[0])
                        self.green.append(GmeanValues[0])
                        self.red.append(RmeanValues[0])
                        self.grey.append(greymeanValues)

                        # Temp fps wise
                        self.Tempblue.append(BmeanValues[0])
                        self.Tempgreen.append(GmeanValues[0])
                        self.Tempred.append(RmeanValues[0])
                        self.Tempgrey.append(greymeanValues)
                        self.TemptimecolorCount.append(count)

                        # Add Time Stamp with miliseconds
                        self.Frametime_list_color.append(FrameTimeStamp)
                        self.timecolorCount.append(count)
        self.ColorEstimatedFPS = self.getDuplicateValue(ColorfpswithTime) #Only one time
        self.ColorfpswithTime = {} # ColorfpswithTime
        # temp = self.grey
        # temp = np.array(temp).reshape((len(temp), 1))
        for k,v in DataColorfpsTimeWise.items():
            #resample ehre
            if(v.fpscount <self.ColorEstimatedFPS):

                totalSzieforRsmaple = (self.ColorEstimatedFPS - v.fpscount)
                if(totalSzieforRsmaple + len(v.blue) > self.ColorEstimatedFPS ):
                    totalSzieforRsmaple = totalSzieforRsmaple -1
                Addtional = v.blue[-totalSzieforRsmaple:]
                v.blue = v.blue + Addtional #v.blue.append()

                Addtional = v.green[-totalSzieforRsmaple:]
                v.green = v.green + Addtional #v.blue.append()
                Addtional = v.red[-totalSzieforRsmaple:]
                v.red = v.red + Addtional #v.blue.append()
                Addtional = v.grey[-totalSzieforRsmaple:]
                v.grey = v.grey + Addtional #v.blue.append()


                LastTimeStamp = v.Frametime_list_color[-1:][0]
                Addtional = []
                for x in range(0,totalSzieforRsmaple):
                    microSec = LastTimeStamp.microsecond + x
                    NewTimeStamp = datetime.datetime(LastTimeStamp.year,LastTimeStamp.month, LastTimeStamp.day,LastTimeStamp.hour, LastTimeStamp.minute, LastTimeStamp.second,microSec )
                    Addtional.append(NewTimeStamp)

                v.Frametime_list_color =v.Frametime_list_color+ Addtional

            self.ColorfpswithTime[k] = len(v.grey)
                # LastCount = v.timecolorCount[-1:][0]

                # Addtional=[]
                # for x in range(0,totalSzieforRsmaple):
                #     NewCount =LastCount + x
                #     Addtional.append(NewCount)
                # v.timecolorCount =v.timecolorCount+ Addtional

                #
                #
                # ##############ROUGH
                # #resample
                # L = len(v.blue)
                # max_time = L / v.fpscountcolor
                # x = np.linspace(0, max_time, L)  # time_steps# x = np.arange(0, 18)
                # y =v.blue #np.exp(-x / 3.0)
                #
                # f = interp.interp1d(x, y, fill_value="extrapolate")
                # max_time = L / self.ColorEstimatedFPS
                # L = self.ColorEstimatedFPS
                # xnew = np.arange(0,  L,max_time)
                # ynew = f(xnew)  # use interpolation function returned by `interp1d`
                # plt.plot(x, y, 'o', xnew, ynew, '-')
                # plt.show()
                #
                #
                # # f = interp.interp1d(v.blue, timeCount, kind='cubic', fill_value="extrapolate")
                # # max_time = L / self.ColorEstimatedFPS
                # # x_new = np.linspace(0, max_time, self.ColorEstimatedFPS)  # time_steps
                # # ynew = f(x_new)
                # # plt.plot(v.blue, timeCount, 'o', x_new, ynew, '-')
                # # plt.show()
                # upsampleRate = int(len(v.blue)/totalSzieforRsmaple)
                # v.blue = signal.resample(v.blue, len(v.blue) * 2)
                # v.green = signal.resample(v.green, len(v.green) * 1)
                # v.red = signal.resample(v.red, self.ColorEstimatedFP)
                # v.grey = signal.resample(v.grey, len(v.grey) * upsampleRate)

        self.Tempblue = []
        self.Tempgreen = []
        self.Tempred = []
        self.Tempgrey = []
        self.TempFrametime_list_color = []
        self.TemptimecolorCount = []
        for k,v in DataColorfpsTimeWise.items():
            if (len(v.grey)) < self.ColorEstimatedFPS:  ## if too less signal, remove that frame
                stop = 0
                self.ColorfpswithTime.pop(k)
            else:
                self.Tempblue = self.Tempblue +v.blue
                self.Tempgreen = self.Tempgreen +v.green
                self.Tempred = self.Tempred +v.red
                self.Tempgrey = self.Tempgrey +v.grey
                self.TempFrametime_list_color = self.TempFrametime_list_color +v.Frametime_list_color
                self.TemptimecolorCount.append(count)

                # self.TemptimecolorCount =self.TemptimecolorCount +v.timecolorCount

        self.blue = self.Tempblue
        self.green = self.Tempgreen
        self.red = self.Tempred
        self.grey = self.Tempgrey
        self.Frametime_list_color = self.TempFrametime_list_color
        self.timecolorCount = list(range(0, len(self.TempFrametime_list_color)))#self.TemptimecolorCount
        #Recalculate
        self.ColorEstimatedFPS = self.getDuplicateValue(self.ColorfpswithTime)
        a=0


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
        DataIRfpsTimeWise = {}
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
                TrimmedTime = datetime.time(FrameTimeStamp.hour, FrameTimeStamp.minute, FrameTimeStamp.second)

                countIR = countIR +1
                count = count+1

                #IR fps
                if(prevFrameTimeStamp == None):
                    prevFrameTimeStamp =  TrimmedTime
                    fpscountir = 1

                    img = value
                    # dimensions = img.shape
                    ImgmeanValues = cv2.mean(img)  # single channel  RmeanValue = cv2.mean(r)

                    self.Irchannel.append(ImgmeanValues[0])
                    # Temp fps wise
                    self.TempIrchannel.append(ImgmeanValues[0])
                    self.TemptimeirCount.append(countIR)
                    self.TempFrametime_list_ir.append(FrameTimeStamp)

                    self.Frametime_list_ir.append(FrameTimeStamp)
                    self.timeirCount.append(countIR)
                    # Distance
                    distanceinM = distanceData.get(TrimmedTime)
                    self.distanceM.append(float(np.abs(distanceinM)))
                    self.TempdistanceM.append(float(np.abs(distanceinM)))

                else:
                    if(prevFrameTimeStamp == TrimmedTime):
                        fpscountir = fpscountir +1

                        img = value
                        # dimensions = img.shape
                        ImgmeanValues = cv2.mean(img)  # single channel  RmeanValue = cv2.mean(r)

                        self.Irchannel.append(ImgmeanValues[0])
                        # Temp fps wise
                        self.TempIrchannel.append(ImgmeanValues[0])
                        self.TemptimeirCount.append(countIR)
                        self.TempFrametime_list_ir.append(FrameTimeStamp)

                        self.Frametime_list_ir.append(FrameTimeStamp)
                        self.timeirCount.append(countIR)
                        # Distance
                        distanceinM = distanceData.get(TrimmedTime)
                        self.distanceM.append(float(np.abs(distanceinM)))
                        self.TempdistanceM.append(float(np.abs(distanceinM)))
                    else:
                        ##Record FPS WISE here?
                        objChannelDataClass = ChannelDataClass(None,None,None,None,self.TempIrchannel,fpscountir,None,self.TempFrametime_list_ir,None,self.TemptimeirCount,self.TempdistanceM)
                        DataIRfpsTimeWise[prevFrameTimeStamp] =objChannelDataClass
                        self.TempIrchannel = []
                        self.TempFrametime_list_ir = []
                        self.TemptimeirCount = []
                        self.TempdistanceM=[]

                        ##UpSample here?
                        IRfpswithTime[prevFrameTimeStamp] = fpscountir
                        prevFrameTimeStamp = TrimmedTime
                        fpscountir= 1

                        img = value
                        # dimensions = img.shape
                        ImgmeanValues = cv2.mean(img)  # single channel  RmeanValue = cv2.mean(r)

                        self.Irchannel.append(ImgmeanValues[0])
                        # Temp fps wise
                        self.TempIrchannel.append(ImgmeanValues[0])
                        self.TemptimeirCount.append(countIR)
                        self.TempFrametime_list_ir.append(FrameTimeStamp)

                        self.Frametime_list_ir.append(FrameTimeStamp)
                        self.timeirCount.append(countIR)
                        # Distance
                        distanceinM = distanceData.get(TrimmedTime)
                        self.distanceM.append(float(np.abs(distanceinM)))
                        self.TempdistanceM.append(float(np.abs(distanceinM)))

        # A=0
        self.IREstimatedFPS = self.getDuplicateValue(IRfpswithTime)
        # self.IRfpswithTime = IRfpswithTime
        self.IRfpswithTime={}
        #a=0
        # temp = np.array(temp).reshape((len(temp), 1))
        for k, v in DataIRfpsTimeWise.items():
            # resample ehre
            if (v.fpscount < self.IREstimatedFPS):
                totalSzieforRsmaple = self.IREstimatedFPS - v.fpscount
                if(totalSzieforRsmaple + len(v.Irchannel) > self.IREstimatedFPS ):
                    totalSzieforRsmaple = totalSzieforRsmaple -1
                Addtional = v.Irchannel[-totalSzieforRsmaple:]
                v.Irchannel = v.Irchannel + Addtional  # v.blue.append()

                LastTimeStamp = v.Frametime_list_ir[-1:][0]
                Addtional = []
                for x in range(0, totalSzieforRsmaple):
                    microSec = LastTimeStamp.microsecond + x
                    NewTimeStamp = datetime.datetime(LastTimeStamp.year, LastTimeStamp.month, LastTimeStamp.day,
                                                     LastTimeStamp.hour, LastTimeStamp.minute, LastTimeStamp.second,
                                                     microSec)
                    Addtional.append(NewTimeStamp)

                v.Frametime_list_ir = v.Frametime_list_ir + Addtional

                # LastCount = v.timeirCount[-1:][0]
                # Addtional = []
                # for x in range(0, totalSzieforRsmaple):
                #     NewCount = LastCount + x
                #     Addtional.append(NewCount)
                # v.timeirCount = v.timeirCount + Addtional

                Lastdm = v.distanceM[-1:][0]
                Addtional = []
                for x in range(0, totalSzieforRsmaple):
                    Newdm = Lastdm
                    Addtional.append(Newdm)
                v.distanceM = v.distanceM + Addtional

            self.IRfpswithTime[k] = len(v.Irchannel)

        self.TempIrchannel = []
        self.TempFrametime_list_ir = []
        self.TemptimeirCount =[]
        self.TempdistanceM =[]
        for k, v in DataIRfpsTimeWise.items():
            if (len(v.Irchannel)) < self.IREstimatedFPS:  ## if too less signal, remove that frame
                stop = 0
                self.IRfpswithTime.pop(k)
            else:
                self.TempIrchannel =self.TempIrchannel  +v.Irchannel
                self.TempFrametime_list_ir = self.TempFrametime_list_ir  +v.Frametime_list_ir
                self.TemptimeirCount =self.TemptimeirCount  +v.timeirCount
                self.TempdistanceM =self.TempdistanceM  +v.distanceM

        self.Irchannel = self.TempIrchannel
        self.Frametime_list_ir = self.TempFrametime_list_ir
        self.timeirCount = list(range(0, len(self.TempFrametime_list_ir)))#self.TemptimeirCount
        self.distanceM =self.TempdistanceM

        #Recalculate
        self.IREstimatedFPS = self.getDuplicateValue(self.IRfpswithTime)
        self.totalTimeinSeconds = len(self.IRfpswithTime) #len(self.Irchannel) / self.IREstimatedFPS

        a=0


class ChannelDataClass:
    red = []
    blue = []
    green = []
    grey = []
    Irchannel = []
    fpscount=0
    Frametime_list_color = []
    timecolorCount = []
    timeirCount = []
    Frametime_list_ir= []
    distanceM = []
# Constructor
    def __init__(self, blue, green,red, grey,Irchannel,fpscount,Frametime_list_color,Frametime_list_ir,timecolorCount,timeirCount,distanceM):
        self.red =red
        self.blue = blue
        self.green = green
        self.grey=grey
        self.Irchannel=Irchannel
        self.fpscount =fpscount
        self.Frametime_list_color =Frametime_list_color
        self.timecolorCount = timecolorCount
        self.timeirCount = timeirCount
        self.Frametime_list_ir = Frametime_list_ir
        self.distanceM = distanceM