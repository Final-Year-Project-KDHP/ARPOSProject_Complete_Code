import os
import glob
import cv2
import numpy as np
from cv2 import IMREAD_UNCHANGED
import datetime
import collections
import sys

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
    GetEstimatedFPS:
    Get and print Color and IR frame rate (to identify what is fps rate for data collected)
    """
    ##TODO: MAKE EFFICIENT
    def GetEstimatedFPS(self,distnacepath):
        # Get FPS for color
        ColorfpswithTime = {}
        fpscountcolor = 0
        isVariable = False
        Prevlisttime = datetime.time(self.Frametime_list_color[0].hour, self.Frametime_list_color[0].minute,
                                     self.Frametime_list_color[0].second)
        for time in self.Frametime_list_color:
            TrimmedTime = datetime.time(time.hour, time.minute, time.second)
            if (Prevlisttime == TrimmedTime):
                fpscountcolor = fpscountcolor + 1
            else:
                ColorfpswithTime[Prevlisttime] = fpscountcolor
                Prevlisttime = TrimmedTime
                fpscountcolor = 1

        ColorFPS = self.getDuplicateValue(ColorfpswithTime)
        self.ColorEstimatedFPS= ColorFPS

        # print
        # print("Color fps:")
        Colorfps =0
        count=0
        for k, v in ColorfpswithTime.items():
            if(count ==0):
                Colorfps = str(v)
            else:
                if(Colorfps != str(v)):
                    isVariable=True
                    break;

            count=count+1
            # print('Time: ' + str(k) + ' , FPS: ' + str(v))

        total_frames=0
        for item in self.Frametime_list_color:
            total_frames = total_frames + 1
            frame_count = total_frames
            FPS = self.ColorEstimatedFPS
            td = datetime.timedelta(seconds=(frame_count / FPS))
            self.time_list_color.append(td)

        ####
        # print data acquistion time details
        # print('Start Time for Color:' + str(self.StartTime))
        # print('End Time for Color:' + str(self.EndTime))
        # print('Total Time:' + str(self.EndTime - self.StartTime))

        timeDifference = (self.EndTime - self.StartTime)
        self.totalTimeinSeconds = timeDifference.total_seconds()

        estimatedseconds = len(self.time_list_color) / self.ColorEstimatedFPS
        if (self.totalTimeinSeconds > estimatedseconds):
            self.totalTimeinSeconds = estimatedseconds

        Timecount = 1
        for time in self.time_list_color:
            self.timecolorCount.append(Timecount)
            Timecount = Timecount + 1
        # End
        # print('Color ROI loaded..')
        #####################################

        # Get FPS for IR
        IRfpswithTime = {}
        fpscountir = 0
        isIRVariable =False
        Prevlisttime = datetime.time(self.Frametime_list_ir[0].hour, self.Frametime_list_ir[0].minute,
                                     self.Frametime_list_ir[0].second)
        for time in self.Frametime_list_ir:
            TrimmedTime = datetime.time(time.hour, time.minute, time.second)
            if (Prevlisttime == TrimmedTime):
                fpscountir = fpscountir + 1
            else:
                IRfpswithTime[Prevlisttime] = fpscountir
                Prevlisttime = TrimmedTime
                fpscountir = 1

        # print
        # print("IR fps:")
        IRfps = 0
        count = 0
        for k, v in IRfpswithTime.items():
            if (count == 0):
                IRfps = str(v)
            else:
                if (IRfps != str(v)):
                    isIRVariable = True
                    break;

            count=count+1
            # print('Time: ' + str(k) + ' , FPS: ' + str(v))

        IRFPS = self.getDuplicateValue(IRfpswithTime)

        self.IREstimatedFPS = IRFPS


        for time in self.Frametime_list_ir:
            # Add Time Stamp
            total_frames = total_frames + 1
            frame_count = total_frames
            FPS = self.IREstimatedFPS
            td = datetime.timedelta(seconds=(frame_count / FPS))
            self.time_list_ir.append(td)

        # print time details for ir
        # print('Start Time for IR:' + str(self.StartTime))
        # print('End Time for IR:' + str(self.EndTime))
        # print('Total Time:' + str(self.EndTime - self.StartTime))
        fdistancem = open(distnacepath, "r")

        # add distance
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

                if (disFrameTime == self.StartTime.time()):  # SKIP FIRST SECOND
                    # Do nothing or add steps here if required
                    test = 0
                elif (disFrameTime == self.EndTime.time()):  # SKIP LAST SECOND
                    test2 = 0  # Do nothing or add steps here if required
                else:
                    for x in range(0, self.IREstimatedFPS):
                        self.distanceM.append(float(np.abs(dm)))

        Timecount = 1
        for time in self.time_list_ir:
            self.timeirCount.append(Timecount)
            Timecount = Timecount + 1
        # End IR
        # print('IR ROI loaded..')

        # FIX TODO if not same
        if (len(self.time_list_ir) > len(self.time_list_color)):
            differnce = len(self.time_list_ir) - len(self.time_list_color)
            for i in range(0, differnce):
                self.time_list_ir.pop()
                self.Frametime_list_ir.pop()
                self.timeirCount.pop()
                self.Irchannel.pop()
                # self.distanceM.pop()

        if (len(self.time_list_color) > len(self.time_list_ir)):
            differnce = len(self.time_list_color) - len(self.time_list_ir)
            for i in range(0, differnce):
                self.time_list_color.pop()
                self.Frametime_list_color.pop()
                self.timecolorCount.pop()
                self.red.pop()
                self.green.pop()
                self.blue.pop()
                self.grey.pop()
            # Reevaluate seconds
            self.totalTimeinSeconds = len(self.time_list_color) / self.IREstimatedFPS


        return ColorfpswithTime, IRfpswithTime, isVariable, isIRVariable, ColorFPS, IRFPS

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
                skip =True
                # Do nothing or add steps here if required
            elif (FrameTimeStampWOMili == self.EndTime):  # SKIP LAST SECOND
                skip = True
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

    """
    ProcessIRImagestoArray:
    Load IR region of interests, get average of b,g,r and time and distance
    also make sure color and ir data has same x and y values for processing and plotting
    skip first and last seconds 
    """
    def ProcessIRImagestoArray(self, filepath):

        Image_Files = self.LoadFiles(filepath)
        Image_Files = self.SortLoadedFiles(Image_Files)
        total_frames=0
        # Go through each image
        for key, value in Image_Files.items():


            FrameTimeStamp = key
            FrameTimeStampWOMili = datetime.datetime(FrameTimeStamp.year, FrameTimeStamp.month, FrameTimeStamp.day,
                                                     FrameTimeStamp.hour, FrameTimeStamp.minute, FrameTimeStamp.second,
                                                     0)

            if (FrameTimeStampWOMili == self.StartTime):  # SKIP FIRST SECOND
                # Do nothing or add steps here if required
                skip = True
            elif (FrameTimeStampWOMili == self.EndTime):  # SKIP LAST SECOND
                # Do nothing or add steps here if required
                skip = True
            else:
                img = value

                ImgmeanValues = cv2.mean(img)  # single channel  RmeanValue = cv2.mean(r)
                self.Irchannel.append(ImgmeanValues[0])

                self.Frametime_list_ir.append(FrameTimeStamp)
