import os
import glob
import cv2
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

    #Depth
    distanceM = []

    #start and end time for data
    HasStartTime = 0
    StartTime = datetime.datetime.now()
    EndTime = datetime.datetime.now()

    # Ohter constatns
    EstimatedFPS = 30  # TODO fix for variable fps

    """
    GetEstimatedFPS:
    Get and print Color and IR frame rate (to identify what is fps rate for data collected)
    """
    def GetEstimatedFPS(self):
        # Get FPS for color
        ColorfpswithTime = {}
        fpscountcolor = 0
        Prevlisttime = datetime.time(self.time_listcolor[0].hour, self.time_listcolor[0].minute,
                                     self.time_listcolor[0].second)
        for time in self.time_listcolor:
            TrimmedTime = datetime.time(time.hour, time.minute, time.second)
            if (Prevlisttime == TrimmedTime):
                fpscountcolor = fpscountcolor + 1
            else:
                ColorfpswithTime[Prevlisttime] = fpscountcolor
                Prevlisttime = TrimmedTime
                fpscountcolor = 1
        # print
        print("Color fps:")
        for k, v in ColorfpswithTime.items():
            print('Time: ' + str(k) + ' , FPS: ' + str(v))

        # Get FPS for IR
        IRfpswithTime = {}
        fpscountir = 0
        Prevlisttime = datetime.time(self.time_listir[0].hour, self.time_listir[0].minute,
                                     self.time_listir[0].second)
        for time in self.time_listir:
            TrimmedTime = datetime.time(time.hour, time.minute, time.second)
            if (Prevlisttime == TrimmedTime):
                fpscountir = fpscountir + 1
            else:
                IRfpswithTime[Prevlisttime] = fpscountir
                Prevlisttime = TrimmedTime
                fpscountir = 1
        # print
        print("IR fps:")
        for k, v in IRfpswithTime.items():
            print('Time: ' + str(k) + ' , FPS: ' + str(v))

        return ColorfpswithTime, IRfpswithTime

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
            filename = filenamearr[8]
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
        total_frames=0
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

                total_frames = total_frames + 1

                frame_count = total_frames
                FPS = self.EstimatedFPS
                td = datetime.timedelta(seconds=(frame_count / FPS))
                self.time_list_color.append(td)

        #print data acquistion time details
        print('Start Time for Color:' + str(self.StartTime))
        print('End Time for Color:' + str(self.EndTime))
        print('Total Time:' + str(self.EndTime - self.StartTime))

        Timecount = 1
        for time in self.time_list_color:
            self.timecolorCount.append(Timecount)
            Timecount = Timecount + 1
        # End
        print('Color ROI loaded..')

    """
    ProcessIRImagestoArray:
    Load IR region of interests, get average of b,g,r and time and distance
    also make sure color and ir data has same x and y values for processing and plotting
    skip first and last seconds 
    """
    def ProcessIRImagestoArray(self, filepath, distnacepath):

        Image_Files = self.LoadFiles(filepath)
        Image_Files = self.SortLoadedFiles(Image_Files)
        fdistancem = open(distnacepath, "r")
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

                # Add Time Stamp
                total_frames = total_frames + 1
                frame_count = total_frames
                FPS = self.EstimatedFPS
                td = datetime.timedelta(seconds=(frame_count / FPS))
                self.time_list_ir.append(td)

        #print time details for ir
        print('Start Time for IR:' + str(self.StartTime))
        print('End Time for IR:' + str(self.EndTime))
        print('Total Time:' + str(self.EndTime - self.StartTime))

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
                    for x in range(0, self.EstimatedFPS):
                        self.distanceM.append(float(dm))

        Timecount = 1
        for time in self.time_list_ir:
            self.timeirCount.append(Timecount)
            Timecount = Timecount + 1
        # End IR
        print('IR ROI loaded..')

        # FIX TODO if not same
        if (len(self.time_list_ir) > len(self.time_list_color)):
            differnce = len(self.time_list_ir) - len(self.time_list_color)
            for i in range(0, differnce):
                self.time_list_ir.pop()
                self.Frametime_list_ir.pop()
                self.timeirCount.pop()
                self.Irchannel.pop()
                self.distanceM.pop()

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
