"""
Common Methods:
These can be used in any class or file.
"""
from datetime import datetime

import cv2
import numpy as np

"""
GetGroundTruth:
get ground truth data of a participant by heart rate status (resting or active)
"""
def GetGroundTruth(participant_number, position,diskpath):
    filepath = diskpath + "\\GroundTruthData" + "\\" + participant_number + "\\" + position + "\\"
    HrFile =  filepath + "HR.txt"
    SPOFile =  filepath + "SPO.txt"

    #read data from files
    HrFiledata = open(HrFile, "r")
    SPOFiledata = open(SPOFile, "r")

    HrGr =HrFiledata.read().split("\n")
    SpoGr =SPOFiledata.read().split("\n")

    HrFiledata.close()
    SPOFiledata.close()

    HrGr = [float(value) for value in HrGr]
    SpoGr = [float(value) for value in SpoGr]

    return HrGr, SpoGr


def splitGroundTruth(groundtruth,TotalWindows):
    initialEndindex =5
    initialindex = 0
    AvgValue = 0
    HrAvgList = []
    TotalWindows= round(TotalWindows)
    groundtruth = np.array(groundtruth)
    for j in range(0, TotalWindows):
        #initialEndindex = initialEndindex + 5
        initialEndindex =5 +j
        for x in range(j, initialEndindex):
            if(x< len(groundtruth)):
                #print(HrGr[x])
                AvgValue = AvgValue + int(groundtruth[x])

        AvgValue = AvgValue / 5
        HrAvgList.append(round(AvgValue))
        AvgValue=0

    return HrAvgList


def AvegrageGroundTruth(groundtruth):
    AvgValue = 0
    groundtruth = np.array(groundtruth)
    AvgValue = round(np.average(groundtruth))

    return AvgValue