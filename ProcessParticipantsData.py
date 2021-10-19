import os
import sys
import numpy as np
import CommonMethods
from CheckReliability import CheckReliability
from Configurations import Configurations
from FileIO import FileIO
from ProcessData import ProcessFaceData

# Global Objects
objConfig = Configurations()
objFile = FileIO()
objReliability = CheckReliability()
objConfig = Configurations()


def Process_Participants_Data_Windows(ROIStore, SavePath, participant_number, position,
                                      Algorithm_type, FFT_type, HrGr, SpoGr,
                                      Filter_type, Result_type, Preprocess_type, isSmoothen,HrType):

    # Initialise object to process face regions signal data
    objProcessData = ProcessFaceData(Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type, SavePath,
                                     objConfig.ignoregray, isSmoothen, objConfig.GenerateGraphs,HrType)

    # Lists to hold heart rate and blood oxygen data
    ListHrdata = []
    ListSPOdata = []
    bestHeartRateSnr = 0.0
    bestBpm = 0.0
    channeltype = ''
    regiontype = ''
    freqencySamplingError = 0.0
    previousComputedHeartRate = 0.0
    smallestOxygenError = sys.float_info.max

    # ROI Window Result list
    WindowRegionList = {}

    # Windows for regions (should be same for all)
    LengthofAllFrames = len(ROIStore.get(objConfig.roiregions[0]).getAllData())  # all have same lenghts
    TimeinSeconds = LengthofAllFrames / objProcessData.EstimatedFPS  # FIX TODO change for differnt fps
    step = 30  # slide window for 1 second or 30 frames
    timeinSeconds = 5
    WindowSlider = step * timeinSeconds  # step * 5 second,  window can hold  150 frames or 5 second data
    WindowCount = 0

    # TotalWindows in this sample
    TotalWindows = (TimeinSeconds - 5) + 1  # 5 second window gorup

    # Split ground truth data
    HrAvgList = CommonMethods.splitGroundTruth(HrGr, TotalWindows)

    # Loop through signal data
    for j in range(0, int(TimeinSeconds)):
        if LengthofAllFrames >= WindowSlider:  # has atleast enoguth data to process and  all rois have same no of data
            # Set default Heart Rate
            grHr = 60
            if (WindowCount) < len(HrAvgList):
                grHr = HrAvgList[WindowCount]

            # Lips
            objProcessData.getSingalData(ROIStore, objConfig.roiregions[0], WindowSlider, step, WindowCount)
            lipsResult = objProcessData.Process_EntireSignalData()

            # Forehead
            objProcessData.getSingalData(ROIStore, objConfig.roiregions[1], WindowSlider, step,
                                         WindowCount)  # Lips
            foreheadResult = objProcessData.Process_EntireSignalData()

            # LeftCheek
            objProcessData.getSingalData(ROIStore, objConfig.roiregions[2], WindowSlider, step,
                                         WindowCount)
            leftcheekResult = objProcessData.Process_EntireSignalData()

            # RightCheek
            objProcessData.getSingalData(ROIStore, objConfig.roiregions[3], WindowSlider, step,
                                         WindowCount)
            rightcheekResult = objProcessData.Process_EntireSignalData()

            # Store Data in Window List
            WindowRegionList['lips'] = lipsResult
            WindowRegionList['forehead'] = foreheadResult
            WindowRegionList['rightcheek'] = rightcheekResult
            WindowRegionList['leftcheek'] = leftcheekResult

            # Get best region data
            for k, v in WindowRegionList.items():
                if (v.IrSnr > bestHeartRateSnr):
                    bestHeartRateSnr = v.IrSnr
                    bestBpm = v.IrBpm
                    channeltype = 'IR'
                    regiontype = k
                    freqencySamplingError = v.IrFreqencySamplingError

                if (v.GreySnr > bestHeartRateSnr):
                    bestHeartRateSnr = v.GreySnr
                    bestBpm = v.GreyBpm
                    channeltype = 'Grey'
                    regiontype = k
                    freqencySamplingError = v.GreyFreqencySamplingError

                if (v.RedSnr > bestHeartRateSnr):
                    bestHeartRateSnr = v.RedSnr
                    bestBpm = v.RedBpm
                    channeltype = 'Red'
                    regiontype = k
                    freqencySamplingError = v.RedFreqencySamplingError

                if (v.GreenSnr > bestHeartRateSnr):
                    bestHeartRateSnr = v.GreenSnr
                    bestBpm = v.GreenBpm
                    channeltype = 'Green'
                    regiontype = k
                    freqencySamplingError = v.GreenFreqencySamplingError

                if (v.BlueSnr > bestHeartRateSnr):
                    bestHeartRateSnr = v.BlueSnr
                    bestBpm = v.BlueBpm
                    channeltype = 'Blue'
                    regiontype = k
                    freqencySamplingError = v.BlueFreqencySamplingError

                if (v.oxygenSaturationValueError < smallestOxygenError):
                    smallestOxygenError = v.oxygenSaturationValueError
                    finaloxy = v.oxygenSaturationValueValue

            # Check reliability and record best readings
            previousHR, numberOfAnalysisFailureshr, HRValue, HRError = objReliability.AddHeartRate(bestHeartRateSnr,
                                                                                                   bestBpm)
            objProcessData.previousComputedHeartRate = previousHR
            objProcessData.numberOfAnalysisFailuresSinceCorrectHeartRate = numberOfAnalysisFailureshr

            # Get difference and append data
            difference = round(float(HrAvgList[WindowCount]) - float(bestBpm))
            ListHrdata.append(str(WindowCount) + " ,\t" + str(round(HrAvgList[WindowCount])) + " ,\t" +
                              str(round(bestBpm)) + " ,\t" + str(difference))
            # Next window
            WindowCount = WindowCount + 1
        else:
            break

    #filename
    fileName = "HRdata_" + regiontype + "_" + Algorithm_type + "_FFT-" + str(FFT_type) + "_FL-" + str(
            Filter_type) + "_RS-" + str(Result_type) + "_HR-" + str(HrType) + "_PR-" + str(Preprocess_type) + "_SM-" + str(
            isSmoothen)
    # Write data to file
    objFile.WriteListDatatoFile(SavePath, fileName, ListHrdata)
