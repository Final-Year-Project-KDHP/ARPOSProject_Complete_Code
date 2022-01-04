import math
import os
import pickle
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

'''
Process participants data 
'''
def Process_SingalData(RunAnalysisForEntireSignalData, ROIStore, SavePath, Algorithm_type, FFT_type, HrGr, SpoGr,
                       Filter_type, Result_type, Preprocess_type, isSmoothen, snrType):
    if (not RunAnalysisForEntireSignalData):
        Process_Participants_Data_Windows(ROIStore, SavePath,
                                          Algorithm_type, FFT_type,
                                          HrGr, SpoGr,
                                          Filter_type, Result_type, Preprocess_type, isSmoothen, snrType)
    else:
        ListHrdata, ListSPOdata, IsSuccess = Process_Participants_Data_EntireSignal(ROIStore, SavePath,
                                               Algorithm_type, FFT_type,
                                               HrGr, SpoGr,
                                               Filter_type, Result_type, Preprocess_type, isSmoothen, snrType)
    return  ListHrdata, ListSPOdata, IsSuccess


def WritetoDisk(location, filename, data):
    ##STORE Data
    with open(location + filename, 'wb') as filehandle:
        pickle.dump(data, filehandle)

'''
Process participants data in window size over the signal
'''
def Process_Participants_Data_Windows(ROIStore, SavePath,
                                      Algorithm_type, FFT_type, HrGr, SpoGr,
                                      Filter_type, Result_type, Preprocess_type, isSmoothen, snrType):

    objReliability = CheckReliability()
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
    timeinSeconds = 10
    finaloxy = 0.0

    # Initialise object to process face regions signal data
    objProcessData = ProcessFaceData(Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type, SavePath,
                                     objConfig.ignoregray, isSmoothen, objConfig.GenerateGraphs,  timeinSeconds,
                                     snrType)
    # ROI Window Result list
    WindowRegionList = {}

    # Windows for regions (should be same for all)
    LengthofAllFrames = len(ROIStore.get(objConfig.roiregions[0]).getAllData())  # all have same lenghts
    TimeinSeconds = ROIStore.get("lips").totalTimeinSeconds # LengthofAllFrames / objProcessData.ColorEstimatedFPS  # take color as color and ir would run for same window
    step = 30  # slide window for 1 second or 30 frames
    WindowSlider = step * timeinSeconds  # step * 5 second,  window can hold  150 frames or 5 second data
    WindowCount = 0

    # TotalWindows in this sample
    TotalWindows = (TimeinSeconds - timeinSeconds) + 1  # second window gorup

    # Split ground truth data
    HrAvgList = CommonMethods.splitGroundTruth(HrGr, TotalWindows,timeinSeconds)
    SPOAvgList = CommonMethods.splitGroundTruth(SpoGr, TotalWindows,timeinSeconds)

    ##Original Data storage
    objWindowProcessedData = WindowProcessedData()
    objWindowProcessedData.HrAvgList = HrAvgList
    objWindowProcessedData.SPOAvgList = SPOAvgList
    objWindowProcessedData.LengthofAllFrames = LengthofAllFrames
    objWindowProcessedData.TimeinSeconds = TimeinSeconds
    objWindowProcessedData.step = step
    objWindowProcessedData.WindowSlider = WindowSlider
    objWindowProcessedData.TotalWindows = TotalWindows
    objWindowProcessedData.ROIStore = ROIStore

    WritetoDisk(SavePath,'objWindowProcessedData_FullWindow',objWindowProcessedData)

    # Loop through signal data
    for j in range(0, int(TimeinSeconds)):
        if LengthofAllFrames >= WindowSlider:  # has atleast enoguth data to process and  all rois have same no of data

            # Lips
            objProcessData.getSingalDataWindow(ROIStore, objConfig.roiregions[0], WindowSlider, step, WindowCount)
            lipsResult = objProcessData.Process_EntireSignalData()

            # Forehead
            objProcessData.getSingalDataWindow(ROIStore, objConfig.roiregions[1], WindowSlider, step,
                                         WindowCount)  # Lips
            foreheadResult = objProcessData.Process_EntireSignalData()

            # LeftCheek
            objProcessData.getSingalDataWindow(ROIStore, objConfig.roiregions[2], WindowSlider, step,
                                         WindowCount)
            leftcheekResult = objProcessData.Process_EntireSignalData()

            # RightCheek
            objProcessData.getSingalDataWindow(ROIStore, objConfig.roiregions[3], WindowSlider, step,
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
            heartRateValue, heartRateError = objReliability.AcceptorRejectHR(bestHeartRateSnr, bestBpm,
                                                                             freqencySamplingError)
            oxygenSaturationValue, oxygenSaturationValueError = objReliability.AcceptorRejectSPO(smallestOxygenError,
                                                                                                 finaloxy)

            # Get difference and append data (heart rate)
            difference = round(float(HrAvgList[WindowCount]) - float(heartRateValue))
            ListHrdata.append(str(WindowCount) + " ,\t" + str(round(HrAvgList[WindowCount])) + " ,\t" +
                              str(round(heartRateValue)) + " ,\t" + str(difference))

            # Get difference and append data (blood oxygen)
            difference = round(float(SPOAvgList[WindowCount]) - float(oxygenSaturationValue))
            ListSPOdata.append(str(WindowCount) + " ,\t" + str(round(SPOAvgList[WindowCount])) + " ,\t" +
                               str(round(oxygenSaturationValue)) + " ,\t" + str(difference))

            # Next window
            WindowCount = WindowCount + 1
        else:
            break

    # filename
    fileNameHr = "HRdata_" + regiontype + "_" + Algorithm_type + "_FFT-" + str(FFT_type) + "_FL-" + str(
        Filter_type) + "_RS-" + str(Result_type) + "_PR-" + str(Preprocess_type) + "_SM-" + str(
        isSmoothen)
    # Write data to file
    objFile.WriteListDatatoFile(SavePath, fileNameHr, ListHrdata)

    # filename
    fileNameSpo = "SPOdata_" + Algorithm_type + "_FFT-" + str(FFT_type) + "_FL-" + str(
        Filter_type) + "_RS-" + str(Result_type)  + "_PR-" + str(Preprocess_type) + "_SM-" + str(
        isSmoothen)
    # Write data to file
    objFile.WriteListDatatoFile(SavePath, fileNameSpo, ListSPOdata)

    del objReliability
    del objProcessData


'''
Process participants data over the entire signal data
'''
def Process_Participants_Data_EntireSignal(ROIStore, SavePath,
                                           Algorithm_type, FFT_type, HrGr, SpoGr,
                                           Filter_type, Result_type, Preprocess_type, isSmoothen, snrType):

    # Lists to hold heart rate and blood oxygen data
    ListHrdata = []
    ListSPOdata = []

    objReliability = CheckReliability()

    diffNow = 0.0
    bestHeartRateSnr = 0.0
    bestBpm = 0.0
    channeltype = ''
    regiontype = ''
    IsSuccess = ''
    finaloxy = 0.0
    freqencySamplingError = 0.0
    smallestOxygenError = sys.float_info.max
    timeinSeconds = ROIStore.get("lips").totalTimeinSeconds  # this is same for all regions as no of frames are same recorded for that time peroid so we can use any region to get time
    # leng = len(ROIStore.get("lips").Irchannel)

    # Initialise object to process face regions signal data
    objProcessData = ProcessFaceData(Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type, SavePath,
                                     objConfig.ignoregray, isSmoothen, objConfig.GenerateGraphs, timeinSeconds,
                                     snrType)

    # ROI Window Result list
    WindowRegionList = {}

    # Average ground truth data
    HrAvegrage = (np.average(CommonMethods.AvegrageGroundTruth(HrGr)))
    SPOAvegrage = (np.average(CommonMethods.AvegrageGroundTruth(SpoGr)))

    # Lips
    objProcessData.getSingalData(ROIStore, objConfig.roiregions[0])
    lipsResult = objProcessData.Process_EntireSignalData()

    # Forehead
    objProcessData.getSingalData(ROIStore, objConfig.roiregions[1])
    foreheadResult = objProcessData.Process_EntireSignalData()

    # LeftCheek
    objProcessData.getSingalData(ROIStore, objConfig.roiregions[2])
    leftcheekResult = objProcessData.Process_EntireSignalData()

    # RightCheek
    objProcessData.getSingalData(ROIStore, objConfig.roiregions[3])
    rightcheekResult = objProcessData.Process_EntireSignalData()

    # Store Data in Window List
    WindowRegionList['lips'] = lipsResult
    WindowRegionList['forehead'] = foreheadResult
    WindowRegionList['rightcheek'] = rightcheekResult
    WindowRegionList['leftcheek'] = leftcheekResult
    diffTimeTotal = {}

    # Get best region data
    for k, v in WindowRegionList.items():
        if(v.BestSnR > bestHeartRateSnr):
            bestHeartRateSnr = v.BestSnR
            bestBpm = v.BestBPM
            # channeltype = v.
            regiontype = k
            # freqencySamplingError = v.IrFreqencySamplingError
            diffNow = v.diffTime
            diffTimeTotal = v.diffTimeLog
        # if (v.IrSnr > bestHeartRateSnr):
        #     bestHeartRateSnr = v.IrSnr
        #     bestBpm = v.IrBpm
        #     channeltype = 'IR'
        #     regiontype = k
        #     freqencySamplingError = v.IrFreqencySamplingError
        #     diffNow = v.diffTime
        #     diffTimeTotal = v.diffTimeLog
        #
        # if (v.GreySnr > bestHeartRateSnr):
        #     bestHeartRateSnr = v.GreySnr
        #     bestBpm = v.GreyBpm
        #     channeltype = 'Grey'
        #     regiontype = k
        #     freqencySamplingError = v.GreyFreqencySamplingError
        #     diffNow = v.diffTime
        #     diffTimeTotal = v.diffTimeLog
        #
        # if (v.RedSnr > bestHeartRateSnr):
        #     bestHeartRateSnr = v.RedSnr
        #     bestBpm = v.RedBpm
        #     channeltype = 'Red'
        #     regiontype = k
        #     freqencySamplingError = v.RedFreqencySamplingError
        #     diffNow = v.diffTime
        #     diffTimeTotal = v.diffTimeLog
        #
        # if (v.GreenSnr > bestHeartRateSnr):
        #     bestHeartRateSnr = v.GreenSnr
        #     bestBpm = v.GreenBpm
        #     channeltype = 'Green'
        #     regiontype = k
        #     freqencySamplingError = v.GreenFreqencySamplingError
        #     diffNow = v.diffTime
        #     diffTimeTotal = v.diffTimeLog
        #
        # if (v.BlueSnr > bestHeartRateSnr):
        #     bestHeartRateSnr = v.BlueSnr
        #     bestBpm = v.BlueBpm
        #     channeltype = 'Blue'
        #     regiontype = k
        #     freqencySamplingError = v.BlueFreqencySamplingError
        #     diffNow = v.diffTime
        #     diffTimeTotal = v.diffTimeLog

        if (v.oxygenSaturationValueError < smallestOxygenError):
            smallestOxygenError = v.oxygenSaturationValueError
            finaloxy = v.oxygenSaturationValueValue

    if (regiontype != ''):
        IsSuccess = True
        # Check reliability and record best readings
        heartRateValue, heartRateError = objReliability.AcceptorRejectHR(bestHeartRateSnr, bestBpm, freqencySamplingError)
        oxygenSaturationValue, oxygenSaturationValueError = objReliability.AcceptorRejectSPO(smallestOxygenError, finaloxy)

        # Get difference and append data (heart rate)
        difference = round(float(HrAvegrage) - float(round(heartRateValue)))
        ListHrdata.append(str(round(HrAvegrage)) + " ,\t" + str(round(heartRateValue)) + " ,\t" + str(difference)+ " ,\t" + str(diffNow)+ " ,\t" + str(regiontype))

        # Get difference and append data (blood oxygen)
        difference = round(float(SPOAvegrage) - float(oxygenSaturationValue))
        ListSPOdata.append(str(round(SPOAvegrage)) + " ,\t" + str(round(oxygenSaturationValue)) + " ,\t" + str(difference) + str(diffNow)+ " ,\t" + str(regiontype))

        printTime = ''
        for k, v in diffTimeTotal.items():
            printTime = printTime + str(k) + ': ' + str(v) + '\n'

        objFile = FileIO()
        objFile.WritedatatoFile(SavePath,'timeLogFile',printTime)
        del objFile

    # # Write data to file
    # objFile.WriteListDatatoFile(SavePath, fileNameHr, ListHrdata)
    # objFile.WriteListDatatoFile(SavePath, fileNameSpo, ListSPOdata)
    else:
        IsSuccess = False

    del objReliability
    del objProcessData

    return ListHrdata, ListSPOdata, IsSuccess



'''
Process participants data over the entire signal data
'''
def Process_Participants_Data_GetBestHR(objresultProcessedDataLips,objresultProcessedDataForehead,objresultProcessedDataRcheek,objresultProcessedDataLCheek, SavePath,HrGr,filename):
    ##Copy obj compute HR data to local class
    # objresultProcessedData.IrSnr = objComputerHeartRate.IrSnr
    # objresultProcessedData.GreySnr = objComputerHeartRate.GreySnr
    # objresultProcessedData.RedSnr = objComputerHeartRate.RedSnr
    # objresultProcessedData.GreenSnr = objComputerHeartRate.GreenSnr
    # objresultProcessedData.BlueSnr = objComputerHeartRate.BlueSnr
    # objresultProcessedData.IrBpm = objComputerHeartRate.IrBpm
    # objresultProcessedData.GreyBpm = objComputerHeartRate.GreyBpm
    # objresultProcessedData.RedBpm = objComputerHeartRate.RedBpm
    # objresultProcessedData.GreenBpm = objComputerHeartRate.GreenBpm
    # objresultProcessedData.BlueBpm = objComputerHeartRate.BlueBpm
    # objresultProcessedData.IrFreqencySamplingError = objComputerHeartRate.IrFreqencySamplingError
    # objresultProcessedData.GreyFreqencySamplingError = objComputerHeartRate.GreyFreqencySamplingError
    # objresultProcessedData.RedFreqencySamplingError = objComputerHeartRate.RedFreqencySamplingError
    # objresultProcessedData.GreenFreqencySamplingError = objComputerHeartRate.GreenFreqencySamplingError
    # objresultProcessedData.BlueFreqencySamplingError = objComputerHeartRate.BlueFreqencySamplingError
    # objresultProcessedData.startTime = startTime
    # objresultProcessedData.endTime = endTime
    # objresultProcessedData.diffTime = diffTime
    # objresultProcessedData.blue_fft_realabs = blue_fft_realabs
    # objresultProcessedData.green_fft_realabs = green_fft_realabs
    # objresultProcessedData.red_fft_realabs = red_fft_realabs
    # objresultProcessedData.grey_fft_realabs = grey_fft_realabs
    # objresultProcessedData.ir_fft_realabs = ir_fft_realabs
    # Lists to hold heart rate and blood oxygen data
    ListHrdata = []

    objReliability = CheckReliability()

    diffNow = 0.0
    regiontype = ''
    freqencySamplingError = 0.0

    # ROI Window Result list
    WindowRegionList = {}

    # Average ground truth data
    HrAvegrage = (np.average(CommonMethods.AvegrageGroundTruth(HrGr)))

    # Store Data in Window List
    WindowRegionList['lips'] = objresultProcessedDataLips
    WindowRegionList['forehead'] = objresultProcessedDataForehead
    WindowRegionList['rightcheek'] = objresultProcessedDataRcheek
    WindowRegionList['leftcheek'] = objresultProcessedDataLCheek

    # get best bpm and heart rate period in one region
    bestHeartRateSnr = 0.0
    bestBpm = 0.0

    # Get best region data
    for k, v in WindowRegionList.items():
        if (v.IrSnr > bestHeartRateSnr):
            bestHeartRateSnr = v.IrSnr
            bestBpm = v.IrBpm
            freqencySamplingError= v.IrFreqencySamplingError

        if (v.GreySnr > bestHeartRateSnr):
            bestHeartRateSnr = v.GreySnr
            bestBpm = v.GreyBpm
            freqencySamplingError= v.GreyFreqencySamplingError

        if (v.RedSnr > bestHeartRateSnr):
            bestHeartRateSnr = v.RedSnr
            bestBpm = v.RedBpm
            freqencySamplingError= v.RedFreqencySamplingError

        if (v.GreenSnr > bestHeartRateSnr):
            bestHeartRateSnr = v.GreenSnr
            bestBpm = v.GreenBpm
            freqencySamplingError= v.GreenFreqencySamplingError

        if (v.BlueSnr > bestHeartRateSnr):
            bestHeartRateSnr = v.BlueSnr
            bestBpm = v.BlueBpm
            freqencySamplingError= v.BlueFreqencySamplingError

    # Check reliability and record best readings
    heartRateValue, heartRateError = objReliability.AcceptorRejectHR(bestHeartRateSnr, bestBpm, freqencySamplingError)

    # Get difference and append data (heart rate)
    difference = round(float(HrAvegrage) - float(heartRateValue))
    ListHrdata.append(str(round(HrAvegrage)) + " ,\t" + str(round(heartRateValue)) + " ,\t" + str(difference)+ " ,\t" + str(diffNow)+ " ,\t" + str(regiontype))

    # printTime = str(start) + ': ' + str(v) + '\n'

    objFile = FileIO()
    # objFile.WritedatatoFile(SavePath,'timeLogFile',printTime)
    objFile.WriteListDatatoFile(SavePath,filename,ListHrdata)
    del objFile

    del objReliability

    # return ListHrdata

class WindowProcessedData:
    HrAvgList = None
    SPOAvgList = None
    LengthofAllFrames= None
    TimeinSeconds= None
    step= None
    WindowSlider = None
    TotalWindows = None
    ROIStore= None
