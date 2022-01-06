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
def Process_SingalData(RunAnalysisForEntireSignalData, ROIStore, SavePath, Algorithm_type, FFT_type,
                       Filter_type, Result_type, Preprocess_type, isSmoothen, HrGr, SpoGr,fileName,DumpToDisk):
    if (not RunAnalysisForEntireSignalData):
        Process_Participants_Data_Windows(ROIStore, SavePath,
                                          Algorithm_type, FFT_type,
                                          HrGr, SpoGr,
                                          Filter_type, Result_type, Preprocess_type, isSmoothen,fileName,DumpToDisk)
    else:
        ListHrdata, ListSPOdata, IsSuccess = Process_Participants_Data_EntireSignal(ROIStore, SavePath,
                                               Algorithm_type, FFT_type,
                                               HrGr, SpoGr,
                                               Filter_type, Result_type, Preprocess_type, isSmoothen,fileName,DumpToDisk)
    # return  ListHrdata, ListSPOdata, IsSuccess



'''
Process participants data in window size over the signal
'''
def Process_Participants_Data_Windows(ROIStore, SavePath,
                                      Algorithm_type, FFT_type, HrGr, SpoGr,
                                      Filter_type, Result_type, Preprocess_type, isSmoothen,fileName,DumpToDisk):

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


    # ROI Window Result list
    WindowRegionList = {}

    # Windows for regions (should be same for all)
    LengthofAllFramesColor = ROIStore.get(objConfig.roiregions[0]).getLengthColor() #len()  # all have same lenghts
    LengthofAllFramesIR = ROIStore.get(objConfig.roiregions[0]).getLengthIR()
    TimeinSeconds = ROIStore.get("lips").totalTimeinSeconds # LengthofAllFrames / objProcessData.ColorEstimatedFPS  # take color as color and ir would run for same window
    # step = 30  # slide window for 1 second or 30 frames.
    #this step will aslo be variable as per that seconds frame rate
    # WindowSlider = step * timeinSeconds  # step * 5 second,  window can hold  150 frames or 5 second data #THIS WORKS ONLY IF FPS IS 30 FOR EVERY SECOND so
    # window slider is calculated in variable frames within the window as it changes
    #TODO: CHECK FOR 5 AND 6 SECOND WINDOW COMPARED TO 10 with results
    # TotalWindows in this sample
    TotalWindows = (TimeinSeconds - timeinSeconds) + 1  # second window gorup
    TotalWindows = round(TotalWindows)
    # Split ground truth data
    HrAvgList = CommonMethods.splitGroundTruth(HrGr, TotalWindows,timeinSeconds)
    SPOAvgList = CommonMethods.splitGroundTruth(SpoGr, TotalWindows,timeinSeconds)

    # Initialise object to process face regions signal data
    objProcessDataLips = ProcessFaceData(Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type,
                                         SavePath ,
                                         objConfig.ignoregray, isSmoothen, objConfig.GenerateGraphs, timeinSeconds,
                                         DumpToDisk)
    # Initialise object to process face regions signal data
    objProcessDataForehead = ProcessFaceData(Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type,
                                         SavePath ,
                                         objConfig.ignoregray, isSmoothen, objConfig.GenerateGraphs, timeinSeconds,
                                         DumpToDisk)

    # Initialise object to process face regions signal data
    objProcessDataRcheek = ProcessFaceData(Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type,
                                         SavePath ,
                                         objConfig.ignoregray, isSmoothen, objConfig.GenerateGraphs, timeinSeconds,
                                         DumpToDisk)

    # Initialise object to process face regions signal data
    objProcessDataLcheek = ProcessFaceData(Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type,
                                         SavePath ,
                                         objConfig.ignoregray, isSmoothen, objConfig.GenerateGraphs, timeinSeconds,
                                         DumpToDisk)
    # Loop through signal data
    for WindowCount in range(0, int(TotalWindows)):#RUNNING FOR WINDOW TIMES # TimeinSeconds
        print('Window: '+ str(WindowCount))
        # Lips
        objProcessDataLips.getSingalDataWindow(ROIStore, objConfig.roiregions[0], WindowCount, TotalWindows,timeinSeconds)
        lipsResult = objProcessDataLips.Process_EntireSignalData(objConfig.RunAnalysisForEntireSignalData)
        if(DumpToDisk):
            objFile.DumpObjecttoDisk(SavePath + fileName + '_WindowsBinaryFiles' + '\\','lipsResult_Window_' + str(WindowCount),lipsResult)

        # Forehead
        objProcessDataForehead.getSingalDataWindow(ROIStore, objConfig.roiregions[1],
                                     WindowCount, TotalWindows,timeinSeconds)  # Lips
        foreheadResult = objProcessDataForehead.Process_EntireSignalData(objConfig.RunAnalysisForEntireSignalData)
        if (DumpToDisk):
            objFile.DumpObjecttoDisk(SavePath + fileName + '_WindowsBinaryFiles' + '\\','foreheadResult_Window_' + str(WindowCount),foreheadResult)

        # LeftCheek
        objProcessDataLcheek.getSingalDataWindow(ROIStore, objConfig.roiregions[2],
                                     WindowCount, TotalWindows,timeinSeconds)
        leftcheekResult = objProcessDataLcheek.Process_EntireSignalData(objConfig.RunAnalysisForEntireSignalData)
        if (DumpToDisk):
            objFile.DumpObjecttoDisk(SavePath + fileName + '_WindowsBinaryFiles' + '\\','leftcheekResult_Window_' + str(WindowCount),leftcheekResult)

        # RightCheek
        objProcessDataRcheek.getSingalDataWindow(ROIStore, objConfig.roiregions[3],
                                     WindowCount, TotalWindows,timeinSeconds)
        rightcheekResult = objProcessDataRcheek.Process_EntireSignalData(objConfig.RunAnalysisForEntireSignalData)
        if (DumpToDisk):
            objFile.DumpObjecttoDisk(SavePath + fileName + '_WindowsBinaryFiles' + '\\','rightcheekResult_Window_' + str(WindowCount),rightcheekResult)

        # Store Data in Window List
        WindowRegionList['lips'] = lipsResult
        WindowRegionList['forehead'] = foreheadResult
        WindowRegionList['rightcheek'] = rightcheekResult
        WindowRegionList['leftcheek'] = leftcheekResult

        # Get best region data
        for k, v in WindowRegionList.items():
            if (v.BestSnR > bestHeartRateSnr):
                bestHeartRateSnr = v.BestSnR
                bestBpm = v.BestBPM
                channeltype = v.channeltype
                regiontype = k
                freqencySamplingError = v.IrFreqencySamplingError
                diffNow = v.diffTime
                diffTimeTotal = v.diffTimeLog

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
                          str(round(heartRateValue)) + " ,\t" + str(difference)+ " ,\t" + str(
                diffNow) + " ,\t" + str(regiontype))

        # Get difference and append data (blood oxygen)
        difference = round(float(SPOAvgList[WindowCount]) - float(oxygenSaturationValue))
        ListSPOdata.append(str(WindowCount) + " ,\t" + str(round(SPOAvgList[WindowCount])) + " ,\t" +
                           str(round(oxygenSaturationValue)) + " ,\t" + str(difference)+ " ,\t" + str(
                diffNow) + " ,\t" + str(regiontype))

        # Next window
        # WindowCount = WindowCount + 1

    # filename
    fileNameHr = "HeartRate_" + fileName
        #          regiontype + "_" + Algorithm_type + "_FFT-" + str(FFT_type) + "_FL-" + str(
        # Filter_type) + "_RS-" + str(Result_type) + "_PR-" + str(Preprocess_type) + "_SM-" + str(
        # isSmoothen)
    # Write data to file
    objFile.WriteListDatatoFile(SavePath + 'Result\\', fileNameHr, ListHrdata)

    # filename
    fileNameSpo = "SPO_" + fileName
    # Algorithm_type + "_FFT-" + str(FFT_type) + "_FL-" + str(
    #     Filter_type) + "_RS-" + str(Result_type)  + "_PR-" + str(Preprocess_type) + "_SM-" + str(
    #     isSmoothen)
    # Write data to file
    objFile.WriteListDatatoFile(SavePath+ 'Result\\', fileNameSpo, ListSPOdata)

    del objReliability
    del objProcessDataLips
    del objProcessDataRcheek
    del objProcessDataForehead
    del objProcessDataLcheek


'''
Process participants data over the entire signal data
'''
def Process_Participants_Data_EntireSignal(ROIStore, SavePath,
                                           Algorithm_type, FFT_type, HrGr, SpoGr,
                                           Filter_type, Result_type, Preprocess_type, isSmoothen):

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
                                     objConfig.ignoregray, isSmoothen, objConfig.GenerateGraphs, timeinSeconds
                                     )

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

