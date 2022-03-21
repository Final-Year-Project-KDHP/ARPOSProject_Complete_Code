import os
import sys
from datetime import datetime

import CommonMethods
from CalculateAndStoreOnDisk.ProcessDataPartial import ProcessDataPartial
from CheckReliability import CheckReliability
from Configurations import Configurations
from FileIO import FileIO
from LoadFaceData import LoadFaceData
from GlobalDataFile import GlobalData
import pickle

from ProcessData import ProcessFaceData
from ProcessParticipantsData import Process_Participants_Data_GetBestHR
from ProcessedParticipantsData import ProcessedParticipantsData
from WindowData import Window_Data, LogItems

'''
InitiateProcessingStorage:
'''


class InitiateProcessingStorage:
    # Global Objects
    objConfig = None
    ProcessedDataPath = None
    objFile = None

    # Constructor
    def __init__(self,objConfig):
        self.objConfig = objConfig
        self.objFile = FileIO()

    def WritetoDisk(self, location, filename, data):
        ##STORE Data
        with open(location + filename, 'wb') as filehandle:
            pickle.dump(data, filehandle)

    def ReadfromDisk(self, location, filename):
        ##Read data
        with open(location + filename, 'rb') as filehandle:
            data = pickle.load(filehandle)
        return data

    def GenerateOriginalRawData(self, participant_number, position, objFaceImage,
                                region):
        ##get loading path
        LoadColordataPath, LoadIRdataPath, LoadDistancePath, self.ProcessedDataPath = self.objConfig.getLoadPath(
            participant_number,
            position,
            region)
        # Load Roi data (ALL)
        # print("Loading and processing color roi data")
        objFaceImage.ProcessColorImagestoArray(LoadColordataPath)

        # print("Loading and processing ir roi data")
        objFaceImage.ProcessIRImagestoArray(LoadIRdataPath)

        # GET FPS and distance and other data
        ColorfpswithTime, IRfpswithTime, isVariable, isIRVariable, ColorFPS, IRFPS = objFaceImage.GetEstimatedFPS(
            LoadDistancePath)

        # Create global data object and use dictionary (ROI Store) to uniquely store a regions data
        # Store data to disk
        self.WritetoDisk(self.ProcessedDataPath, 'objFaceImage_' + region,
                         objFaceImage)  # self.WritetoDisk(ProcessedDataPath,'redChannel',objFaceImage.red)
        self.GenerateGraph(objFaceImage, self.ProcessedDataPath, region)

    def ReadOriginalRawData(self, region):
        # Load from disk
        objFaceImage = self.ReadfromDisk(self.ProcessedDataPath, 'objFaceImage_' + region)
        return objFaceImage

    def ReadData(self, name):
        # Load from disk
        dataFile = self.ReadfromDisk(self.ProcessedDataPath, name)
        return dataFile

    def getProcessParameters(self,processingStep, process_type, dataObject=None):

        Algorithm_type = 'None'
        FFT_type = 'None'
        Filter_type = 0
        Result_type = 0
        Preprocess_type = 0
        isSmoothen = None

        if(dataObject != None):
            Algorithm_type = dataObject.Algorithm_type
            FFT_type = dataObject.FFT_type
            Filter_type = dataObject.Filter_type
            Result_type = dataObject.Result_type
            Preprocess_type = dataObject.Preprocess_type
            isSmoothen = dataObject.isSmoothen

        if (processingStep == "PreProcess"):
            Preprocess_type = process_type
        elif (processingStep == "Algorithm"):
            Algorithm_type = process_type
        elif (processingStep == "Smoothen"):
            isSmoothen = process_type
        elif (processingStep == "FFT"):
            FFT_type = process_type
        elif (processingStep == "Filter"):
            Filter_type = process_type
        elif (processingStep == "ComputerHRandSPO"):
            Result_type = process_type

        return Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type, isSmoothen


    '''
    ProcessDatainSteps: 
    '''
    def PreProcessDataWindow(self, dataObject, objConfig, processingStep,process_type, SavePath, prevfileName,currentSaveFileName,timeinSeconds):
        objFile = FileIO()
        WindowRegionList = {} # ROI Window Result list
        minFPSIRvalue = min(dataObject.ROIStore.get(objConfig.roiregions[0]).IRfpswithTime.values())
        maxFPSIRvalue = max(dataObject.ROIStore.get(objConfig.roiregions[0]).IRfpswithTime.values())
        minFPSColorvalue = min(dataObject.ROIStore.get(objConfig.roiregions[0]).ColorfpswithTime.values())
        maxFPSColorvalue = max(dataObject.ROIStore.get(objConfig.roiregions[0]).ColorfpswithTime.values())

        FPSNotes = 'min IRvalue: ' + str(minFPSIRvalue) + ', max IRvalue: ' + str(
            maxFPSIRvalue) + ', min Colorvalue: ' + str(minFPSColorvalue) + ', max Colorvalue: ' + str(maxFPSColorvalue)

        # Windows for regions (should be same for all)
        LengthofAllFramesColor = dataObject.ROIStore.get(objConfig.roiregions[0]).getLengthColor()
        LengthofAllFramesIR = dataObject.ROIStore.get(objConfig.roiregions[0]).getLengthIR()

        totalTimeinSeconds = dataObject.ROIStore.get("lips").totalTimeinSeconds
        HrGr = dataObject.HrGroundTruthList[:totalTimeinSeconds]
        SpoGr = dataObject.SPOGroundTruthList[:totalTimeinSeconds]

        TimeinSeconds = totalTimeinSeconds
        # timeinSeconds = 4 #reason to chose as herat rate for paricipants takes attleast 4 second to change so ground truth is closer to window data genreated

        TotalWindows = (TimeinSeconds - timeinSeconds) + 1  # second window gorup
        TotalWindows = round(TotalWindows)

        # Split ground truth data
        HrAvgList = CommonMethods.splitGroundTruth(HrGr, TotalWindows, timeinSeconds)
        SPOAvgList = CommonMethods.splitGroundTruth(SpoGr, TotalWindows, timeinSeconds)

        # for entire signal
        # TotalWindows = 1
        # timeinSeconds = round(TimeinSeconds)
        # HrAvg = CommonMethods.AvegrageGroundTruth(HrGr)
        # SPOAvg = CommonMethods.AvegrageGroundTruth(SpoGr)

        Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type, isSmoothen = self.getProcessParameters(processingStep, process_type)

        Gy_filtered = None

        # Initialise object to process face regions signal data
        objProcessDataLips = ProcessFaceData(Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type,
                                         SavePath, objConfig.ignoregray, isSmoothen, objConfig.GenerateGraphs,
                                         timeinSeconds,
                                         self.objConfig.DumpToDisk, currentSaveFileName)
        # Initialise object to process face regions signal data
        objProcessDataForehead = ProcessFaceData(Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type,
                                         SavePath, objConfig.ignoregray, isSmoothen, objConfig.GenerateGraphs,
                                         timeinSeconds,
                                         self.objConfig.DumpToDisk, currentSaveFileName)
        # Initialise object to process face regions signal data
        objProcessDataLeftCheek = ProcessFaceData(Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type,
                                         SavePath, objConfig.ignoregray, isSmoothen, objConfig.GenerateGraphs,
                                         timeinSeconds,
                                         self.objConfig.DumpToDisk, currentSaveFileName)
        # Initialise object to process face regions signal data
        objProcessDataRightCheek = ProcessFaceData(Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type,
                                         SavePath, objConfig.ignoregray, isSmoothen, objConfig.GenerateGraphs,
                                         timeinSeconds,
                                         self.objConfig.DumpToDisk, currentSaveFileName)
        # Initialise object to process face regions signal data
        objProcessDataCheeksCombined = ProcessFaceData(Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type,
                                         SavePath, objConfig.ignoregray, isSmoothen, objConfig.GenerateGraphs,
                                         timeinSeconds,
                                         self.objConfig.DumpToDisk, currentSaveFileName)

        for WindowCount in range(0, int(TotalWindows)):  # RUNNING FOR WINDOW TIMES # TimeinSeconds

            objProcessedParticipantsData = ProcessedParticipantsData(minFPSIRvalue, maxFPSIRvalue, minFPSColorvalue,
                                                                     maxFPSColorvalue, FPSNotes,
                                                                     LengthofAllFramesColor, LengthofAllFramesIR,
                                                                     totalTimeinSeconds, TimeinSeconds, timeinSeconds,
                                                                     TotalWindows, HrAvgList, SPOAvgList,
                                                                     Algorithm_type, FFT_type, Filter_type, Result_type,
                                                                     Preprocess_type,
                                                                     isSmoothen)

            #objProcessDataLips process data
            if(SavePath.__contains__("SameLength")):
                objProcessDataLips.getSingalDataWindowSameLenght(dataObject.ROIStore, objConfig.roiregions[0], WindowCount, TotalWindows, timeinSeconds)
            else:
                objProcessDataLips.getSingalDataWindow(dataObject.ROIStore, objConfig.roiregions[0], WindowCount, TotalWindows, timeinSeconds)

            objProcessDataLips.ProcessStep(processingStep)

            # objProcessDataForehead process data
            if (SavePath.__contains__("SameLength")):
                objProcessDataForehead.getSingalDataWindowSameLenght(dataObject.ROIStore, objConfig.roiregions[1], WindowCount,
                                                           TotalWindows, timeinSeconds)
            else:
                objProcessDataForehead.getSingalDataWindow(dataObject.ROIStore, objConfig.roiregions[1], WindowCount,
                                                           TotalWindows, timeinSeconds)
            objProcessDataForehead.ProcessStep(processingStep)

            # objProcessDataLeftCheek process data
            if (SavePath.__contains__("SameLength")):
                objProcessDataLeftCheek.getSingalDataWindowSameLenght(dataObject.ROIStore, objConfig.roiregions[2], WindowCount,
                                                            TotalWindows, timeinSeconds)
            else:
                objProcessDataLeftCheek.getSingalDataWindow(dataObject.ROIStore, objConfig.roiregions[2], WindowCount,
                                                            TotalWindows, timeinSeconds)
            objProcessDataLeftCheek.ProcessStep(processingStep)

            # objProcessDataRightCheek process data
            if (SavePath.__contains__("SameLength")):
                objProcessDataRightCheek.getSingalDataWindowSameLenght(dataObject.ROIStore, objConfig.roiregions[3], WindowCount,
                                                             TotalWindows, timeinSeconds)
            else:
                objProcessDataRightCheek.getSingalDataWindow(dataObject.ROIStore, objConfig.roiregions[3], WindowCount,
                                                             TotalWindows, timeinSeconds)
            objProcessDataRightCheek.ProcessStep(processingStep)

            # objProcessDataCheeksCombined process data
            if (SavePath.__contains__("SameLength")):
                objProcessDataCheeksCombined.getSingalDataWindowSameLenght(dataObject.ROIStore, objConfig.roiregions[4],
                                                                 WindowCount, TotalWindows, timeinSeconds)
            else:
                objProcessDataCheeksCombined.getSingalDataWindow(dataObject.ROIStore, objConfig.roiregions[4],
                                                                 WindowCount, TotalWindows, timeinSeconds)
            objProcessDataCheeksCombined.ProcessStep(processingStep)

            WindowRegionList[objConfig.roiregions[0]] = objProcessDataLips
            WindowRegionList[objConfig.roiregions[1]] = objProcessDataForehead
            WindowRegionList[objConfig.roiregions[2]] = objProcessDataLeftCheek
            WindowRegionList[objConfig.roiregions[3]] = objProcessDataRightCheek
            WindowRegionList[objConfig.roiregions[4]] = objProcessDataCheeksCombined

            objProcessedParticipantsData.WindowRegionList = WindowRegionList

            if (self.objConfig.DumpToDisk):
                objFile.DumpObjecttoDisk(SavePath + currentSaveFileName + '\\' ,
                                        "ProcessedWindow_" + str(WindowCount), objProcessedParticipantsData)
                # objFile.DumpObjecttoDisk(SavePath + currentSaveFileName + '\\' + objConfig.roiregions[1] + "\\",
                #                         "ProcessedWindow_" + str(WindowCount), objProcessDataForehead)
                # objFile.DumpObjecttoDisk(SavePath + currentSaveFileName + '\\' + objConfig.roiregions[2] + "\\",
                #                         "ProcessedWindow_" + str(WindowCount), objProcessDataLeftCheek)
                # objFile.DumpObjecttoDisk(SavePath + currentSaveFileName + '\\' + objConfig.roiregions[3] + "\\",
                #                         "ProcessedWindow_" + str(WindowCount), objProcessDataRightCheek)
                # objFile.DumpObjecttoDisk(SavePath + currentSaveFileName + '\\' + objConfig.roiregions[4] + "\\",
                #                         "ProcessedWindow_" + str(WindowCount), objProcessDataCheeksCombined)

            del objProcessedParticipantsData

        del objProcessDataLips
        del objProcessDataForehead
        del objProcessDataLeftCheek
        del objProcessDataRightCheek
        del objProcessDataCheeksCombined

        objFile.WritedatatoFile(SavePath + currentSaveFileName + '\\' ,
                                        "ProcessedCompleted", "Completed")
        del objFile

    def ProcessDatainStepsWindow(self, dataObject, objConfig, processingStep,process_type, SavePath, prevfileName,currentSaveFileName):
        objFile = FileIO()

        WindowRegionList = {} # ROI Window Result list

        Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type, isSmoothen = self.getProcessParameters(processingStep, process_type, dataObject)

        # for region in self.objConfig.roiregions:
        # print(prevfileName)
        WindowCountList = prevfileName.split("_")
        WindowCount = int(WindowCountList[1])

        # Initialise object to process face regions signal data
        objProcessDataLips = dataObject.WindowRegionList[objConfig.roiregions[0]]
        objProcessDataLips.Algorithm_type =Algorithm_type
        objProcessDataLips.FFT_type =FFT_type
        objProcessDataLips.Filter_type =Filter_type
        objProcessDataLips.Result_type =Result_type
        objProcessDataLips.Preprocess_type =Preprocess_type
        objProcessDataLips.SavePath =SavePath
        objProcessDataLips.ignoregray =objConfig.ignoregray
        objProcessDataLips.isSmoothen =isSmoothen
        objProcessDataLips.GenerateGraphs =objConfig.GenerateGraphs
        objProcessDataLips.fileName =currentSaveFileName
        objProcessDataLips.DumpToDisk = self.objConfig.DumpToDisk
        objProcessDataLips.objAlgorithm.ColorEstimatedFPS = objProcessDataLips.ColorEstimatedFPS
        objProcessDataLips.objAlgorithm.IREstimatedFPS = objProcessDataLips.IREstimatedFPS
        objProcessDataLips.objPlots.ColorEstimatedFPS = objProcessDataLips.ColorEstimatedFPS
        objProcessDataLips.objPlots.IREstimatedFPS = objProcessDataLips.IREstimatedFPS
        objProcessDataLips.Window_count =WindowCount
        objProcessDataLips.region = objConfig.roiregions[0]

        # Initialise object to process face regions signal data
        objProcessDataForehead =dataObject.WindowRegionList[objConfig.roiregions[1]]
        objProcessDataForehead.Algorithm_type = Algorithm_type
        objProcessDataForehead.FFT_type = FFT_type
        objProcessDataForehead.Filter_type = Filter_type
        objProcessDataForehead.Result_type = Result_type
        objProcessDataForehead.Preprocess_type = Preprocess_type
        objProcessDataForehead.SavePath = SavePath
        objProcessDataForehead.ignoregray = objConfig.ignoregray
        objProcessDataForehead.isSmoothen = isSmoothen
        objProcessDataForehead.GenerateGraphs = objConfig.GenerateGraphs
        objProcessDataForehead.fileName = currentSaveFileName
        objProcessDataForehead.DumpToDisk = self.objConfig.DumpToDisk
        objProcessDataForehead.objAlgorithm.ColorEstimatedFPS = objProcessDataForehead.ColorEstimatedFPS
        objProcessDataForehead.objAlgorithm.IREstimatedFPS = objProcessDataForehead.IREstimatedFPS
        objProcessDataForehead.objPlots.ColorEstimatedFPS = objProcessDataForehead.ColorEstimatedFPS
        objProcessDataForehead.objPlots.IREstimatedFPS = objProcessDataForehead.IREstimatedFPS
        objProcessDataForehead.Window_count =WindowCount
        objProcessDataForehead.region = objConfig.roiregions[1]

        # Initialise object to process face regions signal data
        objProcessDataLeftCheek =dataObject.WindowRegionList[objConfig.roiregions[2]]
        objProcessDataLeftCheek.Algorithm_type = Algorithm_type
        objProcessDataLeftCheek.FFT_type = FFT_type
        objProcessDataLeftCheek.Filter_type = Filter_type
        objProcessDataLeftCheek.Result_type = Result_type
        objProcessDataLeftCheek.Preprocess_type = Preprocess_type
        objProcessDataLeftCheek.SavePath = SavePath
        objProcessDataLeftCheek.ignoregray = objConfig.ignoregray
        objProcessDataLeftCheek.isSmoothen = isSmoothen
        objProcessDataLeftCheek.GenerateGraphs = objConfig.GenerateGraphs
        objProcessDataLeftCheek.fileName = currentSaveFileName
        objProcessDataLeftCheek.DumpToDisk = self.objConfig.DumpToDisk
        objProcessDataLeftCheek.objAlgorithm.ColorEstimatedFPS = objProcessDataLeftCheek.ColorEstimatedFPS
        objProcessDataLeftCheek.objAlgorithm.IREstimatedFPS = objProcessDataLeftCheek.IREstimatedFPS
        objProcessDataLeftCheek.objPlots.ColorEstimatedFPS = objProcessDataLeftCheek.ColorEstimatedFPS
        objProcessDataLeftCheek.objPlots.IREstimatedFPS = objProcessDataLeftCheek.IREstimatedFPS
        objProcessDataLeftCheek.Window_count =WindowCount
        objProcessDataLeftCheek.region = objConfig.roiregions[2]

        # Initialise object to process face regions signal data
        objProcessDataRightCheek =dataObject.WindowRegionList[objConfig.roiregions[3]]
        objProcessDataRightCheek.Algorithm_type = Algorithm_type
        objProcessDataRightCheek.FFT_type = FFT_type
        objProcessDataRightCheek.Filter_type = Filter_type
        objProcessDataRightCheek.Result_type = Result_type
        objProcessDataRightCheek.Preprocess_type = Preprocess_type
        objProcessDataRightCheek.SavePath = SavePath
        objProcessDataRightCheek.ignoregray = objConfig.ignoregray
        objProcessDataRightCheek.isSmoothen = isSmoothen
        objProcessDataRightCheek.GenerateGraphs = objConfig.GenerateGraphs
        objProcessDataRightCheek.fileName = currentSaveFileName
        objProcessDataRightCheek.DumpToDisk = self.objConfig.DumpToDisk
        objProcessDataRightCheek.objAlgorithm.ColorEstimatedFPS = objProcessDataRightCheek.ColorEstimatedFPS
        objProcessDataRightCheek.objAlgorithm.IREstimatedFPS = objProcessDataRightCheek.IREstimatedFPS
        objProcessDataRightCheek.objPlots.ColorEstimatedFPS = objProcessDataRightCheek.ColorEstimatedFPS
        objProcessDataRightCheek.objPlots.IREstimatedFPS = objProcessDataRightCheek.IREstimatedFPS
        objProcessDataRightCheek.Window_count =WindowCount
        objProcessDataRightCheek.region = objConfig.roiregions[3]

        # Initialise object to process face regions signal data
        objProcessDataCheeksCombined =dataObject.WindowRegionList[objConfig.roiregions[4]]
        objProcessDataCheeksCombined.Algorithm_type = Algorithm_type
        objProcessDataCheeksCombined.FFT_type = FFT_type
        objProcessDataCheeksCombined.Filter_type = Filter_type
        objProcessDataCheeksCombined.Result_type = Result_type
        objProcessDataCheeksCombined.Preprocess_type = Preprocess_type
        objProcessDataCheeksCombined.SavePath = SavePath
        objProcessDataCheeksCombined.ignoregray = objConfig.ignoregray
        objProcessDataCheeksCombined.isSmoothen = isSmoothen
        objProcessDataCheeksCombined.GenerateGraphs = objConfig.GenerateGraphs
        objProcessDataCheeksCombined.fileName = currentSaveFileName
        objProcessDataCheeksCombined.DumpToDisk = self.objConfig.DumpToDisk
        objProcessDataCheeksCombined.objAlgorithm.ColorEstimatedFPS = objProcessDataCheeksCombined.ColorEstimatedFPS
        objProcessDataCheeksCombined.objAlgorithm.IREstimatedFPS = objProcessDataCheeksCombined.IREstimatedFPS
        objProcessDataCheeksCombined.objPlots.ColorEstimatedFPS = objProcessDataCheeksCombined.ColorEstimatedFPS
        objProcessDataCheeksCombined.objPlots.IREstimatedFPS = objProcessDataCheeksCombined.IREstimatedFPS
        objProcessDataCheeksCombined.Window_count =WindowCount
        objProcessDataCheeksCombined.region = objConfig.roiregions[4]

        objProcessedParticipantsData = ProcessedParticipantsData(dataObject.minFPSIRvalue, dataObject.maxFPSIRvalue, dataObject.minFPSColorvalue,
                                                                 dataObject.maxFPSColorvalue, dataObject.FPSNotes,
                                                                 dataObject.LengthofAllFramesColor, dataObject.LengthofAllFramesIR,
                                                                 dataObject.totalTimeinSeconds, dataObject.TimeinSeconds, dataObject.timeinSeconds,
                                                                 dataObject.TotalWindows, dataObject.HrAvgList, dataObject.SPOAvgList,
                                                                 Algorithm_type, FFT_type, Filter_type, Result_type,
                                                                 Preprocess_type,
                                                                 isSmoothen)

        #objProcessDataLips process data
        objProcessDataLips.ProcessStep(processingStep)
        # objProcessDataForehead process data
        objProcessDataForehead.ProcessStep(processingStep)
        # objProcessDataLeftCheek process data
        objProcessDataLeftCheek.ProcessStep(processingStep)
        # objProcessDataRightCheek process data
        objProcessDataRightCheek.ProcessStep(processingStep)
        # objProcessDataCheeksCombined process data
        objProcessDataCheeksCombined.ProcessStep(processingStep)

        WindowRegionList[objConfig.roiregions[0]] = objProcessDataLips
        WindowRegionList[objConfig.roiregions[1]] = objProcessDataForehead
        WindowRegionList[objConfig.roiregions[2]] = objProcessDataLeftCheek
        WindowRegionList[objConfig.roiregions[3]] = objProcessDataRightCheek
        WindowRegionList[objConfig.roiregions[4]] = objProcessDataCheeksCombined

        objProcessedParticipantsData.WindowRegionList = WindowRegionList

        if (self.objConfig.DumpToDisk):
            objFile.DumpObjecttoDisk(SavePath + currentSaveFileName  + '\\',
                                     "ProcessedWindow_" + str(WindowCount), objProcessedParticipantsData)

        del objProcessedParticipantsData

        del objProcessDataLips
        del objProcessDataForehead
        del objProcessDataLeftCheek
        del objProcessDataRightCheek
        del objProcessDataCheeksCombined

        del objFile


    '''
    Save readings to file or db
    '''
    def ExtractReadings(self, dataObject, objConfig, prevfileName, objReliability,participant_number,position):
        # Lists to hold heart rate and blood oxygen data
        bestHeartRateSnr = 0.0
        bestBpm = 0.0
        channeltype = ''
        regiontype = ''
        freqencySamplingError = 0.0
        previousComputedHeartRate = 0.0
        smallestOxygenError = sys.float_info.max
        finaloxy = 0.0
        WindowDataRow=""

        TotalWindowCalculationTime = "00:00:00"
        PreProcessTime = "00:00:00"
        AlgorithmTime = "00:00:00"
        FFTTime = "00:00:00"
        SmoothTime = "00:00:00"
        FilterTime = "00:00:00"
        ComputerHRSNRTime = "00:00:00"
        ComputerSPOTime = "00:00:00"

        # ROI Window Result list
        WindowRegionList = {}

        # HrAvgList = dataObject.HrAvgList
        # SPOAvgList = dataObject.SPOAvgList
        HrGr= self.objFile.ReaddatafromFile(self.objConfig.DiskPath + "GroundTruthData\\"+ participant_number + "\\" +position+"\\","HR")
        SpoGr=self.objFile.ReaddatafromFile(self.objConfig.DiskPath + "GroundTruthData\\"+ participant_number + "\\" +position+"\\","SPO")

        HrGrFinal=[]
        SpoGrFinal=[]

        for element in HrGr:
            HrGrFinal.append(int(element.strip()))

        for element in SpoGr:
            SpoGrFinal.append(int(element.strip()))

        # Split ground truth data
        HrAvgWindowSizeList = CommonMethods.splitGroundTruth(HrGrFinal, dataObject.TotalWindows, dataObject.timeinSeconds)
        SPOAvgWindowSizeList = CommonMethods.splitGroundTruth(SpoGrFinal, dataObject.TotalWindows, dataObject.timeinSeconds)
        
        HrLastSecondList = CommonMethods.splitLastSecondGroundTruth(HrGrFinal, dataObject.TotalWindows, dataObject.timeinSeconds)
        SPOLastSecondList = CommonMethods.splitLastSecondGroundTruth(SpoGrFinal, dataObject.TotalWindows, dataObject.timeinSeconds)

        # print(prevfileName)
        WindowCountList = prevfileName.split("_")
        WindowCount = int(WindowCountList[1])

        # Store Data in Window List
        WindowRegionList['lips'] = dataObject.WindowRegionList[objConfig.roiregions[0]]
        WindowRegionList['forehead'] = dataObject.WindowRegionList[objConfig.roiregions[1]]
        WindowRegionList['leftcheek'] = dataObject.WindowRegionList[objConfig.roiregions[2]]
        WindowRegionList['rightcheek'] = dataObject.WindowRegionList[objConfig.roiregions[3]]
        WindowRegionList['cheeksCombined'] = dataObject.WindowRegionList[objConfig.roiregions[4]]

        # Get best region data
        for k, v in WindowRegionList.items():
            if (v.ProcessedwindowList.BestSnR > bestHeartRateSnr):
                bestHeartRateSnr = v.ProcessedwindowList.BestSnR
                bestBpm = v.ProcessedwindowList.BestBPM
                channeltype = v.channeltype
                regiontype = k
                freqencySamplingError = v.FrequencySamplieErrorForChannel
                SelectedColorFPS = v.ColorEstimatedFPS
                SelectedIRFPS = v.IREstimatedFPS
                SelectedColorFPSMethod = v.SelectedColorFPSMethod
                SelectedIRFPSMethod = v.SelectedIRFPSMethod

                #todo: Fix IN ALL FILES
                # TotalWindowCalculationTime, PreProcessTime, AlgorithmTime, \
                # FFTTime, SmoothTime, FilterTime, \
                # ComputerHRSNRTime, ComputerSPOTime = v.ProcessedwindowList.gettimeDifferencesToStringIndividual()

            if (v.ProcessedwindowList.oxygenSaturationValueError < smallestOxygenError):
                smallestOxygenError = v.ProcessedwindowList.oxygenSaturationValueError
                finaloxy = v.ProcessedwindowList.oxygenSaturationValueValue

        if (bestBpm > 30 and bestBpm < 250):
            # Check reliability and record best readings
            heartRateValue, heartRateError = objReliability.AcceptorRejectHR(bestHeartRateSnr, bestBpm,
                                                                             freqencySamplingError)
            oxygenSaturationValue, oxygenSaturationValueError = objReliability.AcceptorRejectSPO(smallestOxygenError,
                                                                                                 finaloxy)
            # Get difference and append data (heart rate)
            AveragedifferenceHR = round(float(HrAvgWindowSizeList[WindowCount]) - float(heartRateValue))
            OriginalObtianedAveragedifferenceHR = round(float(HrAvgWindowSizeList[WindowCount]) - float(bestBpm))
            OriginalObtianedLastSecondWindowdifferenceHR = round(float(HrLastSecondList[WindowCount]) - float(bestBpm))
            LastSecondWindowdifferenceHR = round(float(HrLastSecondList[WindowCount]) - float(heartRateValue))

            # Get difference and append data (blood oxygen)
            AveragedifferenceSPO = round(float(SPOAvgWindowSizeList[WindowCount]) - float(oxygenSaturationValue))
            OriginalObtianedAveragedifferenceSPO = round(float(SPOAvgWindowSizeList[WindowCount]) - float(finaloxy))
            LastSecondWindowdifferenceSPO = round(float(SPOLastSecondList[WindowCount]) - float(oxygenSaturationValue))
            OriginalObtianedLastSecondWindowdifferenceSPO = round(float(SPOLastSecondList[WindowCount]) - float(finaloxy))

            bestSnrString = str(float(bestHeartRateSnr))
            if (bestSnrString == "inf"):
                bestSnrString = '-1'

            WindowDataRow = str( str(WindowCount) + "," +
                str(bestSnrString) + "," +
                str(round(HrAvgWindowSizeList[WindowCount])) + "," +
                str(round(heartRateValue)) + "," +
                str(AveragedifferenceHR) + "," +
                str(bestBpm) + "," +
                str(OriginalObtianedAveragedifferenceHR) + "," +
                str((HrLastSecondList[WindowCount])) + "," +
                str(LastSecondWindowdifferenceHR ) + "," +
                str(OriginalObtianedLastSecondWindowdifferenceHR) + "," +
                str(round(SPOAvgWindowSizeList[WindowCount])) + "," +
                str(round(oxygenSaturationValue)) + "," +
                str(AveragedifferenceSPO) + "," +
                str(finaloxy) + "," +
                str(OriginalObtianedAveragedifferenceSPO) + "," +
                str((SPOLastSecondList[WindowCount])) + "," +
                str(LastSecondWindowdifferenceSPO ) + "," +
                str(OriginalObtianedLastSecondWindowdifferenceSPO ) + "," +
                str(regiontype) + "," +
                str(channeltype) + "," +
                str(float(freqencySamplingError)) + "," +
                str(float(heartRateError)) + "," +
                str(TotalWindowCalculationTime) + "," +
                str(PreProcessTime) + "," +
                str(AlgorithmTime) + "," +
                str(FFTTime) + "," +
                str(SmoothTime) + "," +
                str(FilterTime) + "," +
                str(ComputerHRSNRTime) + "," +
                str(ComputerSPOTime) + "," +
                str(dataObject.Algorithm_type) + "," +
                str(dataObject.FFT_type) + "," +
                str(dataObject.Filter_type) + "," +
                str(dataObject.Result_type) + "," +
                str(dataObject.Preprocess_type) + "," +
                str(dataObject.isSmoothen) + "," +
                str(SelectedColorFPS) + "," +
                str(SelectedIRFPS) + "," +
                str(SelectedColorFPSMethod) + "," +
                str(SelectedIRFPSMethod) + "," +
                str(1) + "," +
                dataObject.FPSNotes.replace(",",";") + "," +
                str(False) )
        else:
            WindowDataRow =str( "," + "," + "," + "," + ","  + "," + "," + ","
                                + ","+ "," + ","+ ","  + "," + "," + "," + ","  + "," + "," + "," + "," + ","+ ","  + "," + "," + ","
                                + "," + ","  + "," + ","  + "," + "," + "," + "," + "," + "," + ","  + ","  + ","  + ","  + ","  + ","  + "," )

        # del objReliability

        return WindowDataRow

    '''
    Process participants data over the entire signal data
    '''

    def GenerateGraph(self, objFaceImage, ProcessedDataPath, region):

        # Initialise object to process face regions
        objProcessData = ProcessDataPartial(objFaceImage.ColorEstimatedFPS, objFaceImage.IREstimatedFPS,
                                            ProcessedDataPath)
        # Generate graph
        objProcessData.GenerateGrapth("RawData_" + region, objFaceImage.blue, objFaceImage.green, objFaceImage.red,
                                      objFaceImage.grey, objFaceImage.Irchannel)
        # clear
        del objProcessData

    '''
       Process participants data over the entire signal data
       '''

    def PreProcessData(self, blue, green, red, grey, Irchannel, ColorEstimatedFPS, IREstimatedFPS, region,
                       timecolorCount, timeirCount):
        # Initialise object to process face regions
        objProcessData = ProcessDataPartial(ColorEstimatedFPS, IREstimatedFPS, self.ProcessedDataPath)
        for Preprocess_type in self.objConfig.preprocesses:
            # Generate data
            objProcessData.PreProcessData(blue, green, red, grey, Irchannel, self.objConfig.GenerateGraphs, region,
                                          Preprocess_type, timecolorCount, timeirCount)
        # clear
        del objProcessData

    '''
       Process participants data over the entire signal data
       '''

    def ApplyAlgorithm(self, SCombined, ColorEstimatedFPS, IREstimatedFPS, region, preProcessType):
        # Initialise object to process face regions
        objProcessData = ProcessDataPartial(ColorEstimatedFPS, IREstimatedFPS, self.ProcessedDataPath)
        for algo_type in self.objConfig.AlgoList:
            # Generate data
            objProcessData.ApplyAlgorithm(SCombined, algo_type, 5, self.objConfig.GenerateGraphs, region,
                                          preProcessType)
        # clear
        del objProcessData

    '''
       Process participants data over the entire signal data
       '''

    def ApplySmooth(self, SCombined, ColorEstimatedFPS, IREstimatedFPS, region, preProcessType, Algotype):
        # Initialise object to process face regions
        objProcessData = ProcessDataPartial(ColorEstimatedFPS, IREstimatedFPS, self.ProcessedDataPath)
        # Generate data
        objProcessData.SmoothenData(SCombined, self.objConfig.GenerateGraphs, region, preProcessType, Algotype)
        # clear
        del objProcessData

    '''
       Process participants data over the entire signal data
       '''

    def ApplyFFT(self, SCombined, ColorEstimatedFPS, IREstimatedFPS, region, preProcessType, Algotype, IsSmooth):
        # Initialise object to process face regions
        objProcessData = ProcessDataPartial(ColorEstimatedFPS, IREstimatedFPS, self.ProcessedDataPath)
        for fft_type in self.objConfig.fftTypeList:
            # Generate data
            objProcessData.ApplyFFT(SCombined, fft_type, region, self.objConfig.GenerateGraphs, IsSmooth, Algotype,
                                    preProcessType)
        # clear
        del objProcessData

    '''
          Process participants data over the entire signal data
          '''

    def ApplyFilter(self, ProcessedSignalData, ColorEstimatedFPS, IREstimatedFPS,
                    region, preProcessType, Algotype, IsSmooth, FFT_type,
                    Colorfrequency, IRfrequency, ignore_freq_below, ignore_freq_above):
        # Initialise object to process face regions
        objProcessData = ProcessDataPartial(ColorEstimatedFPS, IREstimatedFPS, self.ProcessedDataPath)
        for filtertype in self.objConfig.filtertypeList:
            # Generate data
            objProcessData.FilterTechniques(ProcessedSignalData, region, self.objConfig.GenerateGraphs, IsSmooth,
                                            Algotype,
                                            preProcessType,
                                            FFT_type, filtertype, Colorfrequency, IRfrequency, ignore_freq_below,
                                            ignore_freq_above)
        # clear
        del objProcessData

    '''
    Process participants data over the entire signal data
    '''

    def ComputerResultData(self, B_filtered, G_filtered, R_filtered, Gy_filtered, IR_filtered, ColorEstimatedFPS,
                           IREstimatedFPS,
                           region, preProcessType, Algotype, IsSmooth, FFT_type, filterType,
                           Colorfrequency, IRfrequency, ignore_freq_below_bpm, ignore_freq_above_bpm):
        # Initialise object to process face regions
        objProcessData = ProcessDataPartial(ColorEstimatedFPS, IREstimatedFPS, self.ProcessedDataPath)
        for resultType in self.objConfig.resulttypeList:
            ##If esitss
            ResultFilesAlreadyGenerated = "ResultType_RS-" + str(resultType) + "_Filtered_FL-" + str(
                filterType) + "_" + region + \
                                          "_FFTtype-" + str(FFT_type) + "_algotype-" + str(
                Algotype) + '_PreProcessType-' + str(preProcessType) + \
                                          "_Smoothed-" + str(IsSmooth)
            if not os.path.exists(self.ProcessedDataPath + ResultFilesAlreadyGenerated):
                # '_PreProcessType-'+str(preProcessType)+ "_Smoothed-" + str(IsSmooth)
                # Generate data
                objProcessData.generateHeartRateandSNR(B_filtered, G_filtered, R_filtered, Gy_filtered, IR_filtered,
                                                       resultType,
                                                       Colorfrequency, IRfrequency, ColorEstimatedFPS, IREstimatedFPS,
                                                       preProcessType, Algotype, IsSmooth, FFT_type, filterType,
                                                       ignore_freq_below_bpm, ignore_freq_above_bpm, region)
        # clear
        del objProcessData

    def getFileName(self, typeList, region, ProcesstypeName):
        FileNames = []
        for type in typeList:
            fileName = ProcesstypeName + '_' + region + '_type-' + str(type)
            if (fileName not in FileNames):
                FileNames.append(fileName)
        return FileNames

    def getPreProcessedAlgorithms(self, region):
        FileNames = []
        for preprocess_type in self.objConfig.preprocesses:
            for algo_type in self.objConfig.AlgoList:
                fileName = 'AlgorithmData_' + region + "_type-" + str(algo_type) + '_PreProcessType-' + str(
                    preprocess_type)
                if (fileName not in FileNames):
                    FileNames.append(fileName)
        return FileNames

    def getPreProcessedAlgorithmsFFT(self, region, isSmoothed):
        FileNames = []
        fileConent = {}
        for preprocess_type in self.objConfig.preprocesses:
            for algo_type in self.objConfig.AlgoList:
                for fft_type in self.objConfig.fftTypeList:
                    fileName = "FFT-" + fft_type + "_" + region + "_Smoothed-" + str(isSmoothed) + "_Algotype-" + str(
                        algo_type) + '_PreProcessType-' + str(preprocess_type)

                    if (fileName not in FileNames):
                        FileNames.append(fileName)
                        ###
                        dataContent = self.ReadData(fileName)
                        fileConent[fileName] = dataContent

        FileNames = []
        return fileConent

    def getPreProcessedFiltered(self, region, isSmoothed):

        FileNames = []
        fileConent = {}
        for preprocess_type in self.objConfig.preprocesses:
            for algo_type in self.objConfig.AlgoList:
                for fft_type in self.objConfig.fftTypeList:
                    for filterType in self.objConfig.filtertypeList:
                        fileName = "Filtered_FL-" + str(filterType) + "_" + region + "_FFTtype-" + str(
                            fft_type) + "_algotype-" + str(algo_type) + '_PreProcessType-' + str(
                            preprocess_type) + "_Smoothed-" + str(isSmoothed)
                        if (fileName not in FileNames):
                            FileNames.append(fileName)
                            dataContent = self.ReadData(fileName)
                            fileConent[fileName] = dataContent
        FileNames = []
        return fileConent

    def getResult(self, region, participant_number, position):
        FileNames = []
        fileConent = {}
        for preprocess_type in self.objConfig.preprocesses:
            for algo_type in self.objConfig.AlgoList:
                for fft_type in self.objConfig.fftTypeList:
                    for filterType in self.objConfig.filtertypeList:
                        for resulttype in self.objConfig.resulttypeList:
                            for isSmoothed in self.objConfig.Smoothen:
                                fileName = "ResultType_RS-" + str(resulttype) + "_Filtered_FL-" + str(
                                    filterType) + "_" + region + "_FFTtype-" + str(fft_type) \
                                           + "_algotype-" + str(algo_type) + \
                                           '_PreProcessType-' + str(preprocess_type) + "_Smoothed-" + str(isSmoothed)
                                if (fileName not in FileNames):
                                    FileNames.append(fileName)
                                    if (isSmoothed == False):
                                        self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'ResultDataWithoutSmooth\\'
                                    else:
                                        self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'ResultDataSmoothed\\'
                                    dataContent = self.ReadData(fileName)
                                    fileConent[fileName] = dataContent
        FileNames = []
        return fileConent

    def getPreProcessedAlgorithmSmoothedFiles(self, region):
        FileNames = []
        for preprocess_type in self.objConfig.preprocesses:
            for algo_type in self.objConfig.AlgoList:
                fileName = 'SmoothedData_' + region + "_Algotype-" + str(algo_type) + '_PreProcessType-' + str(
                    preprocess_type)
                if (fileName not in FileNames):
                    FileNames.append(fileName)
        return FileNames

    def GenerateCases(self):
        CaseList = []
        LoadfNameList = []
        for preprocesstype in self.objConfig.preprocesses:
            for algoType in self.objConfig.AlgoList:
                for isSmooth in self.objConfig.Smoothen:
                    for fftype in self.objConfig.fftTypeList:
                        for filtertype in self.objConfig.filtertypeList:
                            for resulttype in self.objConfig.resulttypeList:
                                fileName = "HRSPOwithLog_" + str(algoType) + "_PR-" + str(
                                    preprocesstype) + "_FFT-" + str(fftype) \
                                           + "_FL-" + str(filtertype) + "_RS-" + str(resulttype) + "_SM-" + str(
                                    isSmooth)
                                LoadName = 'ResultSignal_Result-' + str(resulttype) + '_PreProcess-' + str(
                                    preprocesstype) \
                                           + '_Algorithm-' + str(algoType) + '_Smoothen-' + str(isSmooth) \
                                           + '_FFT-' + str(fftype) + '_Filter-' + str(filtertype)
                                if (fileName not in CaseList):  # not os.path.exists(self.ProcessedDataPath + fName):
                                    CaseList.append(fileName)
                                    LoadfNameList.append(LoadName)
        return CaseList, LoadfNameList

    def ProduceFinalResult(self):
        # each particpant
        for participant_number in self.objConfig.ParticipantNumbers:
            # each position
            for position in self.objConfig.hearratestatus:
                # print
                print(participant_number + ', ' + position)
                # set path
                self.objConfig.setSavePath(participant_number, position)

                # self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'ResultDataSmoothed\\'
                RegionStoreDataContent = {}
                lips_DataFileNameWithContent = self.getResult("lips", participant_number, position)
                forehead_DataFileNameWithContent = self.getResult("forehead", participant_number, position)
                leftcheek_DataFileNameWithContent = self.getResult("leftcheek", participant_number, position)
                rightcheek_DataFileNameWithContent = self.getResult("rightcheek", participant_number, position)
                self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'RawOriginal\\'
                lipsobjFaceImage = self.ReadOriginalRawData('lips')
                self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'FinalComputedResult\\'
                caselist = self.GenerateCasesNewMethod(participant_number, position)
                # self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'ResultDataSmoothed\\'
                HrGr, SpoGr = CommonMethods.GetGroundTruth(participant_number, position,
                                                           self.objConfig.DiskPath,
                                                           int(lipsobjFaceImage.totalTimeinSeconds))
                self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'FinalComputedResult\\'
                for filename in caselist:
                    saveFilename = filename.replace('_REGION_', '')  ##IF EXISTS
                    lipsFileName = filename.replace('REGION', 'lips')
                    foreheadFileName = filename.replace('REGION', 'forehead')
                    leftcheekFileName = filename.replace('REGION', 'leftcheek')
                    rightcheekFileName = filename.replace('REGION', 'rightcheek')
                    lipsContent = lips_DataFileNameWithContent.get(lipsFileName)
                    foreheadContent = forehead_DataFileNameWithContent.get(foreheadFileName)
                    leftcheekContent = leftcheek_DataFileNameWithContent.get(leftcheekFileName)
                    rightcheekContent = rightcheek_DataFileNameWithContent.get(rightcheekFileName)
                    Process_Participants_Data_GetBestHR(lipsContent, foreheadContent, rightcheekContent,
                                                        leftcheekContent, self.ProcessedDataPath, HrGr, saveFilename)

    def LogTime(self):
        logTime = datetime(datetime.now().year, datetime.now().month, datetime.now().day,
                           datetime.now().time().hour, datetime.now().time().minute,
                           datetime.now().time().second, datetime.now().time().microsecond)
        return logTime


    def CustomCaseList(self):
        # CustomCases = self.objFile.ReaddatafromFile(self.objConfig.DiskPath,'NoHrFilesCases')
        CustomCases = []
        CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-3_RS-2_SM-True')
        LoadFileName = []
        Cases = []
        for case in CustomCases:
            case = case.replace('\n','')
            Cases.append(case)
            caseSplit = case.split('_')
            algoType = caseSplit[1]
            preprocesstype = caseSplit[2].replace('PR-','')
            fftype = caseSplit[3].replace('FFT-','')
            filtertype = caseSplit[4].replace('FL-','')
            resulttype = caseSplit[5].replace('RS-','')
            isSmooth = caseSplit[6].replace('SM-','')
            LoadName = 'ResultSignal_Result-' + str(resulttype) + '_PreProcess-' + str(
                preprocesstype) \
                       + '_Algorithm-' + str(algoType) + '_Smoothen-' + str(isSmooth) \
                       + '_FFT-' + str(fftype) + '_Filter-' + str(filtertype)
            LoadFileName.append(LoadName)
        return Cases, LoadFileName

    def getCase(self, fileName):
        SavePath = self.objConfig.SavePath +'ComputedFinalResult\\'
        filepath = SavePath + fileName + '.txt'  # HeartRate_FastICA_FFT-M1_FL-6_RS-1_PR-1_SM-False

        pathExsists = self.objFile.FileExits(filepath)
        data = None
        # already generated
        if (pathExsists):
            Filedata = open(filepath, "r")
            data = Filedata.read().split("\n")[0]
            Filedata.close()
        return data

    '''
       Process_Participants_Data_EntireSignalINChunks:
       '''
    def Process_Participants_Result_forEntireSignal(self, typeProcessing):
        CaseList, LoadfNameList = self.GenerateCases()

        TotalCasesCount = len(CaseList)
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                print(participant_number + ', ' + position)
                self.objConfig.setSavePath(participant_number, position, 'ProcessedDatabyProcessType')
                SavePath = self.objConfig.SavePath
                currentCasesDone = 0

                # Load groundtruth
                CurrentSavePath = self.objConfig.SavePath
                OriginalPath = CurrentSavePath.replace('ProcessedDatabyProcessType', 'RawOriginal')
                objOriginalUncompressedData = self.objFile.ReadfromDisk(OriginalPath,
                                                                        'UnCompressedBinaryLoadedData')
                HrGr = objOriginalUncompressedData.HrGroundTruthList
                SpoGr = objOriginalUncompressedData.SPOGroundTruthList
                HrAvg = CommonMethods.AvegrageGroundTruth(HrGr)
                SPOAvg = CommonMethods.AvegrageGroundTruth(SpoGr)

                countFname = 0
                for fileName in CaseList:
                    # CaseData = self.getCase(fileName)
                    # if(CaseData !=''):
                    #     continue

                    # Case percentage
                    currentCasesDone = currentCasesDone + 1
                    currentPercentage = round((currentCasesDone / TotalCasesCount) * 100)
                    currentPercentage = str(currentPercentage)
                    print(str(fileName) + "  -> " + str(currentPercentage) + " out of 100%")
                    if(self.objFile.FileExits(SavePath + 'ComputedFinalResult\\'+ fileName + '.txt')):
                        skip=0
                    else:
                        WindowRegionList = {}

                        objReliability = CheckReliability()
                        # Lists to hold heart rate and blood oxygen data
                        Listdata = []
                        # ListSPOdata = []
                        bestHeartRateSnr = 0.0
                        bestBpm = 0.0
                        channeltype = ''
                        regiontype = ''
                        freqencySamplingError = 0.0
                        previousComputedHeartRate = 0.0
                        smallestOxygenError = sys.float_info.max
                        timeinSeconds = 10
                        finaloxy = 0.0
                        SPOstd = 0.0
                        FullTimeLog = None
                        WindowCount = 0


                        LoadFileName = LoadfNameList[countFname]
                        SaveName = fileName
                        # LoadData
                        objWindowProcessedData = self.objFile.ReadfromDisk(self.objConfig.SavePath + typeProcessing + '\\',
                                                                           LoadFileName)

                        countFname = countFname + 1

                        # Store Data in Window List
                        WindowRegionList['lips'] = objWindowProcessedData.get(self.objConfig.roiregions[0])
                        WindowRegionList['forehead'] = objWindowProcessedData.get(self.objConfig.roiregions[1])
                        WindowRegionList['leftcheek'] = objWindowProcessedData.get(self.objConfig.roiregions[2])
                        WindowRegionList['rightcheek'] = objWindowProcessedData.get(self.objConfig.roiregions[3])

                        # Get best region data
                        for k, v in WindowRegionList.items():
                            if (v.bestHeartRateSnr > bestHeartRateSnr):
                                bestHeartRateSnr = v.bestHeartRateSnr
                                bestBpm = v.bestBpm
                                channeltype = v.channeltype
                                regiontype = k
                                freqencySamplingError = v.IrFreqencySamplingError
                                FullTimeLog = v.FullLog
                                # diffNow = v.diffTime
                                # diffTimeLog = v.diffTimeLog

                            if (v.SPOerr < smallestOxygenError):
                                smallestOxygenError = v.SPOerr  # oxygenSaturationValueError
                                finaloxy = v.SPOoxylevl  # oxygenSaturationValueValue
                                SPOstd = v.SPOstd  # oxygenSaturationValueValue

                        if (bestBpm > 0):
                            # Check reliability and record best readings
                            heartRateValue, heartRateError = objReliability.AcceptorRejectHR(bestHeartRateSnr, bestBpm,
                                                                                             freqencySamplingError)
                            oxygenSaturationValue, oxygenSaturationValueError = objReliability.AcceptorRejectSPO(
                                smallestOxygenError,
                                finaloxy)

                            # Get difference and append data (heart rate)
                            differenceHR = round(float(HrAvg) - float(heartRateValue))

                            # Get difference and append data (blood oxygen)
                            differenceSPO = round(float(SPOAvg) - float(oxygenSaturationValue))

                            Listdata.append(
                                'WindowCount: ' + str(WindowCount) + " ,\t" + 'GroundTruthHeartRate: ' + str(
                                    round(HrAvg)) + " ,\t" + 'ComputedHeartRate: ' + str(
                                    round(heartRateValue)) + " ,\t" + 'HRDifference: ' + str(
                                    differenceHR) + " ,\t" + 'GroundTruthSPO: ' + str(
                                    round(SPOAvg)) + " ,\t" + 'ComputedSPO: ' + str(
                                    round(oxygenSaturationValue)) + " ,\t" + 'SPODifference: ' + str(
                                    differenceSPO) + " ,\t" + 'Regiontype: ' + " ,\t" + str(
                                    regiontype) + " ,\t" + FullTimeLog)

                        # filename
                        fileNameResult =  fileName #"HRSPOwithLog_" +
                        #          regiontype + "_" + Algorithm_type + "_FFT-" + str(FFT_type) + "_FL-" + str(
                        # Filter_type) + "_RS-" + str(Result_type) + "_PR-" + str(Preprocess_type) + "_SM-" + str(
                        # isSmoothen)
                        # Write data to file
                        self.objFile.WriteListDatatoFile(SavePath + 'ComputedFinalResult\\', fileNameResult, Listdata)

                        del objReliability

    '''
    Process_Participants_Data_EntireSignalINChunks:
    '''

    def Process_Participants_Data_EntireSignalINChunks(self, objWindowProcessedDataOriginal, SavePath, DumpToDisk,
                                                       ProcessingStep,
                                                       ProcessingType, regions):
        # ROI Window Result list
        WindowRegionList = {}
        objWindowProcessedData = objWindowProcessedDataOriginal
        # Loop through signal data
        WindowCount = 0
        Preprocess_type = 0
        Algorithm_type = 0
        FFT_type = 0
        Filter_type = 0
        isSmoothen = 0
        Result_type = 0
        PreviousStepDetails = ''

        ##Process by type
        if (ProcessingStep == 'PreProcess'):
            Preprocess_type = ProcessingType
        elif (ProcessingStep == 'Algorithm'):
            Algorithm_type = ProcessingType
        elif (ProcessingStep == 'Smoothen'):
            isSmoothen = ProcessingType
        elif (ProcessingStep == 'FFT'):
            FFT_type = ProcessingType
        elif (ProcessingStep == 'Filter'):
            Filter_type = ProcessingType
        elif (ProcessingStep == 'Result'):
            Result_type = ProcessingType

        TotalWindows = 1
        fileName = 'ResultSignal_' + ProcessingStep + '-' + str(ProcessingType)
        objProcessData = None
        for region in regions:
            if (ProcessingStep == 'PreProcess'):
                ROIStore = objWindowProcessedData.ROIStore
                TimeinSeconds = ROIStore.get("lips").totalTimeinSeconds
                timeinSeconds = TimeinSeconds
                objProcessData = ProcessFaceData(Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type,
                                                 SavePath,
                                                 self.objConfig.ignoregray, isSmoothen, self.objConfig.GenerateGraphs,
                                                 timeinSeconds,
                                                 DumpToDisk, fileName)
                objProcessData.getSingalData(ROIStore, region, WindowCount, TotalWindows, timeinSeconds)
                # Windows for regions (should be same for all)

                # Log Start Time  signal data
                startLogTime = self.LogTime()  # foreach region
                # PreProcess Signal
                processedBlue, processedGreen, processedRed, processedGrey, processedIR = objProcessData.preprocessSignalData(
                    objProcessData.regionBlueData, objProcessData.regionGreenData,
                    objProcessData.regionRedData, objProcessData.regionGreyData,
                    objProcessData.regionIRData)

                endlogTime = self.LogTime()
                diffTime = endlogTime - startLogTime
                # Record
                objProcessData.regionWindowBlueData = processedBlue
                objProcessData.regionWindowGreenData = processedGreen
                objProcessData.regionWindowRedData = processedRed
                objProcessData.regionWindowGreyData = processedGrey
                objProcessData.regionWindowIRData = processedIR
                objProcessData.WindowProcessStartTime = startLogTime
                objProcessData.WindowProcessEndTime = endlogTime
                objProcessData.WindowProcessDifferenceTime = diffTime

                WindowRegionList[region] = objProcessData

            elif (ProcessingStep == 'Algorithm'):
                # regionWindowGreyData-> Previously Processoed , so apply further processing on that, passing that as input to new object, and output would be windowregion
                objProcessDataPrevious = objWindowProcessedData.get(region)
                PreviousStepDetails = '_PreProcess-' + str(objProcessDataPrevious.Preprocess_type)
                objProcessData = ProcessFaceData(Algorithm_type, FFT_type, Filter_type, Result_type,
                                                 objProcessDataPrevious.Preprocess_type,
                                                 SavePath,
                                                 self.objConfig.ignoregray, isSmoothen, self.objConfig.GenerateGraphs,
                                                 objProcessDataPrevious.timeinSeconds,
                                                 DumpToDisk, fileName)
                objProcessData.SetFromDataParameters(region, objProcessDataPrevious.IRfpswithTime,
                                                     objProcessDataPrevious.ColorfpswithTime, TotalWindows,
                                                     objProcessDataPrevious.timeinSeconds,
                                                     objProcessDataPrevious.WindowIRfpswithTime,
                                                     objProcessDataPrevious.WindowColorfpswithTime, WindowCount,
                                                     objProcessDataPrevious.ColorEstimatedFPS,
                                                     objProcessDataPrevious.IREstimatedFPS,
                                                     objProcessDataPrevious.regionStore,
                                                     objProcessDataPrevious.regionWindowBlueData,
                                                     objProcessDataPrevious.regionWindowGreenData,
                                                     objProcessDataPrevious.regionWindowRedData,
                                                     objProcessDataPrevious.regionWindowGreyData,
                                                     objProcessDataPrevious.regionWindowIRData,
                                                     objProcessDataPrevious.distanceM, None, None, None, None, None,
                                                     objProcessDataPrevious.timecolorCount,
                                                     objProcessDataPrevious.timeirCount,
                                                     objProcessDataPrevious.Frametime_list_ir,
                                                     objProcessDataPrevious.Frametime_list_color,
                                                     objProcessDataPrevious.Colorfrequency,
                                                     objProcessDataPrevious.IRfrequency
                                                     )
                startLogTime = self.LogTime()  # foreach region
                # Apply Algorithm on preprocseed
                processedBlue, processedGreen, processedRed, processedGrey, processedIR = objProcessData.ApplyAlgorithm(
                    objProcessData.regionBlueData, objProcessData.regionGreenData,
                    objProcessData.regionRedData, objProcessData.regionGreyData,
                    objProcessData.regionIRData)
                endlogTime = self.LogTime()
                diffTime = endlogTime - startLogTime

                objProcessData.regionWindowBlueData = processedBlue  # Algorithm processed to window channel data
                objProcessData.regionWindowGreenData = processedGreen
                objProcessData.regionWindowRedData = processedRed
                objProcessData.regionWindowGreyData = processedGrey
                objProcessData.regionWindowIRData = processedIR
                objProcessData.WindowProcessStartTime = startLogTime
                objProcessData.WindowProcessEndTime = endlogTime
                objProcessData.WindowProcessDifferenceTime = diffTime

                WindowRegionList[region] = objProcessData

            elif (ProcessingStep == 'Smoothen'):
                objProcessDataPrevious = objWindowProcessedData.get(region)
                PreviousStepDetails = '_PreProcess-' + str(
                    objProcessDataPrevious.Preprocess_type) + '_Algorithm-' + str(objProcessDataPrevious.Algorithm_type)
                objProcessData = ProcessFaceData(objProcessDataPrevious.Algorithm_type, FFT_type, Filter_type,
                                                 Result_type,
                                                 objProcessDataPrevious.Preprocess_type,
                                                 SavePath,
                                                 self.objConfig.ignoregray, isSmoothen, self.objConfig.GenerateGraphs,
                                                 objProcessDataPrevious.timeinSeconds,
                                                 DumpToDisk, fileName)
                objProcessData.SetFromDataParameters(region, objProcessDataPrevious.IRfpswithTime,
                                                     objProcessDataPrevious.ColorfpswithTime, TotalWindows,
                                                     objProcessDataPrevious.timeinSeconds,
                                                     objProcessDataPrevious.WindowIRfpswithTime,
                                                     objProcessDataPrevious.WindowColorfpswithTime, WindowCount,
                                                     objProcessDataPrevious.ColorEstimatedFPS,
                                                     objProcessDataPrevious.IREstimatedFPS,
                                                     objProcessDataPrevious.regionStore,
                                                     objProcessDataPrevious.regionWindowBlueData,
                                                     objProcessDataPrevious.regionWindowGreenData,
                                                     objProcessDataPrevious.regionWindowRedData,
                                                     objProcessDataPrevious.regionWindowGreyData,
                                                     objProcessDataPrevious.regionWindowIRData,
                                                     objProcessDataPrevious.distanceM, None, None, None, None, None
                                                     , objProcessDataPrevious.timecolorCount,
                                                     objProcessDataPrevious.timeirCount,
                                                     objProcessDataPrevious.Frametime_list_ir,
                                                     objProcessDataPrevious.Frametime_list_color,
                                                     objProcessDataPrevious.Colorfrequency,
                                                     objProcessDataPrevious.IRfrequency
                                                     )
                startLogTime = self.LogTime()  # foreach region
                processedBlue = None
                processedGreen = None
                processedRed = None
                processedGrey = None
                processedIR = None
                # Smooth Signal
                if (isSmoothen):
                    processedBlue = objProcessData.SmoothenData(objProcessData.regionBlueData)
                    processedGreen = objProcessData.SmoothenData(objProcessData.regionGreenData)
                    processedRed = objProcessData.SmoothenData(objProcessData.regionRedData)
                    processedGrey = objProcessData.SmoothenData(objProcessData.regionGreyData)
                    processedIR = objProcessData.SmoothenData(objProcessData.regionIRData)
                else:
                    processedBlue = objProcessData.regionBlueData
                    processedGreen = objProcessData.regionGreenData
                    processedRed = objProcessData.regionRedData
                    processedGrey = objProcessData.regionGreyData
                    processedIR = objProcessData.regionIRData

                endlogTime = self.LogTime()
                diffTime = endlogTime - startLogTime

                objProcessData.regionWindowBlueData = processedBlue
                objProcessData.regionWindowGreenData = processedGreen
                objProcessData.regionWindowRedData = processedRed
                objProcessData.regionWindowGreyData = processedGrey
                objProcessData.regionWindowIRData = processedIR
                objProcessData.WindowProcessStartTime = startLogTime
                objProcessData.WindowProcessEndTime = endlogTime
                objProcessData.WindowProcessDifferenceTime = diffTime

                WindowRegionList[region] = objProcessData

            elif (ProcessingStep == 'FFT'):
                objProcessDataPrevious = objWindowProcessedData.get(region)
                PreviousStepDetails = '_PreProcess-' + str(objProcessDataPrevious.Preprocess_type) + '_Algorithm-' \
                                      + str(objProcessDataPrevious.Algorithm_type) + '_Smoothen-' + str(
                    objProcessDataPrevious.isSmoothen)
                objProcessData = ProcessFaceData(objProcessDataPrevious.Algorithm_type, FFT_type, Filter_type,
                                                 Result_type,
                                                 objProcessDataPrevious.Preprocess_type,
                                                 SavePath,
                                                 self.objConfig.ignoregray, objProcessDataPrevious.isSmoothen,
                                                 self.objConfig.GenerateGraphs,
                                                 objProcessDataPrevious.timeinSeconds,
                                                 DumpToDisk, fileName)
                objProcessData.SetFromDataParameters(region, objProcessDataPrevious.IRfpswithTime,
                                                     objProcessDataPrevious.ColorfpswithTime, TotalWindows,
                                                     objProcessDataPrevious.timeinSeconds,
                                                     objProcessDataPrevious.WindowIRfpswithTime,
                                                     objProcessDataPrevious.WindowColorfpswithTime, WindowCount,
                                                     objProcessDataPrevious.ColorEstimatedFPS,
                                                     objProcessDataPrevious.IREstimatedFPS,
                                                     objProcessDataPrevious.regionStore,
                                                     objProcessDataPrevious.regionWindowBlueData,
                                                     objProcessDataPrevious.regionWindowGreenData,
                                                     objProcessDataPrevious.regionWindowRedData,
                                                     objProcessDataPrevious.regionWindowGreyData,
                                                     objProcessDataPrevious.regionWindowIRData,
                                                     objProcessDataPrevious.distanceM, None, None, None, None, None,
                                                     objProcessDataPrevious.timecolorCount,
                                                     objProcessDataPrevious.timeirCount,
                                                     objProcessDataPrevious.Frametime_list_ir,
                                                     objProcessDataPrevious.Frametime_list_color,
                                                     objProcessDataPrevious.Colorfrequency,
                                                     objProcessDataPrevious.IRfrequency
                                                     )
                startLogTime = self.LogTime()  # foreach region
                # Apply FFT
                processedBlue, processedGreen, processedRed, processedGrey, processedIR = objProcessData.ApplyFFT(
                    objProcessData.regionBlueData, objProcessData.regionGreenData,
                    objProcessData.regionRedData, objProcessData.regionGreyData,
                    objProcessData.regionIRData)
                endlogTime = self.LogTime()
                diffTime = endlogTime - startLogTime

                objProcessData.regionWindowBlueData = processedBlue
                objProcessData.regionWindowGreenData = processedGreen
                objProcessData.regionWindowRedData = processedRed
                objProcessData.regionWindowGreyData = processedGrey
                objProcessData.regionWindowIRData = processedIR
                objProcessData.WindowProcessStartTime = startLogTime
                objProcessData.WindowProcessEndTime = endlogTime
                objProcessData.WindowProcessDifferenceTime = diffTime

                WindowRegionList[region] = objProcessData

            elif (ProcessingStep == 'Filter'):
                objProcessDataPrevious = objWindowProcessedData.get(region)
                PreviousStepDetails = '_PreProcess-' + str(
                    objProcessDataPrevious.Preprocess_type) + '_Algorithm-' + str(objProcessDataPrevious.Algorithm_type) \
                                      + '_Smoothen-' + str(objProcessDataPrevious.isSmoothen) + '_FFT-' + str(
                    objProcessDataPrevious.FFT_type)
                objProcessData = ProcessFaceData(objProcessDataPrevious.Algorithm_type, objProcessDataPrevious.FFT_type,
                                                 Filter_type, Result_type,
                                                 objProcessDataPrevious.Preprocess_type,
                                                 SavePath,
                                                 self.objConfig.ignoregray, objProcessDataPrevious.isSmoothen,
                                                 self.objConfig.GenerateGraphs,
                                                 objProcessDataPrevious.timeinSeconds,
                                                 DumpToDisk, fileName)
                objProcessData.SetFromDataParameters(region, objProcessDataPrevious.IRfpswithTime,
                                                     objProcessDataPrevious.ColorfpswithTime, TotalWindows,
                                                     objProcessDataPrevious.timeinSeconds,
                                                     objProcessDataPrevious.WindowIRfpswithTime,
                                                     objProcessDataPrevious.WindowColorfpswithTime, WindowCount,
                                                     objProcessDataPrevious.ColorEstimatedFPS,
                                                     objProcessDataPrevious.IREstimatedFPS,
                                                     objProcessDataPrevious.regionStore,
                                                     objProcessDataPrevious.regionWindowBlueData,
                                                     objProcessDataPrevious.regionWindowGreenData,
                                                     objProcessDataPrevious.regionWindowRedData,
                                                     objProcessDataPrevious.regionWindowGreyData,
                                                     objProcessDataPrevious.regionWindowIRData,
                                                     objProcessDataPrevious.distanceM, None, None, None, None, None,
                                                     objProcessDataPrevious.timecolorCount,
                                                     objProcessDataPrevious.timeirCount,
                                                     objProcessDataPrevious.Frametime_list_ir,
                                                     objProcessDataPrevious.Frametime_list_color,
                                                     objProcessDataPrevious.Colorfrequency,
                                                     objProcessDataPrevious.IRfrequency
                                                     )
                startLogTime = self.LogTime()  # foreach region
                # Apply FFT
                processedBlue, processedGreen, processedRed, processedGrey, processedIR = objProcessData.FilterTechniques(
                    objProcessData.regionBlueData, objProcessData.regionGreenData,
                    objProcessData.regionRedData, objProcessData.regionGreyData,
                    objProcessData.regionIRData)
                endlogTime = self.LogTime()
                diffTime = endlogTime - startLogTime

                objProcessData.regionWindowBlueData = processedBlue
                objProcessData.regionWindowGreenData = processedGreen
                objProcessData.regionWindowRedData = processedRed
                objProcessData.regionWindowGreyData = processedGrey
                objProcessData.regionWindowIRData = processedIR
                objProcessData.WindowProcessStartTime = startLogTime
                objProcessData.WindowProcessEndTime = endlogTime
                objProcessData.WindowProcessDifferenceTime = diffTime

                WindowRegionList[region] = objProcessData

            elif (ProcessingStep == 'Result'):
                objProcessDataPrevious = objWindowProcessedData.get(region)
                PreviousStepDetails = '_PreProcess-' + str(
                    objProcessDataPrevious.Preprocess_type) + '_Algorithm-' + str(objProcessDataPrevious.Algorithm_type) \
                                      + '_Smoothen-' + str(objProcessDataPrevious.isSmoothen) + '_FFT-' + str(
                    objProcessDataPrevious.FFT_type) + '_Filter-' + str(objProcessDataPrevious.Filter_type)
                objProcessData = ProcessFaceData(objProcessDataPrevious.Algorithm_type, objProcessDataPrevious.FFT_type,
                                                 objProcessDataPrevious.Filter_type, Result_type,
                                                 objProcessDataPrevious.Preprocess_type,
                                                 SavePath,
                                                 self.objConfig.ignoregray, objProcessDataPrevious.isSmoothen,
                                                 self.objConfig.GenerateGraphs,
                                                 objProcessDataPrevious.timeinSeconds,
                                                 DumpToDisk, fileName)
                objProcessData.SetFromDataParameters(region, objProcessDataPrevious.IRfpswithTime,
                                                     objProcessDataPrevious.ColorfpswithTime, TotalWindows,
                                                     objProcessDataPrevious.timeinSeconds,
                                                     objProcessDataPrevious.WindowIRfpswithTime,
                                                     objProcessDataPrevious.WindowColorfpswithTime, WindowCount,
                                                     objProcessDataPrevious.ColorEstimatedFPS,
                                                     objProcessDataPrevious.IREstimatedFPS,
                                                     objProcessDataPrevious.regionStore,
                                                     objProcessDataPrevious.regionWindowBlueData,
                                                     objProcessDataPrevious.regionWindowGreenData,
                                                     objProcessDataPrevious.regionWindowRedData,
                                                     objProcessDataPrevious.regionWindowGreyData,
                                                     objProcessDataPrevious.regionWindowIRData,
                                                     objProcessDataPrevious.distanceM, None, None, None, None, None,
                                                     objProcessDataPrevious.timecolorCount,
                                                     objProcessDataPrevious.timeirCount,
                                                     objProcessDataPrevious.Frametime_list_ir,
                                                     objProcessDataPrevious.Frametime_list_color,
                                                     objProcessDataPrevious.Colorfrequency,
                                                     objProcessDataPrevious.IRfrequency
                                                     )
                startLogTime = self.LogTime()  # foreach region
                # Apply FFT
                objProcessData.generateHeartRateandSNR(
                    objProcessData.regionBlueData, objProcessData.regionGreenData,
                    objProcessData.regionRedData, objProcessData.regionGreyData,
                    objProcessData.regionIRData, Result_type)

                # get best bpm and heart rate period in one region
                objProcessData.bestHeartRateSnr = 0.0
                objProcessData.bestBpm = 0.0
                objProcessData.GetBestBpm()

                endlogTime = self.LogTime()
                diffTime = endlogTime - startLogTime

                objProcessData.WindowProcessStartTime = startLogTime
                objProcessData.WindowProcessEndTime = endlogTime
                objProcessData.WindowProcessDifferenceTime = diffTime

                ########SPO#####################

                ###get original grey, ir and red
                GreyOriginal = None
                RedOriginal = None
                IROriginal = None
                CurrentSavePath = self.objConfig.SavePath
                OriginalPath = CurrentSavePath.replace('ProcessedDatabyProcessType', 'RawOriginal')
                objOriginalUncompressedData = self.objFile.ReadfromDisk(OriginalPath, 'UnCompressedBinaryLoadedData')
                GreyOriginal = objOriginalUncompressedData.ROIStore.get(region).grey  # .regionGreyData
                RedOriginal = objOriginalUncompressedData.ROIStore.get(region).red
                IROriginal = objOriginalUncompressedData.ROIStore.get(region).Irchannel  # regionIRData
                dM = objOriginalUncompressedData.ROIStore.get(region).distanceM
                #TODO:FIX
                if(len(dM) <len(IROriginal)):
                    dmLast = dM[-1]
                    diffdm = len(IROriginal)-len(dM)
                    for x in range(0, diffdm):
                        dM.append(dmLast)

                # Caclulate
                startLogTime = self.LogTime()  # foreach region
                objProcessData.SPOWindowProcessStartTime = startLogTime
                G_Filtered = objProcessData.regionGreenData

                ###Filter
                Gy_filteredCopy = G_Filtered
                greyCopy = GreyOriginal
                # redCopy = red
                if (len(GreyOriginal) > len(IROriginal)):
                    greyCopy = GreyOriginal.copy()
                    lengthDiff = len(greyCopy) - len(IROriginal)
                    for i in range(lengthDiff):
                        greyCopy.pop()
                if (len(G_Filtered) > len(IROriginal)):
                    Gy_filteredCopy = G_Filtered.copy()
                    Gy_filteredCopy = Gy_filteredCopy[0:len(IROriginal)]  # all but the first and last element

                std, err, oxylevl = objProcessData.getSpo(greyCopy, Gy_filteredCopy, IROriginal, RedOriginal,
                                                          dM)  # Irchannel and distanceM as IR channel lengh can be smaller so passing full array
                objProcessData.SPOstd = std
                objProcessData.SPOerr = err
                objProcessData.SPOoxylevl = oxylevl

                endlogTime = self.LogTime()
                diffTime = endlogTime - startLogTime
                objProcessData.SPOWindowProcessEndTime = endlogTime
                objProcessData.SPOWindowProcessDifferenceTime = diffTime

                ######LOG###
                # PreProcess
                path = self.objConfig.SavePath + 'PreProcess_WindowsBinaryFiles\\'
                objLog = self.objFile.ReadfromDisk(path,
                                                   'ResultSignal_PreProcess-' + str(
                                                       objProcessDataPrevious.Preprocess_type)).get(
                    region)
                PreProcessdifferenceTime = objLog.WindowProcessDifferenceTime

                # Algorithm
                path = self.objConfig.SavePath + 'Algorithm_WindowsBinaryFiles\\'
                objLog = self.objFile.ReadfromDisk(path,
                                                   'ResultSignal_Algorithm-' + str(
                                                       objProcessDataPrevious.Algorithm_type) + '_PreProcess-' + str(
                                                       objProcessDataPrevious.Preprocess_type)).get(
                    region)
                AlgorithmdifferenceTime = objLog.WindowProcessDifferenceTime

                # Smooth
                path = self.objConfig.SavePath + 'Smoothen_WindowsBinaryFiles\\'
                objLog = self.objFile.ReadfromDisk(path,
                                                   'ResultSignal_Smoothen-' + str(
                                                       objProcessDataPrevious.isSmoothen) + '_PreProcess-' + str(
                                                       objProcessDataPrevious.Preprocess_type)
                                                   + '_Algorithm-' + str(objProcessDataPrevious.Algorithm_type)).get(
                    region)
                SmoothdifferenceTime = objLog.WindowProcessDifferenceTime

                # FFT
                path = self.objConfig.SavePath + 'FFT_WindowsBinaryFiles\\'
                objLog = self.objFile.ReadfromDisk(path,
                                                   'ResultSignal_FFT-' + str(
                                                       objProcessDataPrevious.FFT_type) + '_PreProcess-' + str(
                                                       objProcessDataPrevious.Preprocess_type)
                                                   + '_Algorithm-' + str(
                                                       objProcessDataPrevious.Algorithm_type) + '_Smoothen-' + str(
                                                       objProcessDataPrevious.isSmoothen)).get(
                    region)
                FFTdifferenceTime = objLog.WindowProcessDifferenceTime

                # Filter
                path = self.objConfig.SavePath + 'Filter_WindowsBinaryFiles\\'
                objLog = self.objFile.ReadfromDisk(path,
                                                   'ResultSignal_Filter-' + str(
                                                       objProcessDataPrevious.Filter_type) + '_PreProcess-' + str(
                                                       objProcessDataPrevious.Preprocess_type) +
                                                   '_Algorithm-' + str(
                                                       objProcessDataPrevious.Algorithm_type) + '_Smoothen-' + str(
                                                       objProcessDataPrevious.isSmoothen) +
                                                   '_FFT-' + str(objProcessDataPrevious.FFT_type)).get(
                    region)
                FilterdifferenceTime = objLog.WindowProcessDifferenceTime

                TotalTime = PreProcessdifferenceTime + AlgorithmdifferenceTime + SmoothdifferenceTime + FFTdifferenceTime + FilterdifferenceTime + \
                            objProcessData.WindowProcessDifferenceTime + objProcessData.SPOWindowProcessDifferenceTime
                FullLog = 'TotalWindowCalculationTime: ' + str(TotalTime) + ' ,\t' \
                          + 'PreProcess: ' + str(PreProcessdifferenceTime) + ' ,\t' \
                          + 'Algorithm: ' + str(AlgorithmdifferenceTime) + ' ,\t' \
                          + 'FFT: ' + str(FFTdifferenceTime) + ' ,\t' \
                          + 'Smooth: ' + str(SmoothdifferenceTime) + ' ,\t' \
                          + 'Filter: ' + str(FilterdifferenceTime) + ' ,\t' \
                          + 'ComputerHRSNR: ' + str(objProcessData.WindowProcessDifferenceTime) + ' ,\t' \
                          + 'ComputerSPO: ' + str(objProcessData.SPOWindowProcessDifferenceTime)
                objProcessData.FullLog = FullLog
                WindowRegionList[region] = objProcessData

            del objProcessData

        # Dumpt binary Store to disk
        self.objFile.DumpObjecttoDisk(SavePath + ProcessingStep + '_WindowsBinaryFiles' + '\\',
                                      fileName + PreviousStepDetails, WindowRegionList)  # + str(WindowCount)

    def RunDatabyProcessingType(self, data, case, ProcessingStep):
        # IsGenerated = self.CheckIfGenerated(case) #TODO: Ccheck
        if (ProcessingStep == 'PreProcess'):
            # IsGenerated = self.CheckIfGenerated(case)
            # Generate Data for all Techniques
            self.Process_Participants_Data_EntireSignalINChunks(
                data, self.objConfig.SavePath,
                self.objConfig.DumpToDisk,
                ProcessingStep, case, self.objConfig.roiregions)
        else:
            objdata = data.copy()
            # for k, v in objdata.items():
            # IsGenerated = self.CheckIfGenerated(case)
            # Generate Data for all Techniques
            self.Process_Participants_Data_EntireSignalINChunks(
                objdata, self.objConfig.SavePath,
                self.objConfig.DumpToDisk,
                ProcessingStep, case, self.objConfig.roiregions)

    """
     GenerateResultsfromParticipants:
     """

    def GenerateResultsfromParticipants(self, objProcessedDataOrginal, typeProcessing, ProcessingStep, ProcessCase,
                                        participant_number, position, fileName):  # ParticipantsOriginalDATA
        TotalCasesCount = len(ProcessCase)
        # for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
        #     for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
        # print(participant_number + ', ' + position)
        # objProcessedDataOrginal = ParticipantsOriginalDATA.get(participant_number + '_' + position)
        self.objConfig.setSavePath(participant_number, position, typeProcessing)
        currentCasesDone = 0
        for case in ProcessCase:

            # Run
            newFileName= fileName
            pathexists = False
            if(ProcessingStep == 'Result'):
                splitedFileName = fileName.split('_')
                newFileName = splitedFileName[0] + '_' + ProcessingStep + '-' + str(case) + '_' \
                              + splitedFileName[2] + '_' + splitedFileName[3] + '_' + \
                              splitedFileName[4] + '_' + splitedFileName[5] + '_' + splitedFileName[1]
                pathexists = self.objFile.FileExits(self.objConfig.SavePath + ProcessingStep + '_WindowsBinaryFiles\\' + newFileName)

            if (pathexists):
                skip = 0
            else:
                currentCasesDone = currentCasesDone + 1
                currentPercentage = round((currentCasesDone / TotalCasesCount) * 100)
                currentPercentage = str(currentPercentage)
                print(str(case) + "  -> " + str(currentPercentage) + " out of 100%")

                self.RunDatabyProcessingType(objProcessedDataOrginal, case, ProcessingStep)

            # ParticipantsOriginalDATA.pop(participant_number + '_' + position)

    '''
    LoadBinaryData: load data from disk ParticipantsOriginalDATA[ParticipantNumber + Position] -> ROISTORE data
    '''

    def LoadBinaryData(self):
        ParticipantsOriginalDATA = {}
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                self.objConfig.setSavePath(participant_number, position, 'RawOriginal')  # set path
                print('Loading and generating FaceData for ' + participant_number + ', ' + position)
                print('Loading from path ' + self.objConfig.SavePath)

                ##Original Data in binary read from disk
                objWindowProcessedData = self.objFile.ReadfromDisk(self.objConfig.SavePath,
                                                                   'UnCompressedBinaryLoadedData')

                # Store for procesing locally
                ParticipantsOriginalDATA[participant_number + '_' + position] = objWindowProcessedData

        return ParticipantsOriginalDATA

    '''
     LoadPartiallyProcessedBinaryData: load data from disk ParticipantsPartiallyProcessedBinaryData[ParticipantNumber + Position] -> ROISTORE data
     '''

    def LoadPartiallyProcessedBinaryData(self, LoadFolder, FolderNameforSave, ProcessingStep,
                                         ProcessCase):

        CaseList, LoadfNameList =  self.GenerateCases()

        # CaseList, LoadfNameList = self.GenerateCases()
        # ParticipantsPartiallyProcessedBinaryData = {}
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                self.objConfig.setSavePath(participant_number, position, 'ProcessedDatabyProcessType')  # set path
                print('Loading and generating FaceData for ' + participant_number + ', ' + position)

                LoadPath = self.objConfig.DiskPath + FolderNameforSave + '\\' + participant_number + '\\' + position + '\\' +LoadFolder+ '\\'
                if(ProcessingStep == 'PreProcess'):
                    LoadPath = self.objConfig.DiskPath + LoadFolder+ '\\' + participant_number + '\\' + position+'\\'
                print('Loading from path ' + LoadPath)
                # if(ProcessingStep == "Result"):
                #     AllFileNames = CaseList  # os.listdir(LoadPath)  # 'ResultSignal_Type-' + str(Preprocess_type)
                # else:
                AllFileNames = os.listdir(LoadPath)  # 'ResultSignal_Type-' + str(Preprocess_type)

                ListProcessedData = {}
                fileCount = 0
                for fileName in AllFileNames:
                    # fileNameSplit = fileName.split('-')
                    # if generated already skip
                    # ProcessedDataType = fileName.replace('ResultSignal_', '')
                    ##Data in binary read from disk
                    if(ProcessingStep =='PreProcess'):
                        fileName = 'UnCompressedBinaryLoadedData'
                    objWindowProcessedData = self.objFile.ReadfromDisk(LoadPath, fileName)
                    ####
                    print(str(fileCount) + ') Applying on data that has been ' + fileName)
                    fileCount = fileCount + 1
                    self.GenerateResultsfromParticipants(objWindowProcessedData, FolderNameforSave, ProcessingStep,
                                                         ProcessCase, participant_number, position,
                                                         fileName)  # FOR Window processing
                    # ListProcessedData[ProcessedDataType] = objWindowProcessedData
                    # Store for procesing locally
                # ParticipantsPartiallyProcessedBinaryData[participant_number + '_' + position] = ListProcessedData

        # return ParticipantsPartiallyProcessedBinaryData

    def mainMethod(self, ProcessingStep):
        # Process for entire signal
        FolderNameforSave = 'ProcessedDatabyProcessType'
        print(FolderNameforSave)
        #  Load Data from path and Process Data
        LoadedData = None
        ProcessCase = None
        LoadFolderName = ProcessingStep + '_WindowsBinaryFiles'
        ##Process by type
        if (ProcessingStep == 'PreProcess'):
            ProcessCase = self.objConfig.preprocesses
            LoadedData = self.LoadBinaryData()
            LoadedData = self.LoadPartiallyProcessedBinaryData('RawOriginal', FolderNameforSave,
                                                               ProcessingStep,
                                                               ProcessCase)
        elif (ProcessingStep == 'Algorithm'):  # TODO: Run 3times, 5 times ica and so on PLOT diference
            ProcessCase = self.objConfig.AlgoList
            LoadedData = self.LoadPartiallyProcessedBinaryData('PreProcess_WindowsBinaryFiles', FolderNameforSave,
                                                               ProcessingStep,
                                                               ProcessCase)
        elif (ProcessingStep == 'Smoothen'):
            ProcessCase = self.objConfig.Smoothen
            LoadedData = self.LoadPartiallyProcessedBinaryData('Algorithm_WindowsBinaryFiles', FolderNameforSave,
                                                               ProcessingStep,
                                                               ProcessCase)
        elif (ProcessingStep == 'FFT'):
            ProcessCase = self.objConfig.fftTypeList
            LoadedData = self.LoadPartiallyProcessedBinaryData('Smoothen_WindowsBinaryFiles', FolderNameforSave,
                                                               ProcessingStep,
                                                               ProcessCase)
        elif (ProcessingStep == 'Filter'):
            ProcessCase = self.objConfig.filtertypeList
            LoadedData = self.LoadPartiallyProcessedBinaryData('FFT_WindowsBinaryFiles', FolderNameforSave,
                                                               ProcessingStep,
                                                               ProcessCase)
        elif (ProcessingStep == 'Result'):
            ProcessCase = self.objConfig.resulttypeList
            LoadedData = self.LoadPartiallyProcessedBinaryData('Filter_WindowsBinaryFiles', FolderNameforSave,
                                                               ProcessingStep,
                                                               ProcessCase)

        # if (ProcessingStep == "ComputeFinalResults"):
            # self.Process_Participants_Result_forEntireSignal('Result_WindowsBinaryFiles')
#a

###RUN this file CODE###
# skintype = 'SouthAsian_BrownSkin_Group'#'Europe_WhiteSkin_Group'
# print('Program started for ' + skintype)
# objInitiateProcessing = InitiateProcessingStorage()#skintype
# objInitiateProcessing.GenerateOriginalRawData()# Only do once
# objInitiateProcessing.mainMethod('PreProcess') #Completed
# objInitiateProcessing.mainMethod('Algorithm')  # Completed
# objInitiateProcessing.mainMethod('Smoothen')  # Completed
# objInitiateProcessing.mainMethod('FFT')  # Completed
# print('FFT Complteted')
# objInitiateProcessing.mainMethod('Filter')  # Completed
# print('Filter Complteted')
# objInitiateProcessing.mainMethod('Result') # restart from PIS-6327
# print('Result Complteted')
# objInitiateProcessing.mainMethod('ComputeFinalResults')  # to process
# print('ComputeFinalResults Complteted')  # TODO:GENERATE GRAPHS and object and other diagrams , class from code
# print('Program completed')
# objInitiateProcessing.ProduceFinalResult() #OLD METHOD
