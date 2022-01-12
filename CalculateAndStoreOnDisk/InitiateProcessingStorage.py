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

'''
InitiateProcessingStorage:
'''


class InitiateProcessingStorage:
    # Global Objects
    objConfig = None
    ProcessedDataPath = None
    objFile = None

    # Constructor
    def __init__(self, skinGroup):
        self.objConfig = Configurations(False, skinGroup)
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
                        dataContent = objInitiateProcessing.ReadData(fileName)
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
                            dataContent = objInitiateProcessing.ReadData(fileName)
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
                                    dataContent = objInitiateProcessing.ReadData(fileName)
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

    '''
       Process_Participants_Data_EntireSignalINChunks:
       '''

    def Process_Participants_Result_forEntireSignal(self, typeProcessing):
        self.objConfig.ParticipantNumbers = ["PIS-8073"]
        self.objConfig.hearratestatus = ["Resting1"]
        CaseList, LoadfNameList = self.GenerateCases()
        TotalCasesCount = len(CaseList)
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                print(participant_number + ', ' + position)
                self.objConfig.setSavePath(participant_number, position, 'ProcessedDatabyProcessType')
                SavePath = self.objConfig.SavePath
                currentCasesDone = 0

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
                SPOstd=0.0
                FullTimeLog = None
                WindowCount = 0
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
                WindowRegionList = {}
                for fileName in CaseList:

                    currentCasesDone = currentCasesDone + 1
                    currentPercentage = round((currentCasesDone / TotalCasesCount) * 100)
                    currentPercentage = str(currentPercentage)
                    print(str(fileName) + "  -> " + str(currentPercentage) + " out of 100%")

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
                            smallestOxygenError = v.SPOerr#oxygenSaturationValueError
                            finaloxy = v.SPOoxylevl#oxygenSaturationValueValue
                            SPOstd = v.SPOstd#oxygenSaturationValueValue

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
                    fileNameResult = "HRSPOwithLog_" + fileName
                    #          regiontype + "_" + Algorithm_type + "_FFT-" + str(FFT_type) + "_FL-" + str(
                    # Filter_type) + "_RS-" + str(Result_type) + "_PR-" + str(Preprocess_type) + "_SM-" + str(
                    # isSmoothen)
                    # Write data to file
                    self.objFile.WriteListDatatoFile(SavePath + 'ComputedFinalResult\\', fileNameResult, Listdata)

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
                                                   'ResultSignal_PreProcess-' + str(objProcessDataPrevious.Preprocess_type)).get(
                    region)
                PreProcessdifferenceTime = objLog.WindowProcessDifferenceTime

                # Algorithm
                path = self.objConfig.SavePath + 'Algorithm_WindowsBinaryFiles\\'
                objLog = self.objFile.ReadfromDisk(path,
                                                   'ResultSignal_Algorithm-' + str(objProcessDataPrevious.Algorithm_type) + '_PreProcess-' +str( objProcessDataPrevious.Preprocess_type)).get(
                    region)
                AlgorithmdifferenceTime = objLog.WindowProcessDifferenceTime

                # Smooth
                path = self.objConfig.SavePath + 'Smoothen_WindowsBinaryFiles\\'
                objLog = self.objFile.ReadfromDisk(path,
                                                   'ResultSignal_Smoothen-' + str(objProcessDataPrevious.isSmoothen )+ '_PreProcess-' + str(objProcessDataPrevious.Preprocess_type )
                                                   + '_Algorithm-' + str(objProcessDataPrevious.Algorithm_type)).get(
                    region)
                SmoothdifferenceTime = objLog.WindowProcessDifferenceTime

                # FFT
                path = self.objConfig.SavePath + 'FFT_WindowsBinaryFiles\\'
                objLog = self.objFile.ReadfromDisk(path,
                                                   'ResultSignal_FFT-' + str(objProcessDataPrevious.FFT_type) + '_PreProcess-' + str(objProcessDataPrevious.Preprocess_type)
                                                   + '_Algorithm-' + str(objProcessDataPrevious.Algorithm_type) + '_Smoothen-' + str(objProcessDataPrevious.isSmoothen)).get(
                    region)
                FFTdifferenceTime = objLog.WindowProcessDifferenceTime

                # Filter
                path = self.objConfig.SavePath + 'Filter_WindowsBinaryFiles\\'
                objLog = self.objFile.ReadfromDisk(path,
                                                   'ResultSignal_Filter-' + str(objProcessDataPrevious.Filter_type) + '_PreProcess-' + str(objProcessDataPrevious.Preprocess_type) +
                                                   '_Algorithm-' + str(objProcessDataPrevious.Algorithm_type) + '_Smoothen-' + str(objProcessDataPrevious.isSmoothen) +
                                                   '_FFT-' + str(objProcessDataPrevious.FFT_type)).get(
                    region)
                FilterdifferenceTime = objLog.WindowProcessDifferenceTime

                TotalTime = PreProcessdifferenceTime + AlgorithmdifferenceTime + SmoothdifferenceTime + FFTdifferenceTime +FilterdifferenceTime +\
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
            for k, v in objdata.items():
                # IsGenerated = self.CheckIfGenerated(case)
                # Generate Data for all Techniques
                self.Process_Participants_Data_EntireSignalINChunks(
                    v, self.objConfig.SavePath,
                    self.objConfig.DumpToDisk,
                    ProcessingStep, case, self.objConfig.roiregions)

    """
     GenerateResultsfromParticipants:
     """

    def GenerateResultsfromParticipants(self, ParticipantsOriginalDATA, typeProcessing, ProcessingStep, ProcessCase):
        TotalCasesCount = len(ProcessCase)
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                print(participant_number + ', ' + position)
                objProcessedDataOrginal = ParticipantsOriginalDATA.get(participant_number + '_' + position)
                self.objConfig.setSavePath(participant_number, position, typeProcessing)
                currentCasesDone = 0
                for case in ProcessCase:
                    currentCasesDone = currentCasesDone + 1
                    currentPercentage = round((currentCasesDone / TotalCasesCount) * 100)
                    currentPercentage = str(currentPercentage)
                    print(str(case) + "  -> " + str(currentPercentage) + " out of 100%")
                    # Run
                    self.RunDatabyProcessingType(objProcessedDataOrginal, case, ProcessingStep)

                ParticipantsOriginalDATA.pop(participant_number + '_' + position)

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

    def LoadPartiallyProcessedBinaryData(self, LoadFolder):
        ParticipantsPartiallyProcessedBinaryData = {}
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                self.objConfig.setSavePath(participant_number, position, 'ProcessedDatabyProcessType')  # set path
                print('Loading and generating FaceData for ' + participant_number + ', ' + position)
                LoadPath = self.objConfig.SavePath + LoadFolder + '\\'
                print('Loading from path ' + LoadPath)
                AllFileNames = os.listdir(LoadPath)  # 'ResultSignal_Type-' + str(Preprocess_type)
                ListProcessedData = {}
                for fileName in AllFileNames:
                    # fileNameSplit = fileName.split('-')
                    ProcessedDataType = fileName.replace('ResultSignal_', '')
                    ##Data in binary read from disk
                    objWindowProcessedData = self.objFile.ReadfromDisk(LoadPath, fileName)
                    ListProcessedData[ProcessedDataType] = objWindowProcessedData
                    # Store for procesing locally
                ParticipantsPartiallyProcessedBinaryData[participant_number + '_' + position] = ListProcessedData

        return ParticipantsPartiallyProcessedBinaryData

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
        elif (ProcessingStep == 'Algorithm'):  # TODO: Run 3times, 5 times ica and so on PLOT diference
            LoadedData = self.LoadPartiallyProcessedBinaryData('PreProcess_WindowsBinaryFiles')
            ProcessCase = self.objConfig.AlgoList
        elif (ProcessingStep == 'Smoothen'):
            LoadedData = self.LoadPartiallyProcessedBinaryData('Algorithm_WindowsBinaryFiles')
            ProcessCase = self.objConfig.Smoothen
        elif (ProcessingStep == 'FFT'):
            ProcessCase = self.objConfig.fftTypeList
            LoadedData = self.LoadPartiallyProcessedBinaryData('Smoothen_WindowsBinaryFiles')
        elif (ProcessingStep == 'Filter'):
            ProcessCase = self.objConfig.filtertypeList
            LoadedData = self.LoadPartiallyProcessedBinaryData('FFT_WindowsBinaryFiles')
        elif (ProcessingStep == 'Result'):
            ProcessCase = self.objConfig.resulttypeList
            LoadedData = self.LoadPartiallyProcessedBinaryData('Filter_WindowsBinaryFiles')

        if (ProcessingStep == "ComputeFinalResults"):
            self.Process_Participants_Result_forEntireSignal('Result_WindowsBinaryFiles')
        else:
            self.GenerateResultsfromParticipants(LoadedData, FolderNameforSave, ProcessingStep,
                                                 ProcessCase)  # FOR Window processing


###RUN this file CODE###
skintype = 'Europe_WhiteSkin_Group'
print('Program started for ' + skintype)
objInitiateProcessing = InitiateProcessingStorage(skintype)
# objInitiateProcessing.GenerateOriginalRawData()# Only do once
# objInitiateProcessing.mainMethod('PreProcess') #Completed
# objInitiateProcessing.mainMethod('Algorithm')  # Completed
# objInitiateProcessing.mainMethod('Smoothen')  # Completed
objInitiateProcessing.mainMethod('FFT')  # to process
print('FFT Complteted')
objInitiateProcessing.mainMethod('Filter')  # to process
print('Filter Complteted')
objInitiateProcessing.mainMethod('Result')  # to process
print('Result Complteted')
objInitiateProcessing.mainMethod('ComputeFinalResults')  # to process
print('ComputeFinalResults Complteted')
print('Program completed')
# objInitiateProcessing.ProduceFinalResult() #OLD METHOD
