import os
import sys
from datetime import datetime

import CommonMethods
from CalculateAndStoreOnDisk.ProcessDataPartial import ProcessDataPartial
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

    #Constructor
    def __init__(self, skinGroup):
        self.objConfig = Configurations(False,skinGroup)
        self.objFile = FileIO()

    def WritetoDisk(self,location, filename, data):
        ##STORE Data
        with open(location + filename, 'wb') as filehandle:
            pickle.dump(data, filehandle)

    def ReadfromDisk(self,location, filename):
        ##Read data
        with open(location + filename, 'rb') as filehandle:
            data = pickle.load(filehandle)
        return data

    def GenerateOriginalRawData(self,participant_number,position,objFaceImage,region): #TODO: CHANGE AS PER NEW CHANGES OTHER WISE WILL GIVE ERROR
        ##get loading path
        LoadColordataPath, LoadIRdataPath, LoadDistancePath, self.ProcessedDataPath = self.objConfig.getLoadPath(participant_number,
                                                                                         position,
                                                                                         region)
        # Load Roi data (ALL)
        # print("Loading and processing color roi data")
        objFaceImage.ProcessColorImagestoArray(LoadColordataPath)

        # print("Loading and processing ir roi data")
        objFaceImage.ProcessIRImagestoArray(LoadIRdataPath)

        # GET FPS and distance and other data
        ColorfpswithTime, IRfpswithTime, isVariable, isIRVariable, ColorFPS, IRFPS = objFaceImage.GetEstimatedFPS(LoadDistancePath)

        # Create global data object and use dictionary (ROI Store) to uniquely store a regions data
        #Store data to disk
        self.WritetoDisk(self.ProcessedDataPath,'objFaceImage_'+region,objFaceImage) # self.WritetoDisk(ProcessedDataPath,'redChannel',objFaceImage.red)
        self.GenerateGraph(objFaceImage,self.ProcessedDataPath,region)

    def ReadOriginalRawData(self,region):
        #Load from disk
        objFaceImage = self.ReadfromDisk(self.ProcessedDataPath,'objFaceImage_'+region)
        return objFaceImage

    def ReadData(self,name):
        #Load from disk
        dataFile = self.ReadfromDisk(self.ProcessedDataPath,name)
        return dataFile
    '''
    Process participants data over the entire signal data
    '''
    def GenerateGraph(self,objFaceImage,ProcessedDataPath,region):

        # Initialise object to process face regions
        objProcessData = ProcessDataPartial(objFaceImage.ColorEstimatedFPS, objFaceImage.IREstimatedFPS,
                                            ProcessedDataPath)
        # Generate graph
        objProcessData.GenerateGrapth("RawData_" + region, objFaceImage.blue, objFaceImage.green, objFaceImage.red,
                                      objFaceImage.grey, objFaceImage.Irchannel)
        #clear
        del objProcessData

    '''
       Process participants data over the entire signal data
       '''
    def PreProcessData(self, blue,green,red,grey,Irchannel,ColorEstimatedFPS,IREstimatedFPS,region,timecolorCount,timeirCount):
        # Initialise object to process face regions
        objProcessData = ProcessDataPartial(ColorEstimatedFPS, IREstimatedFPS,self.ProcessedDataPath)
        for Preprocess_type in self.objConfig.preprocesses:
            # Generate data
            objProcessData.PreProcessData(blue, green, red, grey, Irchannel, self.objConfig.GenerateGraphs, region, Preprocess_type,timecolorCount,timeirCount)
        # clear
        del objProcessData

    '''
       Process participants data over the entire signal data
       '''
    def ApplyAlgorithm(self, SCombined,ColorEstimatedFPS,IREstimatedFPS,region,preProcessType):
        # Initialise object to process face regions
        objProcessData = ProcessDataPartial(ColorEstimatedFPS, IREstimatedFPS,self.ProcessedDataPath)
        for algo_type in self.objConfig.AlgoList:
            # Generate data
            objProcessData.ApplyAlgorithm(SCombined,algo_type,5,self.objConfig.GenerateGraphs,region,preProcessType)
        # clear
        del objProcessData

    '''
       Process participants data over the entire signal data
       '''
    def ApplySmooth(self, SCombined,ColorEstimatedFPS,IREstimatedFPS,region,preProcessType,Algotype):
        # Initialise object to process face regions
        objProcessData = ProcessDataPartial(ColorEstimatedFPS, IREstimatedFPS,self.ProcessedDataPath)
        # Generate data
        objProcessData.SmoothenData( SCombined,self.objConfig.GenerateGraphs,region,preProcessType,Algotype)
        # clear
        del objProcessData

    '''
       Process participants data over the entire signal data
       '''
    def ApplyFFT(self, SCombined,ColorEstimatedFPS,IREstimatedFPS,region,preProcessType,Algotype, IsSmooth):
        # Initialise object to process face regions
        objProcessData = ProcessDataPartial(ColorEstimatedFPS, IREstimatedFPS,self.ProcessedDataPath)
        for fft_type in self.objConfig.fftTypeList:
            # Generate data
            objProcessData.ApplyFFT( SCombined,fft_type,region,self.objConfig.GenerateGraphs,IsSmooth , Algotype, preProcessType)
        # clear
        del objProcessData

    '''
          Process participants data over the entire signal data
          '''
    def ApplyFilter(self, ProcessedSignalData, ColorEstimatedFPS, IREstimatedFPS,
                    region, preProcessType, Algotype, IsSmooth,FFT_type,
                    Colorfrequency, IRfrequency, ignore_freq_below, ignore_freq_above):
        # Initialise object to process face regions
        objProcessData = ProcessDataPartial(ColorEstimatedFPS, IREstimatedFPS, self.ProcessedDataPath)
        for filtertype in self.objConfig.filtertypeList:
            # Generate data
            objProcessData.FilterTechniques(ProcessedSignalData,region, self.objConfig.GenerateGraphs, IsSmooth, Algotype,
                             preProcessType,
                             FFT_type, filtertype, Colorfrequency, IRfrequency, ignore_freq_below, ignore_freq_above)
        # clear
        del objProcessData


    '''
    Process participants data over the entire signal data
    '''
    def ComputerResultData(self, B_filtered, G_filtered, R_filtered, Gy_filtered, IR_filtered, ColorEstimatedFPS, IREstimatedFPS,
                    region, preProcessType, Algotype, IsSmooth,FFT_type,filterType,
                    Colorfrequency, IRfrequency, ignore_freq_below_bpm, ignore_freq_above_bpm):
        # Initialise object to process face regions
        objProcessData = ProcessDataPartial(ColorEstimatedFPS, IREstimatedFPS, self.ProcessedDataPath)
        for resultType in self.objConfig.resulttypeList:
            ##If esitss
            ResultFilesAlreadyGenerated = "ResultType_RS-" + str(resultType) + "_Filtered_FL-" + str(filterType) + "_" + region + \
                                          "_FFTtype-" + str(FFT_type)+ "_algotype-" + str(Algotype) + '_PreProcessType-' + str(preProcessType) + \
                                          "_Smoothed-" + str(IsSmooth)
            if not os.path.exists(self.ProcessedDataPath + ResultFilesAlreadyGenerated):
                # '_PreProcessType-'+str(preProcessType)+ "_Smoothed-" + str(IsSmooth)
                # Generate data
                objProcessData.generateHeartRateandSNR(B_filtered, G_filtered, R_filtered, Gy_filtered, IR_filtered, resultType,
                                    Colorfrequency,IRfrequency,ColorEstimatedFPS,IREstimatedFPS,
                                    preProcessType, Algotype, IsSmooth, FFT_type, filterType,
                                    ignore_freq_below_bpm, ignore_freq_above_bpm,region)
        # clear
        del objProcessData


    def getFileName(self, typeList,region,ProcesstypeName):
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
                fileName = 'AlgorithmData_'+ region + "_type-"+ str(algo_type) + '_PreProcessType-'+str(preprocess_type)
                if(fileName not in FileNames):
                    FileNames.append(fileName)
        return FileNames

    def getPreProcessedAlgorithmsFFT(self, region,isSmoothed):
        FileNames = []
        fileConent = {}
        for preprocess_type in self.objConfig.preprocesses:
            for algo_type in self.objConfig.AlgoList:
                for fft_type in self.objConfig.fftTypeList:
                    fileName = "FFT-" + fft_type + "_" + region + "_Smoothed-" + str(isSmoothed) + "_Algotype-" + str(algo_type) + '_PreProcessType-' + str(preprocess_type)

                    if(fileName not in FileNames):
                        FileNames.append(fileName)
                        ###
                        dataContent = objInitiateProcessing.ReadData(fileName)
                        fileConent[fileName] = dataContent

        FileNames = []
        return fileConent

    def getPreProcessedFiltered(self, region,isSmoothed):

        FileNames = []
        fileConent = {}
        for preprocess_type in self.objConfig.preprocesses:
            for algo_type in self.objConfig.AlgoList:
                for fft_type in self.objConfig.fftTypeList:
                    for filterType in self.objConfig.filtertypeList:
                        fileName = "Filtered_FL-"+ str(filterType)+ "_"+ region+ "_FFTtype-" + str(fft_type)  + "_algotype-" + str(algo_type) +'_PreProcessType-'+str(preprocess_type)+ "_Smoothed-" + str(isSmoothed)
                        if(fileName not in FileNames):
                            FileNames.append(fileName)
                            dataContent = objInitiateProcessing.ReadData(fileName)
                            fileConent[fileName] = dataContent
        FileNames = []
        return fileConent

    def getResult(self,region,participant_number,position):
        FileNames = []
        fileConent = {}
        for preprocess_type in self.objConfig.preprocesses:
            for algo_type in self.objConfig.AlgoList:
                for fft_type in self.objConfig.fftTypeList:
                    for filterType in self.objConfig.filtertypeList:
                        for resulttype in self.objConfig.resulttypeList:
                            for isSmoothed in self.objConfig.Smoothen:
                                fileName = "ResultType_RS-" + str(resulttype) + "_Filtered_FL-"+ str(filterType)+ "_"+ region+ "_FFTtype-" + str(fft_type)  \
                                           + "_algotype-" + str(algo_type) +\
                                           '_PreProcessType-'+str(preprocess_type)+ "_Smoothed-" + str(isSmoothed)
                                if(fileName not in FileNames):
                                    FileNames.append(fileName)
                                    if(isSmoothed == False):
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
                fileName = 'SmoothedData_'+ region + "_Algotype-"+ str(algo_type) + '_PreProcessType-'+str(preprocess_type)
                if(fileName not in FileNames):
                    FileNames.append(fileName)
        return FileNames

    def GenerateCasesNewMethod(self,participant_number,position):
        CaseList = []
        self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'FinalComputedResult\\'
        for preprocesstype in self.objConfig.preprocesses:
            for algoType in self.objConfig.AlgoList:
                for fftype in self.objConfig.fftTypeList:
                    for resulttype in self.objConfig.resulttypeList:
                        for filtertype in self.objConfig.filtertypeList:
                            for isSmooth in self.objConfig.Smoothen:
                                fileName = "ResultType_RS-" + str(resulttype) + "_Filtered_FL-"+ str(filtertype)+ "_"+ "REGION"+ "_FFTtype-" + str(fftype)  + "_algotype-" + str(algoType) +\
                                           '_PreProcessType-'+str(preprocesstype)+ "_Smoothed-" + str(isSmooth)
                                fName = fileName.replace('_REGION_','') + '.txt'
                                if not os.path.exists(self.ProcessedDataPath +fName ):
                                    # print(fileName)
                                    CaseList.append(fileName)
        print('Files Loaded')
        return CaseList

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
                lips_DataFileNameWithContent = self.getResult("lips",participant_number,position)
                forehead_DataFileNameWithContent = self.getResult("forehead",participant_number,position)
                leftcheek_DataFileNameWithContent = self.getResult("leftcheek",participant_number,position)
                rightcheek_DataFileNameWithContent = self.getResult("rightcheek",participant_number,position)
                self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'RawOriginal\\'
                lipsobjFaceImage = self.ReadOriginalRawData('lips')
                self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'FinalComputedResult\\'
                caselist = self.GenerateCasesNewMethod(participant_number,position)
                # self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'ResultDataSmoothed\\'
                HrGr, SpoGr = CommonMethods.GetGroundTruth(participant_number, position,
                                                           self.objConfig.DiskPath, int(lipsobjFaceImage.totalTimeinSeconds))
                self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'FinalComputedResult\\'
                for filename in caselist:
                    saveFilename = filename.replace('_REGION_','') ##IF EXISTS
                    lipsFileName = filename.replace('REGION', 'lips')
                    foreheadFileName = filename.replace('REGION', 'forehead')
                    leftcheekFileName = filename.replace('REGION', 'leftcheek')
                    rightcheekFileName = filename.replace('REGION', 'rightcheek')
                    lipsContent = lips_DataFileNameWithContent.get(lipsFileName)
                    foreheadContent = forehead_DataFileNameWithContent.get(foreheadFileName)
                    leftcheekContent = leftcheek_DataFileNameWithContent.get(leftcheekFileName)
                    rightcheekContent = rightcheek_DataFileNameWithContent.get(rightcheekFileName)
                    Process_Participants_Data_GetBestHR(lipsContent,foreheadContent,rightcheekContent,leftcheekContent,self.ProcessedDataPath,HrGr,saveFilename)

    def LogTime(self):
        logTime = datetime(datetime.now().year, datetime.now().month, datetime.now().day,
                           datetime.now().time().hour, datetime.now().time().minute,
                           datetime.now().time().second, datetime.now().time().microsecond)
        return logTime

    def Process_Participants_Data_EntireSignalINChunks(self,objWindowProcessedData, SavePath, DumpToDisk,
                                                       ProcessingStep,
                                                       ProcessingType,regions):
        # ROI Window Result list
        WindowRegionList = {}

        # Loop through signal data
        WindowCount = 0
        Preprocess_type = 0
        Algorithm_type = 0
        FFT_type = 0
        Filter_type = 0
        isSmoothen = 0
        fileName = 'ResultSignal_Type-'+ str(Preprocess_type)

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

        TotalWindows = 1


        for region in regions:

            if (ProcessingStep == 'PreProcess'):
                # Windows for regions (should be same for all)
                ROIStore=objWindowProcessedData.ROIStore
                TimeinSeconds = ROIStore.get("lips").totalTimeinSeconds
                timeinSeconds = TimeinSeconds
                objProcessData = ProcessFaceData(Algorithm_type, FFT_type, Filter_type, 0, Preprocess_type,
                                                 SavePath,
                                                 self.objConfig.ignoregray, isSmoothen, self.objConfig.GenerateGraphs,
                                                 timeinSeconds,
                                                 DumpToDisk, fileName)
                objProcessData.getSingalDataWindow(ROIStore, self.objConfig.roiregions[0], WindowCount, TotalWindows,
                                                   timeinSeconds)
                # Log Start Time  signal data
                startLogTime = self.LogTime()  # foreach region
                # PreProcess Signal
                processedBlue, processedGreen, processedRed, processedGrey, processedIR = objProcessData.preprocessSignalData(
                    objProcessData.regionBlueData, objProcessData.regionGreenData,
                    objProcessData.regionRedData, objProcessData.regionGreyData,
                    objProcessData.regionIRData)

                endlogTime = self.LogTime()
                diffTime = endlogTime - startLogTime
                #Record
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
                objProcessData = objWindowProcessedData
                startLogTime = self.LogTime()  # foreach region
                # Apply Algorithm
                processedBlue, processedGreen, processedRed, processedGrey, processedIR = objProcessData.ApplyAlgorithm(
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

            elif (ProcessingStep == 'Smoothen'):
                objProcessData = objWindowProcessedData
                startLogTime = self.LogTime()  # foreach region
                # Smooth Signal
                processedBlue = objProcessData.SmoothenData(objProcessData.regionBlueData)
                processedGreen = objProcessData.SmoothenData(objProcessData.regionGreenData)
                processedRed = objProcessData.SmoothenData(objProcessData.regionRedData)
                processedGrey = objProcessData.SmoothenData(objProcessData.regionGreyData)
                processedIR = objProcessData.SmoothenData(objProcessData.regionIRData)
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
                objProcessData = objWindowProcessedData
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
                objProcessData = objWindowProcessedData
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

        #Dumpt binary Store to disk
        self.objFile.DumpObjecttoDisk(SavePath + ProcessingStep + '_WindowsBinaryFiles' + '\\', fileName, WindowRegionList)#+ str(WindowCount)

        del objProcessData

    """
     GenerateResultsfromParticipants:
     """
    def GenerateResultsfromParticipants(self, ParticipantsOriginalDATA,typeProcessing,ProcessingStep,ProcessCase):
        TotalCasesCount = len(ProcessCase)
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                print(participant_number + ', ' + position)
                # ParticipantsOriginalDATA[participant_number + '_' + position] = objWindowProcessedData
                objWindowProcessedData= None
                if (ProcessingStep is 'PreProcess'):
                    objWindowProcessedData = ParticipantsOriginalDATA.get(participant_number + '_' + position)

                self.objConfig.setSavePath(participant_number, position,typeProcessing)
                currentCasesDone = 0
                for case in ProcessCase:
                    if (ProcessingStep is not 'PreProcess'):
                        objWindowProcessedData = ParticipantsOriginalDATA.get(participant_number + '_' + position+ '_' + str(case))
                    # IsGenerated = self.CheckIfGenerated(case) #TODO: Ccheck
                    currentCasesDone = currentCasesDone + 1
                    currentPercentage = ((currentCasesDone/TotalCasesCount)*100)
                    print(case + '  -> ' + str(currentPercentage) + ' out of 100%')

                    # Generate Data for all Techniques
                    self.Process_Participants_Data_EntireSignalINChunks(
                        objWindowProcessedData, self.objConfig.SavePath,
                        # objWindowProcessedData.HrGroundTruthList, objWindowProcessedData.SPOGroundTruthList, TODO:FIX
                        self.objConfig.DumpToDisk,
                        ProcessingStep, case)
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
                objWindowProcessedData = self.objFileIO.ReadfromDisk(self.objConfig.SavePath,
                                                                     'UnCompressedBinaryLoadedData')

                # Store for procesing locally
                ParticipantsOriginalDATA[participant_number + '_' + position] = objWindowProcessedData

        return ParticipantsOriginalDATA

    '''
     LoadPartiallyProcessedBinaryData: load data from disk ParticipantsPartiallyProcessedBinaryData[ParticipantNumber + Position] -> ROISTORE data
     '''
    def LoadPartiallyProcessedBinaryData(self,LoadFolder):
        ParticipantsPartiallyProcessedBinaryData = {}
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                self.objConfig.setSavePath(participant_number, position, LoadFolder)  # set path
                print('Loading and generating FaceData for ' + participant_number + ', ' + position)
                print('Loading from path ' + self.objConfig.SavePath)
                AllFileNames = os.listdir(self.objConfig.SavePath) #'ResultSignal_Type-' + str(Preprocess_type)
                for fileName in AllFileNames:
                    fileNameSplit = fileName.split('-')
                    processType = fileNameSplit[1]
                    ##Data in binary read from disk
                    objWindowProcessedData = self.objFileIO.ReadfromDisk(self.objConfig.SavePath,
                                                                         fileName)
                    # Store for procesing locally
                    ParticipantsPartiallyProcessedBinaryData[participant_number + '_' + position + '_' + processType ] = objWindowProcessedData

        return ParticipantsPartiallyProcessedBinaryData

    def mainMethod(self,ProcessingStep):
        #Process for entire signal
        FolderNameforSave= 'ProcessedDatabyProcessType'
        print(FolderNameforSave)
        #  Load Data from path and Process Data
        LoadedData = None
        ProcessCase = None
        LoadFolderName = ProcessingStep + '_WindowsBinaryFiles'
        ##Process by type
        if (ProcessingStep == 'PreProcess'):
            ProcessCase = self.objConfig.preprocesses
            LoadedData = self.LoadBinaryData()
        elif (ProcessingStep == 'Algorithm'):
            LoadedData = self.LoadPartiallyProcessedBinaryData(LoadFolderName)
            ProcessCase = self.objConfig.AlgoList
        elif (ProcessingStep == 'Smoothen'):
            ProcessCase = self.objConfig.Smoothen
            LoadedData = self.LoadPartiallyProcessedBinaryData(LoadFolderName)
        elif (ProcessingStep == 'FFT'):
            ProcessCase = self.objConfig.fftTypeList
            LoadedData = self.LoadPartiallyProcessedBinaryData(LoadFolderName)
        elif (ProcessingStep == 'Filter'):
            ProcessCase = self.objConfig.filtertypeList
            LoadedData = self.LoadPartiallyProcessedBinaryData(LoadFolderName)
        elif (ProcessingStep == 'Result'):        # Result_type = 0 #TODO:RESULT
            ProcessCase = self.objConfig.resulttypeList
            LoadedData = self.LoadPartiallyProcessedBinaryData(LoadFolderName)
        self.GenerateResultsfromParticipants(LoadedData,FolderNameforSave,ProcessingStep,ProcessCase)#FOR Window processing

###RUN this file CODE###
skintype = 'Europe_WhiteSkin_Group'
print('Program started for ' +skintype)
objInitiateProcessing = InitiateProcessingStorage(skintype)
# objInitiateProcessing.GenerateOriginalRawData()# Only do once
ProcessingStep='PreProcess'
objInitiateProcessing.mainMethod(1,ProcessingStep) #type
# objInitiateProcessing.main(2) #type
# objInitiateProcessing.main(3) #type
# objInitiateProcessing.main(4) #type
# objInitiateProcessing.main(5) #type
# objInitiateProcessing.main(6) #type
# objInitiateProcessing.main(7) #type
# print('Calculating final result')
# objInitiateProcessing.ProduceFinalResult()
print('Program completed')