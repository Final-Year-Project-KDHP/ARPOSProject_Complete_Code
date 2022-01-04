import os

import CommonMethods
from CalculateAndStoreOnDisk.ProcessDataPartial import ProcessDataPartial
from Configurations import Configurations
from LoadFaceData import LoadFaceData
from GlobalDataFile import GlobalData
import pickle

from ProcessParticipantsData import Process_Participants_Data_GetBestHR

'''
InitiateProcessingStorage:
'''
class InitiateProcessingStorage:
    # Global Objects
    objConfig = None
    ProcessedDataPath = None

    #Constructor
    def __init__(self, skinGroup):
        self.objConfig = Configurations(False,skinGroup)

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
        self.objConfig.ParticipantNumbers.remove('PIS-8073')
        self.objConfig.ParticipantNumbers.remove('PIS-2047')
        self.objConfig.ParticipantNumbers.remove('PIS-4014')
        self.objConfig.ParticipantNumbers.remove('PIS-1949')
        self.objConfig.ParticipantNumbers.remove('PIS-3186')
        self.objConfig.ParticipantNumbers.remove('PIS-7381')
        self.objConfig.ParticipantNumbers.remove('PIS-6729')
        self.objConfig.ParticipantNumbers.remove('PIS-6327')
        self.objConfig.ParticipantNumbers.remove('PIS-6888')
        self.objConfig.ParticipantNumbers.remove('PIS-4709')
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

    def main(self,type):

        # setup highpass filter
        ignore_freq_below_bpm = 40
        ignore_freq_below = ignore_freq_below_bpm / 60
        # setup low pass filter
        ignore_freq_above_bpm = 200
        ignore_freq_above = ignore_freq_above_bpm / 60

        # each particpant
        for participant_number in self.objConfig.ParticipantNumbers:
            # each position
            for position in self.objConfig.hearratestatus:
                # print
                print(participant_number + ', ' + position)
                # set path
                self.objConfig.setSavePath(participant_number, position)

                # for each region of interest
                for region in self.objConfig.roiregions:
                    # Init for each region

                    # Run as per type
                    if (type == 1):  # get and write orignial data
                        objFaceImage = LoadFaceData()
                        objFaceImage.Clear()
                        self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'RawOriginal\\'
                        objInitiateProcessing.GenerateOriginalRawData(participant_number, position,objFaceImage,region)
                        # Clear
                        del objFaceImage
                    elif (type == 2): #PreProcess
                        objFaceImage = LoadFaceData()
                        objFaceImage.Clear()
                        self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'RawOriginal\\'
                        objFaceImage = objInitiateProcessing.ReadOriginalRawData(region)
                        self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'PreProcessed\\'
                        self.PreProcessData(objFaceImage.blue,objFaceImage.green,objFaceImage.red,objFaceImage.grey,objFaceImage.Irchannel,
                                            objFaceImage.ColorEstimatedFPS,objFaceImage.IREstimatedFPS,region,
                                            objFaceImage.timecolorCount,objFaceImage.timeirCount)
                        # Clear
                        del objFaceImage
                    elif (type == 3): #ApplyAlgorithm
                        self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'RawOriginal\\'
                        preProcessedDataFileNames = self.getFileName(self.objConfig.preprocesses,region,"PreProcessedData")
                        for filename in preProcessedDataFileNames:
                            splitFilename = filename.split('-')
                            preProcessType = splitFilename[1]
                            self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'PreProcessed\\'
                            dataContent = objInitiateProcessing.ReadData(filename)
                            self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'AlgorithmProcessed\\'
                            self.ApplyAlgorithm(dataContent.ProcessedSignalData,dataContent.ColorEstimatedFPS,dataContent.IREstimatedFPS,region,preProcessType)

                    elif (type == 4): #IsSmooth
                        # Apply smoothen only before fft
                        # if (self.isSmoothen):
                        DataFileNames = self.getPreProcessedAlgorithms(region)
                        for filename in DataFileNames:
                            splitFilename = filename.split('-')
                            algoType = (splitFilename[1]).replace('_PreProcessType','')
                            preProcessType = splitFilename[2]
                            self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'AlgorithmProcessed\\'
                            dataContent = objInitiateProcessing.ReadData(filename)
                            self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'SmoothenProcessed\\'
                            self.ApplySmooth(dataContent.ProcessedSignalData,dataContent.ColorEstimatedFPS,dataContent.IREstimatedFPS,region,preProcessType,algoType)

                    elif (type == 5): #Apply FFT WITH smoothed and without smooth (apply both types here)
                        ##First Without Smoothen
                        IsSmoothProcessed =False
                        DataFileNames = self.getPreProcessedAlgorithms(region)
                        for filename in DataFileNames:
                            splitFilename = filename.split('-')
                            algoType = (splitFilename[1]).replace('_PreProcessType','')
                            preProcessType = splitFilename[2]
                            self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'AlgorithmProcessed\\'
                            dataContent = objInitiateProcessing.ReadData(filename)
                            self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'FFTProcessedWithoutSmooth\\'
                            self.ApplyFFT(dataContent.ProcessedSignalData,dataContent.ColorEstimatedFPS,dataContent.IREstimatedFPS,region,preProcessType,algoType,IsSmoothProcessed)

                        ###For Smoothen True
                        IsSmoothProcessed =True
                        DataFileNames = self.getPreProcessedAlgorithmSmoothedFiles(region)
                        for filename in DataFileNames:
                            splitFilename = filename.split('-')
                            algoType = (splitFilename[1]).replace('_PreProcessType','') #'SmoothedData_'+ region + "_Algotype-"+ str(algo_type) + '_PreProcessType-'+str(preprocess_type)
                            preProcessType = filename.split('-')[2]
                            self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'SmoothenProcessed\\'
                            dataContent = objInitiateProcessing.ReadData(filename)
                            self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'FFTProcessedSmoothed\\'
                            self.ApplyFFT(dataContent.ProcessedSignalData,dataContent.ColorEstimatedFPS,dataContent.IREstimatedFPS,region,preProcessType,algoType,IsSmoothProcessed)

                    elif (type == 6): #Apply filter WITH smoothed and without smooth (apply both types here)
                        ##First Without Smoothen
                        IsSmoothProcessed =False
                        # DataFileNames = self.getPreProcessedAlgorithmsFFT(region,IsSmoothProcessed)
                        self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'FFTProcessedWithoutSmooth\\'
                        DataFileNameWithContent = self.getPreProcessedAlgorithmsFFT(region,IsSmoothProcessed)
                        for filename, dataContent in DataFileNameWithContent.items():
                        # for filename in DataFileNames:
                            splitFilename = filename.split('_') #fileName = "FFT-" + fft_type + "_" + region + "_Smoothed-" + str(isSmoothed) + "_Algotype-" + str(algo_type) + '_PreProcessType-' + str(preprocess_type)
                            fft_type = (splitFilename[0]).replace('FFT-','')
                            algotype = splitFilename[3].replace('Algotype-','')
                            preprocesstype = splitFilename[4].replace('PreProcessType-','')
                        #     self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'FFTProcessedWithoutSmooth\\'
                        #     dataContent = objInitiateProcessing.ReadData(filename)
                            self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'FilteredWithoutSmooth\\'
                            self.ApplyFilter(dataContent.ProcessedSignalData,
                                             dataContent.ColorEstimatedFPS,dataContent.IREstimatedFPS,region,preprocesstype,algotype,IsSmoothProcessed,fft_type,
                                             dataContent.Colorfrequency, dataContent.IRfrequency, ignore_freq_below, ignore_freq_above)
                        ###For Smoothen True
                        IsSmoothProcessed =True
                        self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'FFTProcessedSmoothed\\'
                        DataFileNameWithContent = self.getPreProcessedAlgorithmsFFT(region,IsSmoothProcessed)
                        for filename, dataContent in DataFileNameWithContent.items():
                            splitFilename = filename.split('_')
                            fft_type = (splitFilename[0]).replace('FFT-', '')
                            algotype = splitFilename[3].replace('Algotype-', '')
                            preprocesstype = splitFilename[4].replace('PreProcessType-', '')
                            # dataContent = objInitiateProcessing.ReadData(filename)
                            self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'FilteredWithSmoothen\\'
                            self.ApplyFilter(dataContent.ProcessedSignalData,
                                             dataContent.ColorEstimatedFPS, dataContent.IREstimatedFPS, region, preprocesstype,
                                             algotype, IsSmoothProcessed, fft_type,
                                             dataContent.Colorfrequency, dataContent.IRfrequency, ignore_freq_below,
                                             ignore_freq_above)
                    elif (type == 7):  # Computer results
                        ##First Without Smoothen
                        IsSmoothProcessed = False
                        self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'FilteredWithoutSmooth\\'
                        DataFileNameWithContent = self.getPreProcessedFiltered(region, IsSmoothProcessed)
                        self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'ResultDataWithoutSmooth\\'
                        for filename, dataContent in DataFileNameWithContent.items():
                            splitFilename = filename.split('_')  # "Filtered_FL-"+ str(filterType)+ "_"+ region+ "_FFTtype-" + str(fft_type)  + "_algotype-" + str(algo_type) +'_PreProcessType-'+str(preprocess_type)+ "_Smoothed-" + str(isSmoothed)
                            filterType = (splitFilename[1]).replace('FL-', '')
                            fft_type = (splitFilename[3]).replace('FFTtype-', '')
                            algotype = splitFilename[4].replace('algotype-', '')
                            preprocesstype = splitFilename[5].replace('PreProcessType-', '')

                            # dataContent = objInitiateProcessing.ReadData(filename)
                            self.ComputerResultData(dataContent.B_signal, dataContent.G_signal, dataContent.R_signal, dataContent.Gy_signal, dataContent.IR_signal,
                                                    dataContent.ColorEstimatedFPS, dataContent.IREstimatedFPS,
                                                    region, preprocesstype, algotype, IsSmoothProcessed,fft_type,filterType,
                                                    dataContent.Colorfrequency, dataContent.IRfrequency, ignore_freq_below_bpm, ignore_freq_above_bpm)
                        ###For Smoothen True
                        IsSmoothProcessed = True
                        self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'FilteredWithSmoothen\\'
                        DataFileNameWithContent = self.getPreProcessedFiltered(region, IsSmoothProcessed)
                        self.ProcessedDataPath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\' + 'ResultDataSmoothed\\'
                        for filename, dataContent in DataFileNameWithContent.items():
                            splitFilename = filename.split('_')
                            filterType = (splitFilename[1]).replace('FL-', '')
                            fft_type = (splitFilename[3]).replace('FFTtype-', '')
                            algotype = splitFilename[4].replace('algotype-', '')
                            preprocesstype = splitFilename[5].replace('PreProcessType-', '')
                            # dataContent = objInitiateProcessing.ReadData(filename)
                            self.ComputerResultData(dataContent.B_signal, dataContent.G_signal, dataContent.R_signal,
                                                    dataContent.Gy_signal, dataContent.IR_signal,
                                                    dataContent.ColorEstimatedFPS, dataContent.IREstimatedFPS,
                                                    region, preprocesstype, algotype, IsSmoothProcessed, fft_type,
                                                    filterType,
                                                    dataContent.Colorfrequency, dataContent.IRfrequency,
                                                    ignore_freq_below_bpm, ignore_freq_above_bpm)

                    # Add to store
                    # objFaceStore[region]= objFaceImage





###RUN this file CODE###
skintype = 'Europe_WhiteSkin_Group'
print('Program started for ' +skintype)
objInitiateProcessing = InitiateProcessingStorage(skintype)
# objInitiateProcessing.GenerateOriginalRawData()# Only do once
# objInitiateProcessing.main(1) #type
# objInitiateProcessing.main(2) #type
# objInitiateProcessing.main(3) #type
# objInitiateProcessing.main(4) #type
# objInitiateProcessing.main(5) #type
# objInitiateProcessing.main(6) #type
# objInitiateProcessing.main(7) #type
print('Calculating final result')
objInitiateProcessing.ProduceFinalResult()
print('Program completed')