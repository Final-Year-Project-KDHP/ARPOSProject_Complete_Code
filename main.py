import csv
import os
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import scipy.stats
from scipy.stats import chisquare
import scipy.stats as stats

import CommonMethods
from CalculateAndStoreOnDisk.InitiateProcessingStorage import InitiateProcessingStorage
from CheckReliability import CheckReliability
from Configurations import Configurations
from FileIO import FileIO
from LoadFaceData import LoadFaceData
from ProcessParticipantsData import Process_SingalData, objFile
from GlobalDataFile import GlobalData, WindowProcessedData
from ResultsandGraphs import ResultsandGraphs
from SQLResults.SQLConfig import SQLConfig
from SaveGraphs import Plots

"""
Main Class:
"""
class Main:
    # Store Region of Interest and Results
    ROIStore = {}

    # Global Objects
    objConfig = None
    objSQLConfig= None
    objFileIO = None
    ParticipantsProcessedHeartRateData = {}
    ParticipantsProcessedBloodOxygenData = {}
    HRNameFileName = "AvgHRdata_"
    SPONameFileName = "AvgSPOdata_"
    CaseList = []

    HeaderRow = 'WindowCount,bestSnrString,GroundTruth HeartRate Averaged,Computed HeartRate,HRDifference from averaged,' \
                'bestBpm Without ReliabilityCheck,OriginalObtianedAveragedifferenceHR, Hr from windows last second, ' \
                'LastSecondWindow differenceHR, OriginalObtianed LastSecondWindow differenceHR,' \
                'GroundTruth SPO Averaged,Computed SPO,SPO Difference from averaged,best SPO WithoutReliability Check,Original Obtianed Average differenceSPO,' \
                'SPOLastSecond,LastSecondWindowdifferenceSPO ,OriginalObtianedLastSecondWindowdifferenceSPO, Regiontype, channeltype,' \
                'FrequencySamplingError,heartRateError,TotalWindowCalculationTimeTaken,' \
                'PreProcessTimeTaken,AlgorithmTimeTaken,FFTTimeTaken,SmoothTimeTaken,' \
                'FilterTimeTaken,ComputingHRSNRTimeTaken,ComputingSPOTimeTaken,Algorithm_type,' \
                'FFT_type,Filter_type,Result_type,Preprocess_type,isSmoothen,ColorFPS,IRFPS,' \
                'SelectedColorFPSMethod,SelectedIRFPSMethod,' \
                'AttemptType,FPSNotes,UpSampled'
    HeaderRowSplit = []

    # Constructor
    def __init__(self, skinGroup='None'):
        self.objConfig = Configurations(skinGroup)
        self.objSQLConfig = SQLConfig()
        self.objFileIO = FileIO()
        self.HeaderRowSplit = self.HeaderRow.split(",")

    """
    Generate cases:
    """
    def GenerateCases(self):
        self.CaseList = []
        for preprocesstype in self.objConfig.preprocesses:
            for algoType in self.objConfig.AlgoList:
                for isSmooth in self.objConfig.Smoothen:
                    for fftype in self.objConfig.fftTypeList:
                        for filtertype in self.objConfig.filtertypeList:
                            for resulttype in self.objConfig.resulttypeList:
                                    fileName = algoType + "_PR-" + str(preprocesstype) + "_FFT-" + str(
                                        fftype) + "_FL-" + str(filtertype) \
                                               + "_RS-" + str(resulttype) + "_SM-" + str(isSmooth)
                                    if (fileName not in self.CaseList):
                                        self.CaseList.append(fileName)

    def CheckIfGenerated(self, fileName):
        pathExsists = objFile.FileExits(self.objConfig.SavePath + 'Result\\' + 'HRSPOwithLog_' + fileName + ".txt")
        # already generated
        if (pathExsists):
            return True
        return False

    """
     GenerateResultsfromParticipants:
     """
    def GenerateResultsfromParticipants(self, ParticipantsOriginalDATA,typeProcessing, UpSampleData,CaseListExists,NoHRCases,AttemptType,skintype):
        self.GenerateCases()
        TotalCasesCount = len(self.CaseList)
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                print(participant_number + ', ' + position)
                objWindowProcessedData = ParticipantsOriginalDATA.get(participant_number + '_' + position)
                self.objConfig.setSavePath(participant_number, position,typeProcessing)
                currentCasesDone = 0
                for case in self.CaseList:
                    casefullValue = case + '+' + participant_number+'+' +position + str(UpSampleData)
                    IsGenerated= True if CaseListExists.count(casefullValue) else False
                    if(not IsGenerated):
                        IsGenerated= True if NoHRCases.__contains__(case) else False
                    if(not IsGenerated):
                        currentCasesDone = currentCasesDone + 1
                        currentPercentage = ((currentCasesDone/TotalCasesCount)*100)
                        print(case + '  -> ' + str(currentPercentage) + ' out of 100%')
                        splitCase = case.split('_')
                        fileName = case
                        algoType = splitCase[0]
                        fftype = splitCase[2].replace('FFT-', '')
                        filtertype = int(splitCase[3].replace('FL-', ''))
                        resulttype = int(splitCase[4].replace('RS-', ''))
                        preprocesstype = int(splitCase[1].replace('PR-', ''))
                        isSmooth = splitCase[5].replace('SM-', '')
                        isSmooth=isSmooth.lower()
                        isSmooth=isSmooth.capitalize()
                        if (isSmooth == 'True'):
                            isSmooth = True
                        else:
                            isSmooth = False

                        # Generate Data for all Techniques
                        objParticipantsResultEntireSignalDataRow = Process_SingalData(
                            self.objConfig.RunAnalysisForEntireSignalData,
                            objWindowProcessedData.ROIStore, self.objConfig.SavePath,
                            algoType, fftype,
                            filtertype, resulttype, preprocesstype, isSmooth,
                            objWindowProcessedData.HrGroundTruthList, objWindowProcessedData.SPOGroundTruthList,
                            fileName, self.objConfig.DumpToDisk,participant_number,position,UpSampleData,AttemptType)
                        if(objParticipantsResultEntireSignalDataRow != 'NO HR'):
                            self.objSQLConfig.SaveRowParticipantsResultsEntireSignal(objParticipantsResultEntireSignalDataRow)
                        else:
                            if(not NoHRCases.__contains__(case)):
                                exists = self.objFileIO.FileExits(self.objConfig.SavePath+ "NOHRCases_" + skintype+ ".txt")
                                mode = "w+"
                                if(exists):
                                    mode = "a"
                                self.objFileIO.WritedatatoFile(self.objConfig.SavePath,"NOHRCases_" + skintype, objParticipantsResultEntireSignalDataRow + "\t" + case,mode)

                ParticipantsOriginalDATA.pop(participant_number + '_' + position)

    def LoadandGenerateFaceDatatoBianryFiles(self,SaveFileName, UpSampleData, SaveFolder):
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                self.objConfig.setSavePath(participant_number, position, SaveFolder)  # set path
                print('Loading and generating FaceData for ' + participant_number + ', ' + position + " and UpSampleData" + str(UpSampleData))
                print('Loading from path ' + self.objConfig.DiskPath + '; Storing data to path ' + self.objConfig.SavePath)
                # for each roi and put in Store Region
                for region in self.objConfig.roiregions:
                    # Init for each region
                    objFaceImage = LoadFaceData()
                    objFaceImage.Clear()

                    ##get loading path
                    LoadColordataPath, LoadIRdataPath, LoadDistancePath = self.objConfig.getLoadPath(
                        participant_number, position,
                        region)

                    # Load Roi data (ALL)
                    # print("Loading and processing color roi data")
                    objFaceImage.ProcessColorImagestoArray(LoadColordataPath,UpSampleData)

                    # print("Loading and processing ir roi data")
                    objFaceImage.ProcessIRImagestoArray(LoadIRdataPath, LoadDistancePath,UpSampleData)

                    # GET FPS and distance and other data
                    # if(region == 'lips'):
                    # objFaceImage.GetDistance(LoadDistancePath)

                    # Create global data object and use dictionary (ROI Store) to uniquely store a regions data
                    self.ROIStore[region] = GlobalData(objFaceImage.time_list_color, objFaceImage.timecolorCount,
                                                       objFaceImage.time_list_ir, objFaceImage.timeirCount,
                                                       objFaceImage.Frametime_list_ir,
                                                       objFaceImage.Frametime_list_color,
                                                       objFaceImage.red, objFaceImage.green, objFaceImage.blue,
                                                       objFaceImage.grey,
                                                       objFaceImage.Irchannel, objFaceImage.distanceM,
                                                       objFaceImage.totalTimeinSeconds,
                                                       objFaceImage.ColorEstimatedFPS, objFaceImage.IREstimatedFPS,
                                                       objFaceImage.ColorfpswithTime, objFaceImage.IRfpswithTime)

                    # delete face image object
                    del objFaceImage

                HrGr, SpoGr = CommonMethods.GetGroundTruth(participant_number, position,
                                                           self.objConfig.DiskPath.replace("SerialisedRawServerData",""),
                                                           int(self.ROIStore.get(self.objConfig.roiregions[0]).totalTimeinSeconds))

                ##Original Data storage
                objWindowProcessedData = WindowProcessedData()
                objWindowProcessedData.HrGroundTruthList = HrGr
                objWindowProcessedData.SPOGroundTruthList = SpoGr
                objWindowProcessedData.ColorLengthofAllFrames = self.ROIStore.get(self.objConfig.roiregions[0]).getLengthColor()
                objWindowProcessedData.IRLengthofAllFrames = self.ROIStore.get(self.objConfig.roiregions[0]).getLengthIR()
                objWindowProcessedData.TimeinSeconds = int(self.ROIStore.get(self.objConfig.roiregions[0]).totalTimeinSeconds)
                objWindowProcessedData.ROIStore = self.ROIStore
                self.objFileIO.DumpObjecttoDisk(self.objConfig.SavePath.replace("SerialisedRawServerData",""), SaveFileName, objWindowProcessedData)

                del objWindowProcessedData

    '''
    LoadBinaryData: load data from disk ParticipantsOriginalDATA[ParticipantNumber + Position] -> ROISTORE data
    '''
    def LoadBinaryData(self, FileName,LoadFolder):
        ParticipantsOriginalDATA = {}
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                self.objConfig.setSavePath(participant_number, position, LoadFolder)  # set path

                ##binary Data read from disk
                objWindowProcessedData = self.objFileIO.ReadfromDisk(self.objConfig.SavePath, FileName)

                # Store for procesing locally
                ParticipantsOriginalDATA[participant_number + '_' + position] = objWindowProcessedData

        return ParticipantsOriginalDATA

    def LoadBinaryDataOld(self,SaveFileName,UpSampleData):
        ParticipantsOriginalDATA = {}
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant

            # if(participant_number == 'PIS-4014' or participant_number == 'PIS-6729'):#TODO:UPDATE
            #     if (UpSampleData):
            #         SaveFileName = 'UnCompressedBinaryLoadedDataSelectiveLowFrameRemovedUpSampled'  # UnCompressedBinaryLoadedDataUpSampled,UnCompressedBinaryLoadedDataSelectiveLowFrameRemovedUpSampled
            #     else:
            #         SaveFileName = 'UnCompressedBinaryLoadedDataSelectiveLowFrameRemoved'  # 'UnCompressedBinaryLoadedData'<-- Orignial , UnCompressedBinaryLoadedData2, UnCompressedBinaryLoadedDataSelectiveLowFrameRemoved
            # else:
            #     if (UpSampleData):
            #         SaveFileName = 'UnCompressedBinaryLoadedDataUpSampled'  # UnCompressedBinaryLoadedDataUpSampled,UnCompressedBinaryLoadedDataSelectiveLowFrameRemovedUpSampled
            #     else:
            #         SaveFileName = 'UnCompressedBinaryLoadedData2'  # 'UnCompressedBinaryLoadedData'<-- Orignial , UnCompressedBinaryLoadedData2, UnCompressedBinaryLoadedDataSelectiveLowFrameRemoved

            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                self.objConfig.setSavePath(participant_number, position, 'RawOriginal')  # set path
                # print('Loading and generating FaceData for ' + participant_number + ', ' + position)
                # print('Loading from path ' + self.objConfig.SavePath)

                # if (self.objConfig.ParticipantNumbers == "4497" or  self.objConfig.ParticipantNumbers == "2047"
                #         or self.objConfig.ParticipantNumbers == "3186"
                #         or self.objConfig.ParticipantNumbers == "6888"
                #         or self.objConfig.ParticipantNumbers == "2212" or self.objConfig.ParticipantNumbers == "8343" ):
                #     SaveFileName = 'UnCompressedBinaryLoadedData2'  # 'UnCompressedBinaryLoadedData' <-- Orignial
                # else:
                #     SaveFileName = 'UnCompressedBinaryLoadedDataUpSampled'

                ##Original Data in binary read from disk
                objWindowProcessedData = self.objFileIO.ReadfromDisk(self.objConfig.SavePath,
                                                                     SaveFileName)
                ##To print fps and update in sql
                # minFPSIRvalue = min(objWindowProcessedData.ROIStore.get(self.objConfig.roiregions[0]).IRfpswithTime.values())
                # maxFPSIRvalue = max(objWindowProcessedData.ROIStore.get(self.objConfig.roiregions[0]).IRfpswithTime.values())
                # minFPSColorvalue = min(objWindowProcessedData.ROIStore.get(self.objConfig.roiregions[0]).ColorfpswithTime.values())
                # maxFPSColorvalue = max(objWindowProcessedData.ROIStore.get(self.objConfig.roiregions[0]).ColorfpswithTime.values())
                # FPSNotes = 'min IRvalue: ' + str(minFPSIRvalue) + ', max IRvalue: ' + str(
                #     maxFPSIRvalue) + ', min Colorvalue: ' + str(minFPSColorvalue) + ', max Colorvalue: ' + str(
                #     maxFPSColorvalue)
                # upsampledb = '1' if UpSampleData == True else '0'
                # FULLSTRing = "update ParticipantsResultsEntireSignal set FPSNotes='" + FPSNotes + "' where ParticipantId = '"+  participant_number+ "' and HeartRateStatus = '" + position + "' and FPSNotes is null and UpSampled=" + str(upsampledb) + " and AttemptType=1"
                # print(FULLSTRing)


                #Only for windows as for same lenght (DOWNSAMPLING) --> Bad idea
                # SaveFileName2 = 'UnCompressedBinaryLoadedData2CombinedSameLength'
                # if(SaveFileName2 == 'UnCompressedBinaryLoadedData2CombinedSameLength'):
                #     #for combine donly
                #     for regionNo in self.objConfig.roiregions:
                #         ROIStoreCurrent = objWindowProcessedData.ROIStore.get(regionNo)
                #         regionColor = ROIStoreCurrent.ColorfpswithTime
                #         regionIR = ROIStoreCurrent.IRfpswithTime
                #         if (len(ROIStoreCurrent.Irchannel) == len(ROIStoreCurrent.grey)):
                #             break
                #         else:
                #             indexTime = 0
                #             IndexArray = 0
                #             for k in regionIR:
                #                 if (regionIR.get(k) > regionColor.get(k)):
                #                     IndexArray = IndexArray + regionIR.get(k)  # as color and ir are same
                #                     difference = regionIR.get(k) - regionColor.get(k)
                #                     for popx in range (0,difference):
                #                         ROIStoreCurrent.Irchannel.pop(IndexArray)
                #                         ROIStoreCurrent.Frametime_list_ir.pop(IndexArray)
                #                         ROIStoreCurrent.distanceM.pop(IndexArray)
                #                         IndexArray = IndexArray - 1
                #                     regionIR[k] = regionColor.get(k)
                #                 elif (regionColor.get(k) >regionIR.get(k)  ):
                #                     IndexArray = IndexArray + regionColor.get(k)  # as color and ir are same
                #                     difference = regionColor.get(k) -regionIR.get(k)
                #                     for popx in range (0,difference):
                #                         ROIStoreCurrent.grey.pop(IndexArray)
                #                         ROIStoreCurrent.red.pop(IndexArray)
                #                         ROIStoreCurrent.green.pop(IndexArray)
                #                         ROIStoreCurrent.blue.pop(IndexArray)
                #                         ROIStoreCurrent.Frametime_list_color.pop(IndexArray)
                #                         IndexArray = IndexArray - 1
                #                     regionColor[k] = regionIR.get(k)
                #                 else:
                #                     IndexArray= IndexArray + regionIR.get(k)# as color and ir are same
                #
                #
                #                 #else do nothing
                #                 indexTime = indexTime+1
                #
                #         ROIStoreCurrent.ColorEstimatedFPS = self.getDuplicateValue(ROIStoreCurrent.ColorfpswithTime)
                #         ROIStoreCurrent.IREstimatedFPS = self.getDuplicateValue(ROIStoreCurrent.IRfpswithTime)
                #         ROIStoreCurrent.timecolorCount = list(range(0, len(ROIStoreCurrent.Frametime_list_color)))  # self.TemptimecolorCount
                #         ROIStoreCurrent.timeirCount = list(range(0, len(ROIStoreCurrent.Frametime_list_ir)))  # self.TemptimecolorCount
                #
                #
                #     self.objFileIO.DumpObjecttoDisk(self.objConfig.SavePath, SaveFileName2,objWindowProcessedData)

                ####PRINT FPS WINDOW DETAIL IN FULL DETAIL
                # ROIStoreCurrent = objWindowProcessedData.ROIStore.get('lips')
                # regionColor = ROIStoreCurrent.ColorfpswithTime
                # regionIR = ROIStoreCurrent.IRfpswithTime
                # if(len(ROIStoreCurrent.grey) != len(ROIStoreCurrent.Irchannel)):
                #     # print(participant_number + ' - ' + position + ' : equal length')
                #     print(participant_number + ' - ' + position + ', GreyL: ' + str(len(ROIStoreCurrent.grey)) + ', IRL: ' + str(len(ROIStoreCurrent.Irchannel)))
                #     for k in regionIR:
                #         if (regionIR.get(k) != regionColor.get(k)):
                #             difference = regionIR.get(k) - regionColor.get(k)
                #             print(str(k) + ' : ' + str(np.abs(difference)))
                # else:
                ##Making equl method random
                # SaveFileName2 = 'UnCompressedBinaryLoadedData2CombinedSameLength'
                # # exists = self.objFileIO.FileExits(self.objConfig.SavePath + SaveFileName2)
                # # if(not exists):
                # if (UpSampleData):
                #     SaveFileName2 = 'UnCompressedBinaryLoadedDataUpSampledCombinedSameLength'
                # for region in self.objConfig.roiregions:
                #     ROIStoreCurrent = objWindowProcessedData.ROIStore.get(region)
                #     # if(ROIStoreCurrent.ColorEstimatedFPS >15):#TODO FIX FOR FPS --15
                #     #
                #     if (len(ROIStoreCurrent.grey) != len(ROIStoreCurrent.Irchannel)):
                #         regionColor = ROIStoreCurrent.ColorfpswithTime
                #         regionIR = ROIStoreCurrent.IRfpswithTime
                #         ####RANDOM REMOVAL
                #         # # REMOVING BELOW as no longer constaant for ir and color
                #         if (len(ROIStoreCurrent.Irchannel) > len(ROIStoreCurrent.grey)):
                #             difference = len(ROIStoreCurrent.Irchannel) -  len(ROIStoreCurrent.grey)
                #             for i in range(0, difference):
                #                 ROIStoreCurrent.Irchannel.pop()
                #                 ROIStoreCurrent.distanceM.pop()
                #
                #             ROIStoreCurrent.timeirCount = list(range(0, len(ROIStoreCurrent.Frametime_list_ir)))
                #             if (ROIStoreCurrent.ColorEstimatedFPS <= 16):  # TODO FIX FOR FPS --15
                #                 ROIStoreCurrent.IREstimatedFPS = 15
                #
                #         if (len(ROIStoreCurrent.grey) > len(ROIStoreCurrent.Irchannel)):
                #             differnce = len(ROIStoreCurrent.grey) - len(ROIStoreCurrent.Irchannel)
                #
                #             for i in range(0, differnce):
                #                 ColorTime = ROIStoreCurrent.Frametime_list_color.pop()
                #                 ROIStoreCurrent.red.pop()
                #                 ROIStoreCurrent.green.pop()
                #                 ROIStoreCurrent.blue.pop()
                #                 ROIStoreCurrent.grey.pop()
                #
                #             ROIStoreCurrent.timecolorCount = list(range(0, len(ROIStoreCurrent.Frametime_list_color)))  # self.TemptimecolorCount
                #
                # self.objFileIO.DumpObjecttoDisk(self.objConfig.SavePath, SaveFileName2,objWindowProcessedData)

                # Store for procesing locally
                ParticipantsOriginalDATA[participant_number + '_' + position] = objWindowProcessedData

        return ParticipantsOriginalDATA

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
    '''
    Loadnohr files:  for testing cases check only
    '''
    def LoadNoHRfILES(self,SaveFileName,skintype):

        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            # ListSumCases = 0
            ListAllbyPostion = []
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                self.objConfig.setSavePath(participant_number, position, SaveFileName)  # set path
                fileDataCases = self.objFileIO.ReaddatafromFile(self.objConfig.SavePath,
                                                                     "NOHRCases_" + skintype )

                fileDataCasesAll = fileDataCases[0].split("NO HR\t")
                # fileDataCasesAll = [ele for ele in fileDataCasesAll if ele != []]
                for item in fileDataCasesAll:
                    if(item != ''):
                        ListAllbyPostion.append(item)

                ListAllbyPostion = list(dict.fromkeys(ListAllbyPostion))

                # fileDataCasesAll = fileDataCasesAll.replace("\t",)
                # print(participant_number + ', ' + position + ', '+ str(len(fileDataCasesAll)))
                # ListSumCases =ListSumCases +len(fileDataCasesAll)
            ListAllbyPostion = list(dict.fromkeys(ListAllbyPostion))
            # print(skintype+ ', ' + str((len(ListAllbyPostion))))
        return ListAllbyPostion

    def GenerateFaceData(self):
        self.LoadandGenerateFaceDatatoBianryFiles("BinaryFaceROI", False,'BinaryFaceROIDataFiles')# Requries -> SaveFileName, UpSampleData, SaveFolder

    def RunAllinOneGo(self,skintype):
        self.GenerateFaceData()  # Run only once

        # Load data
        ParticipantsDATA = self.LoadBinaryData("BinaryFaceROI", 'BinaryFaceROIDataFiles')

        # if(self.objConfig.RunAnalysisForEntireSignalData):
        #     #For entire signal
        # else:
        #     #For Window FolderNameforSave = 'ProcessedDataWindows'
        #  Load Data from path and Process Data
        self.GenerateResultsfromParticipants(ParticipantsDATA, 'ProcessedData',skintype)  # FOR Window processing

    '''
    GenerateGraphfromStoredFile: 
    '''
    def GenerateGraphfromStoredFile(self):
        PlotType = "FFT"# "PreProcessed"
        DiskPath = "D:\\ARPOS_Server_Data\\Server_Study_Data\\AllParticipants\\"
        FolderName = "FFTDataFiles"#"PreProcessDataFilesSameLength"#"PreProcessDataFiles"
        participantNumber = "PIS-1032"
        position = "Resting1"
        subFolderName = "FFT_M1_Algorithm_FastICA3TimesCombined_PreProcess_6"
        fileName = "ProcessedWindow_5"
        fullPath = DiskPath + FolderName + "\\" + participantNumber + "\\" + position + "\\" + subFolderName + "\\"
        self.objFileIO.CreatePath(fullPath)
        LoadedData = self.objFileIO.ReadfromDisk(fullPath, fileName)
        objProcessData = LoadedData.WindowRegionList["cheeksCombined"]
        objProcessData.GenerateGraphs = True
        objProcessData.objAlgorithm.ColorEstimatedFPS = objProcessData.ColorEstimatedFPS
        objProcessData.objAlgorithm.IREstimatedFPS = objProcessData.IREstimatedFPS
        objProcessData.objPlots.ColorEstimatedFPS = objProcessData.ColorEstimatedFPS
        objProcessData.objPlots.IREstimatedFPS = objProcessData.IREstimatedFPS

        processedBlue = objProcessData.ProcessedregionWindowBlueData
        processedGreen = objProcessData.ProcessedregionWindowGreenData
        processedRed = objProcessData.ProcessedregionWindowRedData
        processedGrey = objProcessData.ProcessedregionWindowGreyData
        processedIR = objProcessData.ProcessedregionWindowIRData
        objProcessData.GenerateGrapth(PlotType, processedBlue, processedGreen, processedRed, processedGrey,
                                      processedIR)
        print("Graph generated for " + PlotType)

    def sorted_nicely(self,l):
        """ Sort the given iterable in the way that humans expect."""
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def GenearateResults(self,objProcessingStorage):
        ##Genearte result
        processingStep="SaveResultstoDisk"
        for participant_number in self.objConfig.ParticipantNumbers:
            print(participant_number)
            for position in self.objConfig.hearratestatus:
                print(position)
                previousPath = self.objConfig.getPreviousStepDetail(processingStep)  # get previous step
                currentSavePath = self.objConfig.getPathforFolderName(processingStep + "DataFiles",
                                                                      participant_number, position)
                previousfullPath = self.objConfig.getPathforFolderName(previousPath, participant_number,
                                                                       position)
                currentSaveFileName = processingStep
                # if (not self.objFileIO.FileExits(currentSavePath + "ProcessedCompleted.txt")):
                for folderName in os.listdir(previousfullPath):  # Load previous step files, for each file
                    previousWindowFilesPath = previousfullPath + folderName + "\\"
                    print(previousWindowFilesPath)
                    objReliability = CheckReliability()  # for every type and participant
                    # Data Frame
                    dfWindows = pd.DataFrame(columns=self.HeaderRowSplit)
                    # allfiles = sorted(os.listdir(previousWindowFilesPath))
                    lst = sorted(os.listdir(previousWindowFilesPath))
                    lst.remove("ProcessedCompleted.txt")
                    lst.sort()
                    finallist = self.sorted_nicely(lst)
                    for fileName in finallist:  # Load all window files
                        # if ((not fileName.__contains__("ProcessedCompleted"))):
                        if ((not fileName.__contains__("Graphs"))):
                            previousProcessedStepData = self.objFileIO.ReadfromDisk(
                                previousWindowFilesPath, fileName)
                            currentSaveFileName = folderName
                            WindowCount = fileName.split("_")[1]
                            ##Extract Readings to a file
                            WindowDataRow = objProcessingStorage.ExtractReadings(
                                previousProcessedStepData, self.objConfig, fileName,
                                objReliability,participant_number,position)
                            splitWindowRow =WindowDataRow.split(",")
                            dfWindows.loc[WindowCount] = splitWindowRow

                    # dfWindows = dfWindows.sort_values(by=['WindowCount'], ascending=[True])
                    dfWindows.to_csv(currentSavePath + currentSaveFileName + ".csv")  # Write to file
                    del objReliability
                    self.objFileIO.WritedatatoFile(currentSavePath + '\\',
                                                   "ProcessedCompleted", "Completed")
    '''
    mainMethod: Execute program
    '''
    def mainMethod(self, generateFaceData,generateResults,generateStatResults):
        objProcessingStorage = InitiateProcessingStorage(self.objConfig)
        if(generateFaceData):
            self.GenerateFaceData()  # Run only once
        elif(generateResults):
            self.GenearateResults(objProcessingStorage)
        elif(generateStatResults):
            objResults =  ResultsandGraphs(self.objConfig,self.objFileIO)
            objResults.AllPlotsforComputedResults()
        else:
            for processingStep in self.objConfig.ProcessingSteps:
                print(processingStep)
                # if (processingStep == "SaveResultstoDisk"):
                #     continue
                if (processingStep != "ComputerHRandSPO"):
                    continue
                for process_type in self.objConfig.getProcessType(processingStep):
                    print("started all processing steps for window:" + str(self.objConfig.windowSize) + " ---> for type "+ str(process_type))
                    for participant_number in self.objConfig.ParticipantNumbers:
                        for position in self.objConfig.hearratestatus:
                            print(participant_number +" --> "+position)
                            previousPath = self.objConfig.getPreviousStepDetail(processingStep)  # get previous step
                            currentSavePath = self.objConfig.getPathforFolderName(processingStep + "DataFiles",
                                                                                  participant_number, position)
                            if (processingStep == "PreProcess"):  # ONLY RUN ONCE TO GET same lenght array to use all signal data in any algo
                                currentSavePath = self.objConfig.getPathforFolderName(processingStep + "DataFilesSameLength",
                                                                                      participant_number, position)
                            currentSaveFileName = processingStep + "_" + str(process_type)
                            previousfullPath = self.objConfig.getPathforFolderName(previousPath, participant_number,
                                                                                   position)
                            if (not self.objFileIO.FileExits(currentSavePath + currentSaveFileName + "\\" + "ProcessedCompleted.txt")):
                                if (str(process_type).__contains__("Combined")):  # ONLY RUN ONCE TO GET same lenght array to use all signal data in any algo
                                    previousfullPath = self.objConfig.getPathforFolderName(previousPath + "SameLength",
                                                                                           participant_number, position)  #
                                if (previousfullPath.__contains__("BinaryFaceROI")):
                                    for fileName in os.listdir(previousfullPath):  # Load previous step files, for each file
                                        previousProcessedStepData = self.objFileIO.ReadfromDisk(previousfullPath, fileName)
                                        objProcessingStorage.PreProcessDataWindow(previousProcessedStepData, self.objConfig,
                                                                                  processingStep, process_type,
                                                                                  currentSavePath, fileName,
                                                                                  currentSaveFileName,
                                                                                  self.objConfig.windowSize)
                                else:
                                    for folderName in os.listdir(previousfullPath):  # Load previous step files, for each file
                                        previousWindowFilesPath = previousfullPath + folderName + "\\"
                                        fileExsits = self.objFileIO.FileExits(currentSavePath + processingStep + "_" + str(
                                            process_type) + "_" + folderName + "\\" + "ProcessedCompleted.txt")
                                        if (not fileExsits):
                                            for fileName in os.listdir(previousWindowFilesPath):  # Load all window files
                                                if ((not fileName.__contains__("ProcessedCompleted"))):
                                                    if ((not fileName.__contains__("Graphs"))):
                                                        previousProcessedStepData = self.objFileIO.ReadfromDisk(
                                                            previousWindowFilesPath, fileName)
                                                        currentSaveFileName = processingStep + "_" + str(
                                                            process_type) + "_" + folderName  # previousType[0] + "_" + previousType[1]
                                                        WindowCount = fileName.split("_")[1]
                                                        existingfilePath = currentSavePath + currentSaveFileName + "\\" + "ProcessedWindow_" + str(
                                                            WindowCount)
                                                        isonDisk = self.objFileIO.FileExits(existingfilePath)
                                                        if (not isonDisk):
                                                            objProcessingStorage.ProcessDatainStepsWindow(
                                                                previousProcessedStepData, self.objConfig, processingStep,
                                                                process_type,
                                                                currentSavePath, fileName, currentSaveFileName)
                                            self.objFileIO.WritedatatoFile(currentSavePath + currentSaveFileName + '\\',
                                                                           "ProcessedCompleted", "Completed")


        del objProcessingStorage


# Skin_Group_Types = ['OtherAsian_OtherSkin_Group','SouthAsian_BrownSkin_Group','Europe_WhiteSkin_Group' ]
# for skintype in skinGroup:

generateFaceData = False # Update to False when face data is generated
generateResults = False
generateStatResults =  True
objMain = Main()  # Main(skintype) Pass type none here to process all skin types
objMain.mainMethod(generateFaceData,generateResults,generateStatResults)  # Send true to generate binary object data holding images in arrays meaned
# objMain.GenerateGraphfromStoredFile()
# objMain.LoadComputedResults()
# objMain.AllPlotsforComputedResults(PreProcess,position)
del objMain
print('Program Ended')

##Todo: add
#https://github.com/Aura-healthcare/hrv-analysis
#https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html
