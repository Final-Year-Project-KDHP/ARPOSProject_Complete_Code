import CommonMethods
from Configurations import Configurations
from FileIO import FileIO
from LoadFaceData import LoadFaceData
from ProcessParticipantsData import Process_SingalData, objFile
from GlobalDataFile import GlobalData, WindowProcessedData

"""
Main Class:
"""


class Main:
    # Store Region of Interest and Results
    ROIStore = {}

    # Global Objects
    objConfig = None
    objFileIO = FileIO()
    ParticipantsProcessedHeartRateData = {}
    ParticipantsProcessedBloodOxygenData = {}
    HRNameFileName = "AvgHRdata_"
    SPONameFileName = "AvgSPOdata_"
    CaseList = []

    # Constructor
    def __init__(self, skinGroup='None'):
        self.objConfig = Configurations(True, skinGroup)

    """
    Gemerate cases:
    """

    def GenerateCases(self):
        self.CaseList = []
        for preprocesstype in self.objConfig.preprocesses:
            for algoType in self.objConfig.AlgoList:
                for fftype in self.objConfig.fftTypeList:
                    for resulttype in self.objConfig.resulttypeList:
                        for filtertype in self.objConfig.filtertypeList:
                            for isSmooth in self.objConfig.Smoothen:
                                fileName = algoType + "_PR-" + str(preprocesstype) + "_FFT-" + str(
                                    fftype) + "_FL-" + str(filtertype) \
                                           + "_RS-" + str(resulttype) + "_SM-" + str(isSmooth)
                                if (fileName not in self.CaseList):
                                    self.CaseList.append(fileName)

    def CheckIfGenerated(self, fileName):
        # SavePath = self.objConfig.SavePath + fileName + '\\'
        pathExsists = objFile.FileExits(self.objConfig.SavePath + 'Result\\' + 'HRSPOwithLog_' + fileName + ".txt")
        # already generated
        if (pathExsists):
            return True
        return False

    """
     GenerateResultsfromParticipants:
     """

    def GenerateResultsfromParticipants(self, ParticipantsOriginalDATA,typeProcessing):
        self.GenerateCases()
        TotalCasesCount = len(self.CaseList)
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                print(participant_number + ', ' + position)
                # ParticipantsOriginalDATA[participant_number + '_' + position] = objWindowProcessedData
                objWindowProcessedData = ParticipantsOriginalDATA.get(participant_number + '_' + position)
                self.objConfig.setSavePath(participant_number, position,typeProcessing)
                currentCasesDone = 0
                for case in self.CaseList:
                    IsGenerated = self.CheckIfGenerated(case)
                    currentCasesDone = currentCasesDone + 1
                    currentPercentage = ((currentCasesDone/TotalCasesCount)*100)
                    if (IsGenerated):
                        continue
                    else:
                        print(case + '  -> ' + str(currentPercentage) + ' out of 100%')
                        splitCase = case.split('_')
                        fileName = case
                        algoType = splitCase[0]
                        fftype = splitCase[2].replace('FFT-', '')
                        filtertype = int(splitCase[3].replace('FL-', ''))
                        resulttype = int(splitCase[4].replace('RS-', ''))
                        preprocesstype = int(splitCase[1].replace('PR-', ''))
                        isSmooth = splitCase[5].replace('SM-', '')
                        if (isSmooth == 'True'):
                            isSmooth = True
                        else:
                            isSmooth = False

                        # Generate Data for all Techniques
                        Process_SingalData(
                            self.objConfig.RunAnalysisForEntireSignalData,
                            objWindowProcessedData.ROIStore, self.objConfig.SavePath,
                            algoType, fftype,
                            filtertype, resulttype, preprocesstype, isSmooth,
                            objWindowProcessedData.HrGroundTruthList, objWindowProcessedData.SPOGroundTruthList,
                            fileName, self.objConfig.DumpToDisk)
                ParticipantsOriginalDATA.pop(participant_number + '_' + position)

    def LoadandGenerateFaceDatatoBianryFiles(self):
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                self.objConfig.setSavePath(participant_number, position, 'RawOriginal')  # set path
                print('Loading and generating FaceData for ' + participant_number + ', ' + position)
                print('Loading from path ' + self.objConfig.DiskPath)
                print('Storing data to path ' + self.objConfig.SavePath)
                # generate only if does not exist
                if (not self.objFileIO.FileExits(self.objConfig.SavePath + 'UnCompressedBinaryLoadedData')):
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
                        objFaceImage.ProcessColorImagestoArray(LoadColordataPath)

                        # print("Loading and processing ir roi data")
                        objFaceImage.ProcessIRImagestoArray(LoadIRdataPath, LoadDistancePath)

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

                    ###Get ground Truth
                    HrGr, SpoGr = CommonMethods.GetGroundTruth(participant_number, position,
                                                               self.objConfig.DiskPath, int(self.ROIStore.get(
                            self.objConfig.roiregions[0]).totalTimeinSeconds))

                    ##Original Data storage
                    objWindowProcessedData = WindowProcessedData()
                    objWindowProcessedData.HrGroundTruthList = HrGr
                    objWindowProcessedData.SPOGroundTruthList = SpoGr
                    objWindowProcessedData.ColorLengthofAllFrames = self.ROIStore.get(
                        self.objConfig.roiregions[0]).getLengthColor()
                    objWindowProcessedData.IRLengthofAllFrames = self.ROIStore.get(
                        self.objConfig.roiregions[0]).getLengthIR()
                    objWindowProcessedData.TimeinSeconds = int(
                        self.ROIStore.get(self.objConfig.roiregions[0]).totalTimeinSeconds)
                    objWindowProcessedData.ROIStore = self.ROIStore

                    self.objFileIO.DumpObjecttoDisk(self.objConfig.SavePath, 'UnCompressedBinaryLoadedData',
                                                    objWindowProcessedData)

                    del objWindowProcessedData

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

    def mainMethod(self, generateBinaryData):
        # For generating meaned channel arrays of image and other required data in form of a objectProcessedData (See class WindowProcessedData)
        # Reqruied to run only once, binary data
        if (generateBinaryData):  # RUN only once
            self.LoadandGenerateFaceDatatoBianryFiles()
        else:
            # Load Data from path
            # ParticipantsOriginalDATA = self.LoadBinaryData()
            #Process for entire signal or in windows
            FolderNameforSave = 'ProcessedDataWindows'
            if(self.objConfig.RunAnalysisForEntireSignalData):
                FolderNameforSave= 'ProcessedData'

            #  Load Data from path and Process Data
            self.GenerateResultsfromParticipants(self.LoadBinaryData(),FolderNameforSave)#FOR Window processing


skintype = 'Europe_WhiteSkin_Group'
print('Program started for ' +skintype)
objMain = Main(skintype)  # Add none here to process all skin types [Europe_WhiteSkin_Group,SouthAsian_BrownSkin_Group,OtherAsian_OtherSkin_Group]
objMain.mainMethod(False)  # Send true to generate binary object data holding images in arrays meaned
print('Program Ended')
