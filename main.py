import os

import CommonMethods
from Configurations import Configurations
from FileIO import FileIO
from LoadFaceData import LoadFaceData
from ProcessParticipantsData import Process_SingalData, objFile
from GlobalDataFile import GlobalData

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
    ParticipantsProcessedBloodOxygenData  = {}
    HRNameFileName = "AvgHRdata_"
    SPONameFileName = "AvgSPOdata_"
    CaseList = []

    #Constructor
    def __init__(self, skinGroup='None'):
        self.objConfig = Configurations(True, skinGroup)

    """
    getEstimatedFrame:
    """
    def getEstimatedFrame(self, objFaceImage,distnacepath):
        ColorfpswithTime, IRfpswithTime, isVariable, isIRVariable = objFaceImage.GetEstimatedFPS(distnacepath) #, ColorFPS, IRFPS
        return ColorfpswithTime, IRfpswithTime, isVariable, isIRVariable #, ColorFPS, IRFPS

    """
    getData:
    """
    def getData(self, participant_number, position, region,objFaceImage):
        # estimatedfps=0

        ##get loading path
        LoadColordataPath, LoadIRdataPath, LoadDistancePath,ProcessedDataPath = self.objConfig.getLoadPath(participant_number, position,
                                                                                         region)
        # Load Roi data (ALL)
        # print("Loading and processing color roi data")
        objFaceImage.ProcessColorImagestoArray(LoadColordataPath)

        # print("Loading and processing ir roi data")
        objFaceImage.ProcessIRImagestoArray(LoadIRdataPath)

        # GET FPS and distance and other data
        # if(region == 'lips'):
        objFaceImage.GetDistance(LoadDistancePath) #TODO: DOUBLE CHECK ARRAY SIZE

        # Create global data object and use dictionary (ROI Store) to uniquely store a regions data
        self.ROIStore[region] =  GlobalData(objFaceImage.time_list_color, objFaceImage.timecolorCount,
                                           objFaceImage.time_list_ir, objFaceImage.timeirCount,
                                           objFaceImage.Frametime_list_ir, objFaceImage.Frametime_list_color,
                                           objFaceImage.red, objFaceImage.green, objFaceImage.blue, objFaceImage.grey,
                                           objFaceImage.Irchannel, objFaceImage.distanceM,
                                           objFaceImage.totalTimeinSeconds,objFaceImage.ColorEstimatedFPS, objFaceImage.IREstimatedFPS,
                                            objFaceImage.ColorfpswithTime, objFaceImage.IRfpswithTime)

        return objFaceImage.totalTimeinSeconds
        # return estimatedfps

    """
    GenerateResultsfromParticipants:
    """
    def GenerateResultsfromParticipants(self,participant_number,position,HrGr,SpoGr,fileName ,algoType,fftype ,filtertype , resulttype , preprocesstype , isSmooth):
        self.objConfig.SavePath = self.objConfig.SavePath + 'RawOriginal\\' #+fileName + '\\'
        self.objFileIO.CreatePath(self.objConfig.SavePath)
        # if(not ParticipantsHRfileName in ParticipantsProcessedHeartRateData):
        # Generate Data for all Techniques
        ListHrdata, ListSPOdata, IsSuccess = Process_SingalData(
            self.objConfig.RunAnalysisForEntireSignalData,
            self.ROIStore, self.objConfig.SavePath,
            algoType, fftype,
            HrGr, SpoGr,
            filtertype, resulttype, preprocesstype, isSmooth, 0)
        # hrType, isCompressed,snrType)

        # Save to list for writing to disk later
        objFile.WriteListDatatoFile(self.objConfig.SavePath, 'HeartRate_' + fileName,
                                    ListHrdata)
        objFile.WriteListDatatoFile(self.objConfig.SavePath, 'SPO_' + fileName,
                                    ListSPOdata)
            # ParticipantsProcessedHeartRateData[ParticipantsHRfileName] = ListHrdata
            # ParticipantsProcessedBloodOxygenData[ParticipantsSPOfileName] = ListSPOdata
            # print(str(algoType) +"_"+ str(fftype) +"_"+ str(filtertype)  +"_"+ str(resulttype)+
            #       "_" +  str(preprocesstype)  +"_"+  str(isSmooth)  +"_"+str(hrType)+"_" +
            #       str(isCompressed)  +"_"+ str(snrType))
            # objConfig.RunAnalysisForEntireSignalData, ROIStore, objConfig.SavePath, algoType, fftype, HrGr, SpoGr, filtertype, resulttype, preprocesstype, isSmooth,hrType,isCompressed,snrType
            # objConfig.Processed_participants_data[participant_number] =

        # Start generating data fo following types
        # for preprocesstype in self.objConfig.preprocesses:
        #     for algoType in self.objConfig.AlgoList:
        #         # Create path for each algorithm type to store results in
        #         # objFileIO.CreatePath(objConfig.SavePath + algoType)
        #         for fftype in self.objConfig.fftTypeList:
        #             for resulttype in self.objConfig.resulttypeList:
        #                 # for hrType in self.objConfig.hrTypeList:
        #                 for filtertype in self.objConfig.filtertypeList:
        #                     for isSmooth in self.objConfig.Smoothen:
        #                         #TODO: CHECK IF COMPRESSED are neccessary
        #                         # for isCompressed in self.objConfig.Compressed:
        #                         # for snrType in self.objConfig.SNRTypeList:
        #                         # Sort to list for writing to disk later
        #                         #+ "_HR-" + str(hrType)+ "_CP-" + str('None')
        #                         fileName = algoType + "_FFT-" + str(fftype) + "_FL-" + str(
        #                             filtertype) + "_RS-" + str(resulttype) + "_PR-" + str(preprocesstype) + "_SM-" + str(
        #                             isSmooth) #+ "_SNR-" + str(snrType) #isCompressed
        #                         print(fileName)
        #

            # Write all results to file
            # for k, v in ParticipantsProcessedHeartRateData.items():
            #     fileDetails = k.split('*')
            #     participantnumber = fileDetails[0]
            #     fileName = fileDetails[1]
            #     objFile.WriteListDatatoFile(objConfig.SavePath, fileName, v)
            #
            # for k, v in ParticipantsProcessedBloodOxygenData.items():
            #     fileDetails = k.split('*')
            #     participantnumber = fileDetails[0]
            #     fileName = fileDetails[1]
            #     objFile.WriteListDatatoFile(objConfig.SavePath, fileName, v)
            #
            # ParticipantsProcessedHeartRateData = {}
            # ParticipantsProcessedBloodOxygenData = {}


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
                                fileName = algoType + "_FFT-" + str(fftype) + "_FL-" + str(
                                    filtertype) + "_RS-" + str(resulttype) + "_PR-" + str(
                                    preprocesstype) + "_SM-" + str(isSmooth)
                                # print(fileName)
                                # if(fftype == 'M4'): #Run only once when all cases are done to generate filtering for process type 1 with M4 FFT
                                #     # # Best works with filter type 1 ## later try with filte filter 6 and 4 as well
                                #     # self.Filter_type = 1
                                #     fileName=fileName.replace('_FL-'+ str(filtertype), '_FL-'+ str(1))

                                if(fileName not in self.CaseList):
                                    self.CaseList.append(fileName)

    def CheckIfGenerated(self,fileName):
        # SavePath = self.objConfig.SavePath + fileName + '\\'
        pathExsists = objFile.FileExits(self.objConfig.SavePath + 'HeartRate_' + fileName + ".txt")

        #already generated
        if(pathExsists):
            return True

        return  False

    def mainMethod(self):
        #Check cases
        self.GenerateCases()
        prevParticipantId = ''
        for participant_number in self.objConfig.ParticipantNumbers:
            # for each heart rate status (resting or active)
            for position in self.objConfig.hearratestatus:
                print(participant_number + ', ' + position)
                prevParticipantId = ''

                # set path
                self.objConfig.setSavePath(participant_number, position,'WindowProcessedData')

                for case in self.CaseList:
                    IsGenerated = self.CheckIfGenerated(case)
                    if(IsGenerated):
                        stop = True
                    else:
                        splitCase = case.split('_')
                        fileName = case
                        algoType = splitCase[0]
                        fftype = splitCase[1].replace('FFT-','')
                        filtertype = int(splitCase[2].replace('FL-',''))
                        resulttype = int(splitCase[3].replace('RS-',''))
                        preprocesstype = int(splitCase[4].replace('PR-',''))
                        isSmooth = splitCase[5].replace('SM-','')
                        if (isSmooth == 'True'):
                            isSmooth = True
                        else:
                            isSmooth = False
                        #Gneerate
                        # Load all data for each roi and create roi store
                        if(prevParticipantId != participant_number):
                            prevParticipantId = participant_number
                            # estimatedfps = 0
                            totalTimeinSeconds= 0
                            for region in self.objConfig.roiregions:
                                # Init for each region
                                objFaceImage = LoadFaceData()
                                objFaceImage.Clear()

                                ##get ROI Store
                                totalTimeinSeconds = self.getData(participant_number, position, region, objFaceImage)

                                # objFaceImage.EstimatedFPS = estimatedfps
                                # delete face image object
                                del objFaceImage

                            # # Print FPS Detail
                            # print('ESTIMATED FPS USED: ' + str(estimatedfps))
                            # print(str(participant_number) + ', type= ' + str(position) + ' color min FPS: ' + str(
                            #     min(ColorfpswithTime.values())) + ' ' + str(isVariable) +
                            #       ', IR min FPS: ' + str(min(IRfpswithTime.values())) + ' ' + str(isIRVariable))

                        ###Get ground Truth
                        HrGr, SpoGr = CommonMethods.GetGroundTruth(participant_number, position,
                                                                           self.objConfig.DiskPath,int(totalTimeinSeconds))

                        ##Process and get result of participants data
                        self.GenerateResultsfromParticipants(participant_number, position, HrGr, SpoGr,fileName ,algoType,fftype ,filtertype , resulttype , preprocesstype , isSmooth)

                    stop=0


objMain = Main('Europe_WhiteSkin_Group') #Add none here to process all skin types
objMain.mainMethod()
print('Ended')