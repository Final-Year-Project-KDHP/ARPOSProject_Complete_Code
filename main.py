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
    objConfig = Configurations(True)
    objFileIO = FileIO()
    ParticipantsProcessedHeartRateData = {}
    ParticipantsProcessedBloodOxygenData  = {}
    HRNameFileName = "AvgHRdata_"
    SPONameFileName = "AvgSPOdata_"

    """
    getEstimatedFrame:
    """
    def getEstimatedFrame(self, objFaceImage):
        ColorfpswithTime, IRfpswithTime, isVariable, isIRVariable = objFaceImage.GetEstimatedFPS()
        return ColorfpswithTime, IRfpswithTime, isVariable, isIRVariable

    """
    getData:
    """
    def getData(self, participant_number, position, region,objFaceImage):

        ##get loading path
        LoadColordataPath, LoadIRdataPath, LoadDistancePath = self.objConfig.getLoadPath(participant_number, position,
                                                                                         region)
        # Load Roi data (ALL)
        # print("Loading and processing color roi data")
        objFaceImage.ProcessColorImagestoArray(LoadColordataPath)
        # print("Loading and processing ir roi data")
        objFaceImage.ProcessIRImagestoArray(LoadIRdataPath, LoadDistancePath)

        # Create global data object and use dictionary (ROI Store) to uniquely store a regions data
        self.ROIStore[region] = GlobalData(objFaceImage.time_list_color, objFaceImage.timecolorCount,
                                           objFaceImage.time_list_ir, objFaceImage.timeirCount,
                                           objFaceImage.Frametime_list_ir, objFaceImage.Frametime_list_color,
                                           objFaceImage.red, objFaceImage.green, objFaceImage.blue, objFaceImage.grey,
                                           objFaceImage.Irchannel, objFaceImage.distanceM,
                                           objFaceImage.totalTimeinSeconds)

    """
    GenerateResultsfromParticipants:
    """
    def GenerateResultsfromParticipants(self,participant_number,position,HrGr,SpoGr):
        # Start generating data fo following types
        for preprocesstype in self.objConfig.preprocesses:
            for algoType in self.objConfig.AlgoList:
                # Create path for each algorithm type to store results in
                # objFileIO.CreatePath(objConfig.SavePath + algoType)
                for fftype in self.objConfig.fftTypeList:
                    for resulttype in self.objConfig.resulttypeList:
                        # for hrType in self.objConfig.hrTypeList:
                        for filtertype in self.objConfig.filtertypeList:
                            for isSmooth in self.objConfig.Smoothen:
                                #TODO: CHECK IF COMPRESSED are neccessary
                                # for isCompressed in self.objConfig.Compressed:
                                # for snrType in self.objConfig.SNRTypeList:
                                # Sort to list for writing to disk later
                                #+ "_HR-" + str(hrType)+ "_CP-" + str('None')
                                fileName = algoType + "_FFT-" + str(fftype) + "_FL-" + str(
                                    filtertype) + "_RS-" + str(resulttype) + "_PR-" + str(preprocesstype) + "_SM-" + str(
                                    isSmooth) #+ "_SNR-" + str(snrType) #isCompressed
                                print(fileName)
                                self.objConfig.SavePath = self.objConfig.DiskPath  + '\\Result\\' + participant_number + '\\' + position + '\\' + fileName + '\\'
                                self.objFileIO.CreatePath(self.objConfig.SavePath)
                                # ParticipantsHRfileName = participant_number + "*" + self.HRNameFileName + fileName  # filename
                                # ParticipantsSPOfileName = participant_number + "*" + self.HRNameFileName + fileName  # filename

                                if (not os.path.exists(self.objConfig.SavePath + self.HRNameFileName + fileName + ".txt")):
                                    # if(not ParticipantsHRfileName in ParticipantsProcessedHeartRateData):
                                    # Generate Data for all Techniques
                                    ListHrdata, ListSPOdata = Process_SingalData(
                                        self.objConfig.RunAnalysisForEntireSignalData,
                                        self.ROIStore, self.objConfig.SavePath,
                                        algoType, fftype,
                                        HrGr, SpoGr,
                                        filtertype, resulttype, preprocesstype, isSmooth,0)
                                        #hrType, isCompressed,snrType)

                                    # Save to list for writing to disk later
                                    objFile.WriteListDatatoFile(self.objConfig.SavePath,'HeartRate_' + fileName,
                                                                ListHrdata)
                                    objFile.WriteListDatatoFile(self.objConfig.SavePath, 'SPO_' +fileName,
                                                                    ListSPOdata)
                                        # ParticipantsProcessedHeartRateData[ParticipantsHRfileName] = ListHrdata
                                        # ParticipantsProcessedBloodOxygenData[ParticipantsSPOfileName] = ListSPOdata
                                    # print(str(algoType) +"_"+ str(fftype) +"_"+ str(filtertype)  +"_"+ str(resulttype)+
                                    #       "_" +  str(preprocesstype)  +"_"+  str(isSmooth)  +"_"+str(hrType)+"_" +
                                    #       str(isCompressed)  +"_"+ str(snrType))
                                    # objConfig.RunAnalysisForEntireSignalData, ROIStore, objConfig.SavePath, algoType, fftype, HrGr, SpoGr, filtertype, resulttype, preprocesstype, isSmooth,hrType,isCompressed,snrType
                                    # objConfig.Processed_participants_data[participant_number] =

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

    def mainMethod(self):
        # For each participant
        for participant_number in self.objConfig.ParticipantNumbers:
            print(participant_number)
            # for each heart rate status (resting or active)
            for position in self.objConfig.hearratestatus:
                print(position)
                self.objConfig.setSavePath(participant_number, position)

                # Load all data for each roi and create roi store
                for region in self.objConfig.roiregions:
                    # Init for each region
                    objFaceImage = LoadFaceData()
                    objFaceImage.Clear()

                    ##get ROI Store
                    self.getData(participant_number, position, region,objFaceImage)

                    # Get FPS
                    # if(region == 'lips'):
                    ColorfpswithTime, IRfpswithTime, isVariable, isIRVariable = self.getEstimatedFrame(objFaceImage)

                    # delete face image object
                    del objFaceImage

                #Print FPS Detail
                # print(str(participant_number) + ', type= ' + str(position) + ' color FPS: ' + str(
                #     min(ColorfpswithTime.values())) + ' ' + str(isVariable) +
                #       ', IR FPS: ' + str(min(IRfpswithTime.values())) + ' ' + str(isIRVariable))

                ###Get ground Truth
                HrGr, SpoGr = CommonMethods.GetGroundTruth(participant_number, position, self.objConfig.DiskPath)

                ##Process and get result of participants data
                self.GenerateResultsfromParticipants(participant_number,position, HrGr, SpoGr)

objMain = Main()
objMain.mainMethod()
print('Ended')