import os

import CommonMethods
from Configurations import Configurations
from FileIO import FileIO
from LoadFaceData import LoadFaceData
from ProcessParticipantsData import Process_SingalData, objFile
from GlobalDataFile import GlobalData

# Store Region of Interest and Results
ROIStore = {}

# Global Objects
objConfig = Configurations()
objFileIO = FileIO()
ParticipantsProcessedHeartRateData = {}
ParticipantsProcessedBloodOxygenData  = {}
HRNameFileName = "AvgHRdata_"
SPONameFileName = "AvgSPOdata_"

# For each participant
for participant_number in objConfig.ParticipantNumbers:

    # for each heart rate status (resting or active)
    for position in objConfig.hearratestatus:

        objConfig.setSavePath(participant_number, position)

        # Load all data for each roi and create roi store
        for region in objConfig.roiregions:
            # Init for each region
            objFaceImage = LoadFaceData()
            objFaceImage.Clear()

            ##get loading path
            LoadColordataPath, LoadIRdataPath, LoadDistancePath = objConfig.getLoadPath(participant_number, position,
                                                                                        region)

            # Load Roi data (ALL)
            print("Loading and processing color roi data")
            objFaceImage.ProcessColorImagestoArray(LoadColordataPath)
            print("Loading and processing ir roi data")
            objFaceImage.ProcessIRImagestoArray(LoadIRdataPath, LoadDistancePath)

            # Create global data object and use dictionary (ROI Store) to uniquely store a regions data
            ROIStore[region] = GlobalData(objFaceImage.time_list_color, objFaceImage.timecolorCount,
                                          objFaceImage.time_list_ir, objFaceImage.timeirCount,
                                          objFaceImage.Frametime_list_ir, objFaceImage.Frametime_list_color,
                                          objFaceImage.red, objFaceImage.green, objFaceImage.blue, objFaceImage.grey,
                                          objFaceImage.Irchannel, objFaceImage.distanceM,
                                          objFaceImage.totalTimeinSeconds)
            # delete face image object
            del objFaceImage

        ###Get ground Truth
        HrGr, SpoGr = CommonMethods.GetGroundTruth(participant_number, position, objConfig.DiskPath)

        # Start generating data fo following types
        for algoType in objConfig.AlgoList:
            # Create path for each algorithm type to store results in
            # objFileIO.CreatePath(objConfig.SavePath + algoType)
            for fftype in objConfig.fftTypeList:
                for resulttype in objConfig.resulttypeList:
                    for hrType in objConfig.hrTypeList:
                        for filtertype in objConfig.filtertypeList:
                            for preprocesstype in objConfig.preprocesses:
                                for isSmooth in objConfig.Smoothen:
                                    for isCompressed in objConfig.Compressed:
                                        for snrType in objConfig.SNRTypeList:

                                            #Sort to list for writing to disk later
                                            fileName = algoType + "_FFT-" + str(fftype) + "_FL-" + str(
                                                filtertype) + "_RS-" + str(resulttype) + "_HR-" + str(
                                                hrType) + "_PR-" + str(preprocesstype) + "_SM-" + str(
                                                isSmooth) + "_CP-" + str(isCompressed)+ "_SNR-" + str(snrType)
                                            ParticipantsHRfileName = participant_number + "*" + HRNameFileName + fileName  # filename
                                            ParticipantsSPOfileName = participant_number + "*" + HRNameFileName + fileName  # filename

                                            if(not os.path.exists(objConfig.SavePath + HRNameFileName + fileName + ".txt")):
                                                # if(not ParticipantsHRfileName in ParticipantsProcessedHeartRateData):
                                                    # Generate Data for all Techniques
                                                ListHrdata, ListSPOdata = Process_SingalData(
                                                    objConfig.RunAnalysisForEntireSignalData,
                                                    ROIStore, objConfig.SavePath,
                                                    algoType, fftype,
                                                    HrGr, SpoGr,
                                                    filtertype, resulttype, preprocesstype, isSmooth, hrType, isCompressed,
                                                    snrType)

                                                # Save to list for writing to disk later
                                                objFile.WriteListDatatoFile(objConfig.SavePath, fileName, ListHrdata)
                                                objFile.WriteListDatatoFile(objConfig.SavePath, fileName, ListSPOdata)
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


