import CommonMethods
from Configurations import Configurations
from LoadFaceData import LoadFaceData
from ProcessParticipantsData import Process_Participants_Data_Windows
from GlobalDataFile import GlobalData

# Store Region of Interest and Results
ROIStore = {}

# Global Objects
objConfig = Configurations()

# For each participant
for participant_number in objConfig.ParticipantNumbers:

    # for each heart rate status (resting or active)
    for position in objConfig.hearratestatus:
        SavePath = objConfig.getSavePath(participant_number, position)

        # Load all data for each roi and create roi store
        for region in objConfig.roiregions:
            # Init for each region
            objFaceImage = LoadFaceData()

            ##get loading path
            LoadColordataPath, LoadIRdataPath, LoadDistancePath = objConfig.getLoadPath(participant_number, position,
                                                                                        region)

            # Load Roi data (ALL)
            print("Loading and processing color roi data")
            objFaceImage.ProcessColorImagestoArray(LoadColordataPath)
            print("Loading and processing ir roi data")
            objFaceImage.ProcessIRImagestoArray(LoadIRdataPath, LoadDistancePath)

            #Create global data object and use dictionary (ROI Store) to uniquely store a regions data
            ROIStore[region] = GlobalData(objFaceImage.time_list_color, objFaceImage.timecolorCount,
                                          objFaceImage.time_list_ir, objFaceImage.timeirCount,
                                          objFaceImage.Frametime_list_ir, objFaceImage.Frametime_list_color,
                                          objFaceImage.red, objFaceImage.green, objFaceImage.blue, objFaceImage.grey,
                                          objFaceImage.ir, objFaceImage.distanceM)
            #delete face image object
            del objFaceImage

        ###Get ground Truth
        HrGr, SpoGr = CommonMethods.GetGroundTruth(participant_number, position)

        #Start generating data fo following types
        for algoType in objConfig.AlgoList:
            for fftype in objConfig.fftTypeList:
                for resulttype in objConfig.resulttypeList:
                    for filtertype in objConfig.filtertypeList:
                        for preprocesstype in objConfig.preprocesses:
                            for isSmooth in objConfig.Smoothen:
                                # Generate Data for all Techniques
                                Process_Participants_Data_Windows(ROIStore, SavePath, objConfig.DiskPath,
                                                                  objConfig.ParticipantNumber,
                                                                  position, algoType, fftype, HrGr, filtertype,
                                                                  resulttype,
                                                                  preprocesstype, isSmooth)
