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


    def CustomCaseList(self):
        # CustomCases = self.objFile.ReaddatafromFile(self.objConfig.DiskPath,'NoHrFilesCases')
        CustomCases = []
        #FOR White skin
        ###second with rs=2 only

        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M4_FL-3_RS-1_SM-True') #-->4497

        CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M4_FL-5_RS-1_SM-False') #-->2047,2212,8343
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M4_FL-3_RS-1_SM-False') #-->2047,2212,8343

        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-3_RS-1_SM-True') #-->2212
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M4_FL-5_RS-1_SM-False') #-->2212
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M4_FL-3_RS-1_SM-False') #-->2212

        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M4_FL-5_RS-1_SM-False') #-->8343
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M4_FL-3_RS-1_SM-False') #-->8343
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-7_RS-1_SM-True') #-->8343
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M1_FL-7_RS-1_SM-True') #--> 8343
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-7_RS-1_SM-True') #--> 8343
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M1_FL-6_RS-2_SM-False') #--> 8343
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M1_FL-7_RS-1_SM-False') #--> 8343
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-7_RS-1_SM-False') #--> 8343
        #........................................
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-1_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-1_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-6_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-6_RS-2_SM-False')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M1_FL-1_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M1_FL-1_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M1_FL-6_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M1_FL-6_RS-2_SM-False')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M1_FL-1_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M1_FL-2_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M1_FL-6_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M1_FL-6_RS-2_SM-False')
        ############################
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-3_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-3_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-5_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-5_RS-2_SM-False')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M1_FL-3_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M1_FL-3_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M1_FL-5_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M1_FL-5_RS-2_SM-False')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M1_FL-3_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M1_FL-3_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M1_FL-5_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M1_FL-5_RS-2_SM-False')

        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-3_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-3_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-5_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-5_RS-2_SM-False')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M2_FL-3_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M2_FL-3_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M2_FL-5_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M2_FL-5_RS-2_SM-False')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M2_FL-3_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M2_FL-3_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M2_FL-5_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M2_FL-5_RS-2_SM-False')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-3_FFT-M2_FL-3_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-3_FFT-M2_FL-3_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-3_FFT-M2_FL-5_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-3_FFT-M2_FL-5_RS-2_SM-False')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-4_FFT-M2_FL-3_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-4_FFT-M2_FL-3_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-4_FFT-M2_FL-5_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-4_FFT-M2_FL-5_RS-2_SM-False')
        #
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M2_FL-3_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M2_FL-3_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M2_FL-5_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M2_FL-5_RS-2_SM-False')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-2_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M4_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M1_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M1_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M1_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M2_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M2_FL-2_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M2_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M2_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M4_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M5_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M1_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M1_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M1_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M2_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M2_FL-2_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M2_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M2_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M4_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M5_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-2_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M1_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M1_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M1_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M2_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M2_FL-2_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M2_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M2_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M4_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-7_FFT-M1_FL-6_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-7_FFT-M5_FL-6_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-7_FFT-M1_FL-6_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-7_FFT-M5_FL-6_RS-2_SM-TRUE')
        ################First attempt below
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M4_FL-1_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M4_FL-2_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M5_FL-1_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M5_FL-2_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M5_FL-3_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M5_FL-5_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-5_FFT-M4_FL-1_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-5_FFT-M4_FL-2_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-5_FFT-M5_FL-1_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-5_FFT-M5_FL-2_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-5_FFT-M5_FL-3_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-5_FFT-M5_FL-5_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-5_FFT-M4_FL-1_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-5_FFT-M4_FL-2_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-5_FFT-M5_FL-1_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-5_FFT-M5_FL-2_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-5_FFT-M5_FL-3_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-5_FFT-M5_FL-5_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-5_FFT-M4_FL-1_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-5_FFT-M4_FL-2_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-5_FFT-M5_FL-1_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-5_FFT-M5_FL-2_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-5_FFT-M5_FL-3_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-5_FFT-M5_FL-5_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_None_PR-5_FFT-M4_FL-1_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_None_PR-5_FFT-M4_FL-3_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_None_PR-5_FFT-M4_FL-5_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_None_PR-5_FFT-M5_FL-5_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M4_FL-3_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M5_FL-2_RS-3_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M5_FL-5_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M6_FL-2_RS-3_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-2_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-2_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-3_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-5_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M4_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M4_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M5_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M5_FL-2_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M5_FL-3_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M5_FL-5_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M6_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M6_FL-2_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M5_FL-2_RS-3_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M5_FL-5_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M6_FL-2_RS-3_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M1_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M1_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M1_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M1_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M2_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M2_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M2_FL-2_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M2_FL-2_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M2_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M2_FL-3_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M2_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M2_FL-5_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M4_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M4_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M5_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M5_FL-2_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M5_FL-3_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M5_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M5_FL-5_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M6_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M6_FL-2_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M4_FL-1_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M4_FL-3_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M5_FL-2_RS-3_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M5_FL-5_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M6_FL-2_RS-3_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M1_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M1_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M1_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M1_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M2_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M2_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M2_FL-2_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M2_FL-2_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M2_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M2_FL-3_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M2_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M2_FL-5_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M4_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M4_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M5_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M5_FL-2_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M5_FL-3_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M5_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M5_FL-5_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M6_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M6_FL-2_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M1_FL-2_RS-3_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-2_RS-3_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M4_FL-2_RS-3_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-2_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-2_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-3_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-5_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M4_FL-3_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M5_FL-2_RS-3_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M5_FL-5_RS-1_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M6_FL-2_RS-3_SM-FALSE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M1_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M1_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M1_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M1_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M2_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M2_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M2_FL-2_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M2_FL-2_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M2_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M2_FL-3_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M2_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M2_FL-5_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M4_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M4_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M5_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M5_FL-2_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M5_FL-3_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M5_FL-5_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M6_FL-1_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M6_FL-2_RS-3_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-7_FFT-M1_FL-6_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-7_FFT-M5_FL-6_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-7_FFT-M1_FL-6_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-7_FFT-M5_FL-6_RS-2_SM-TRUE')
        self.CaseList = []
        for case in CustomCases:
            case = case.replace('\n','')
            caseSplit = case.split('_')
            algoType = caseSplit[1]
            preprocesstype = caseSplit[2].replace('PR-','')
            fftype = caseSplit[3].replace('FFT-','')
            filtertype = caseSplit[4].replace('FL-','')
            resulttype = caseSplit[5].replace('RS-','')
            isSmooth = caseSplit[6].replace('SM-','')
            fileName = algoType + "_PR-" + str(preprocesstype) + "_FFT-" + str(
                fftype) + "_FL-" + str(filtertype) \
                       + "_RS-" + str(resulttype) + "_SM-" + str(isSmooth)
            self.CaseList.append(fileName)


    """
     GenerateResultsfromParticipants:
     """
    def GenerateResultsfromParticipants(self, ParticipantsOriginalDATA,typeProcessing, UpSampleData):
        # self.CustomCaseList()#
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
                    IsGenerated =  False#self.CheckIfGenerated(case)
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
                        isSmooth=isSmooth.lower()
                        isSmooth=isSmooth.capitalize()
                        if (isSmooth == 'True'):
                            isSmooth = True
                        else:
                            isSmooth = False

                        savePath = self.objConfig.SavePath
                        # if (UpSampleData):
                        #     savePath = self.objConfig.SavePath+ 'UpSampled\\'

                        # Generate Data for all Techniques
                        Process_SingalData(
                            self.objConfig.RunAnalysisForEntireSignalData,
                            objWindowProcessedData.ROIStore, savePath,
                            algoType, fftype,
                            filtertype, resulttype, preprocesstype, isSmooth,
                            objWindowProcessedData.HrGroundTruthList, objWindowProcessedData.SPOGroundTruthList,
                            fileName, self.objConfig.DumpToDisk,participant_number,position,UpSampleData)
                ParticipantsOriginalDATA.pop(participant_number + '_' + position)

    # a
    def LoadandGenerateFaceDatatoBianryFiles(self,SaveFileName,UpSampleData):
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                self.objConfig.setSavePath(participant_number, position, 'RawOriginal')  # set path
                print('Loading and generating FaceData for ' + participant_number + ', ' + position)
                print('Loading from path ' + self.objConfig.DiskPath)
                print('Storing data to path ' + self.objConfig.SavePath)
                # generate only if does not exist
                # if (True):#not self.objFileIO.FileExits(self.objConfig.SavePath + SaveFileName)
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

                self.objFileIO.DumpObjecttoDisk(self.objConfig.SavePath, SaveFileName,
                                                objWindowProcessedData)

                del objWindowProcessedData

    '''
    LoadBinaryData: load data from disk ParticipantsOriginalDATA[ParticipantNumber + Position] -> ROISTORE data
    '''
    def LoadBinaryData(self,SaveFileName):
        ParticipantsOriginalDATA = {}
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                self.objConfig.setSavePath(participant_number, position, 'RawOriginal')  # set path
                print('Loading and generating FaceData for ' + participant_number + ', ' + position)
                print('Loading from path ' + self.objConfig.SavePath)

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

                # Store for procesing locally
                ParticipantsOriginalDATA[participant_number + '_' + position] = objWindowProcessedData

        return ParticipantsOriginalDATA

    def mainMethod(self):
        # For generating meaned channel arrays of image and other required data in form of a objectProcessedData (See class WindowProcessedData)
        # Reqruied to run only once, binary data
        UpSampleData = True  # UpsampleDATA?
        generateBinaryData = False
        if(UpSampleData):
            SaveFileName = 'UnCompressedBinaryLoadedDataUpSampled'
        else:
            SaveFileName = 'UnCompressedBinaryLoadedData2'  # 'UnCompressedBinaryLoadedData' <-- Orignial

        if (generateBinaryData):  # RUN only once
            self.LoadandGenerateFaceDatatoBianryFiles(SaveFileName,UpSampleData)
        else:
            # Load Data from path
            # ParticipantsOriginalDATA = self.LoadBinaryData()
            #Process for entire signal or in windows
            FolderNameforSave = 'ProcessedDataWindows'
            if(self.objConfig.RunAnalysisForEntireSignalData):
                FolderNameforSave= 'ProcessedData'#'ProcessedData'byProcessType

            print(FolderNameforSave)
            #  Load Data from path and Process Data
            self.GenerateResultsfromParticipants(self.LoadBinaryData(SaveFileName),FolderNameforSave,UpSampleData)#FOR Window processing


# skinGroup = ['Europe_WhiteSkin_Group','OtherAsian_OtherSkin_Group','SouthAsian_BrownSkin_Group']
# for item in skinGroup:
skintype = 'Europe_WhiteSkin_Group'##OtherAsian_OtherSkin_Group,SouthAsian_BrownSkin_Group,Europe_WhiteSkin_Group
print('Program started for ' +skintype)
objMain = Main(skintype)  # Add none here to process all skin types [Europe_WhiteSkin_Group,SouthAsian_BrownSkin_Group,OtherAsian_OtherSkin_Group]
objMain.mainMethod()  # Send true to generate binary object data holding images in arrays meaned
del objMain
print('Program Ended')
