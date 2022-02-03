import numpy as np

import CommonMethods
from Configurations import Configurations
from FileIO import FileIO
from LoadFaceData import LoadFaceData
from ProcessParticipantsData import Process_SingalData, objFile
from GlobalDataFile import GlobalData, WindowProcessedData
from SQLResults.SQLConfig import SQLConfig

"""
Main Class:
"""


class Main:
    # Store Region of Interest and Results
    ROIStore = {}

    # Global Objects
    objConfig = None
    objSQLConfig= None
    objFileIO = FileIO()
    ParticipantsProcessedHeartRateData = {}
    ParticipantsProcessedBloodOxygenData = {}
    HRNameFileName = "AvgHRdata_"
    SPONameFileName = "AvgSPOdata_"
    CaseList = []

    # Constructor
    def __init__(self, skinGroup='None'):
        self.objConfig = Configurations(skinGroup)
        self.objSQLConfig = SQLConfig()

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
                                    # print("INSERT INTO Techniques(AlgorithmType,PreProcess,FFT,Filter,Result,Smoothen)VALUES('" + algoType + "'," +
                                    #       str(preprocesstype) +",'" + fftype + "'," + str(filtertype) + "," + str(resulttype) + ",'" + str(isSmooth) + "')")

        # CaseListExists= []
        # ExistingCasesDT = self.objSQLConfig.getProcessedCases()
        # for fullRow in ExistingCasesDT.iterrows():
        #     rowData = fullRow[1]
        #     case = rowData.get('CaseProcessed')
        #     participant_number = rowData.get('ParticipantId')
        #     position = rowData.get('HeartRateStatus')
        #     IsUpsampled = rowData.get('UpSampled')
        #     CaseListExists.append(case + '+' + participant_number+'+' +position + str(IsUpsampled))
        #
        # return CaseListExists

    def CheckIfGenerated(self, fileName):
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

        CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M1_FL-3_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-3_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M4_FL-3_RS-1_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-3_FFT-M3_FL-7_RS-1_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-3_FFT-M4_FL-3_RS-1_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-3_FFT-M4_FL-7_RS-1_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-4_FFT-M4_FL-7_RS-1_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-4_FFT-M4_FL-7_RS-1_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M3_FL-7_RS-1_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M4_FL-7_RS-1_SM-True')

        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M4_FL-3_RS-1_SM-True') #-->4497

        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M4_FL-5_RS-1_SM-False') #-->2047,2212,8343
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
        # CustomCases.append('HRSPOwithLog_FastICA3Times_PR-6_FFT-M1_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA3Times_PR-6_FFT-M1_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA3Times_PR-6_FFT-M1_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA3Times_PR-6_FFT-M2_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA3Times_PR-6_FFT-M2_FL-2_RS-2_SM-TRUE')
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

    def CustomLeftOverCases(self):
        self.CaseList = []
        self.CaseList.append('FastICACombined_PR-6_FFT-M4_FL-4_RS-3_SM-True')
        # self.CaseList.append('JadeCombined_PR-7_FFT-M4_FL-7_RS-1_SM-False')
        # self.CaseList.append('PCAICA_PR-7_FFT-M4_FL-7_RS-1_SM-False')
        # self.CaseList.append('JadeCombined_PR-1_FFT-M1_FL-5_RS-3_SM-True')
        # self.CaseList.append('JadeCombined_PR-1_FFT-M2_FL-5_RS-3_SM-True')
        # self.CaseList.append('JadeCombined_PR-5_FFT-M4_FL-6_RS-3_SM-False')
        # self.CaseList.append('FastICA_PR-3_FFT-M2_FL-3_RS-3_SM-False')
        # self.CaseList.append('FastICACombined_PR-3_FFT-M2_FL-3_RS-3_SM-False')
        # self.CaseList.append('PCACombined_PR-3_FFT-M2_FL-3_RS-3_SM-False')
        # self.CaseList.append('PCAICACombined_PR-3_FFT-M2_FL-3_RS-3_SM-False')
        # self.CaseList.append('PCAICA_PR-3_FFT-M2_FL-3_RS-3_SM-False')
        # self.CaseList.append('JadeCombined_PR-3_FFT-M2_FL-3_RS-3_SM-False')
        # self.CaseList.append('FastICA3Times_PR-3_FFT-M2_FL-3_RS-3_SM-False')
        # self.CaseList.append('PCA_PR-3_FFT-M2_FL-3_RS-3_SM-False')
        # self.CaseList.append('None_PR-3_FFT-M2_FL-3_RS-3_SM-False')

    """
     GenerateResultsfromParticipants:
     """
    def GenerateResultsfromParticipants(self, ParticipantsOriginalDATA,typeProcessing, UpSampleData,CaseListExists,NoHRCases,AttemptType):
        # self.CustomCaseList()#
        self.GenerateCases()
        # self.CustomLeftOverCases()
        TotalCasesCount = len(self.CaseList)
        print(str(TotalCasesCount))
        for participant_number in self.objConfig.ParticipantNumbers:  # for each participant
            for position in self.objConfig.hearratestatus:  # for each heart rate status (resting or active)
                print(participant_number + ', ' + position)
                objWindowProcessedData = ParticipantsOriginalDATA.get(participant_number + '_' + position)
                self.objConfig.setSavePath(participant_number, position,typeProcessing)
                currentCasesDone = 0
                IsGenerated = False
                for case in self.CaseList:
                    casefullValue = case + '+' + participant_number+'+' +position + str(UpSampleData)
                    IsGenerated= True if CaseListExists.count(casefullValue) else False
                    if(not IsGenerated):
                        IsGenerated= True if NoHRCases.__contains__(case) else False
                    # if casefullValue in CaseListExists:
                    #     print('Item exists!')
                    #     IsGenerated=True
                    # ResultCaseList = CaseListExists.__getitem__(casefullValue) #  self.CheckIfGenerated(CaseListExists)
                    # IsGenerated = False if len(ResultCaseList) <=0 else True #'1' if UpSampleData == True else '0'
                    if(not IsGenerated):
                        currentCasesDone = currentCasesDone + 1
                        currentPercentage = ((currentCasesDone/TotalCasesCount)*100)
                        # if (IsGenerated):
                        #     continue
                        # else:
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
                            # ListobjParticipantsResultEntireSignalDataRowAll.append(objParticipantsResultEntireSignalDataRow)
                            self.objSQLConfig.SaveRowParticipantsResultsEntireSignal(
                                objParticipantsResultEntireSignalDataRow)
                        else:
                            if(not NoHRCases.__contains__(case)):
                                exists = self.objFileIO.FileExits(self.objConfig.SavePath+ "NOHRCases_" + skintype+ ".txt")
                                mode = "w+"
                                if(exists):
                                    mode = "a"
                                self.objFileIO.WritedatatoFile(self.objConfig.SavePath,"NOHRCases_" + skintype, objParticipantsResultEntireSignalDataRow + "\t" + case,mode)

                ParticipantsOriginalDATA.pop(participant_number + '_' + position)

        # for objParticipantsResultEntireSignalDataRow in ListobjParticipantsResultEntireSignalDataRowAll:
        #     self.objSQLConfig.SaveRowParticipantsResultsEntireSignal(objParticipantsResultEntireSignalDataRow)

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
    def LoadBinaryData(self,SaveFileName,UpSampleData):
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
    def LoadNoHRfILES(self,SaveFileName):

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
    def mainMethod(self,UpSampleData,generateBinaryData,skintype,CaseListExists,combinedRun):
        # For generating meaned channel arrays of image and other required data in form of a objectProcessedData (See class WindowProcessedData)
        # Reqruied to run only once, binary data
        # self.objConfig.ParticipantNumbers = ["PIS-1118"]#,"PIS-2047","PIS-8343"
        # self.objConfig.hearratestatus = ["Resting1"]
        AttemptType =1
        if(UpSampleData):
            SaveFileName ='UnCompressedBinaryLoadedDataUpSampled' #'UnCompressedBinaryLoadedDataUpSampledAttempt2TrimmedTime' #UnCompressedBinaryLoadedDataUpSampled,UnCompressedBinaryLoadedDataSelectiveLowFrameRemovedUpSampled
            if (combinedRun):
                SaveFileName = 'UnCompressedBinaryLoadedDataUpSampledCombinedSameLength'
            # SaveFileName = 'UnCompressedBinaryLoadedDataSelectiveLowFrameRemovedUpSampledAttempt2' #for 4014 only
        else:
            SaveFileName = 'UnCompressedBinaryLoadedData2'#'UnCompressedBinaryLoadedData2Attempt2TrimmedTime'  # 'UnCompressedBinaryLoadedData'<-- Orignial , UnCompressedBinaryLoadedData2, UnCompressedBinaryLoadedDataSelectiveLowFrameRemoved
            if(combinedRun):
                SaveFileName = 'UnCompressedBinaryLoadedData2CombinedSameLength'
            # SaveFileName = 'UnCompressedBinaryLoadedDataSelectiveLowFrameRemovedAttempt2' #for 4014 only

        if(SaveFileName.__contains__('Attempt2')):
            AttemptType=2

            #TODO: USE PREVIOUS NONN REVISED
        if (generateBinaryData):  # RUN only once
            self.LoadandGenerateFaceDatatoBianryFiles(SaveFileName,UpSampleData)
        else:
            print('processing started')
            # Load Data from path
            # ParticipantsOriginalDATA = self.LoadBinaryData()
            #Process for entire signal or in windows
            FolderNameforSave = 'ProcessedDataWindows'
            if(self.objConfig.RunAnalysisForEntireSignalData):
                if (UpSampleData):
                    FolderNameforSave= 'ProcessedDataUpSampled'#'ProcessedData'byProcessType ,ProcessedDataRevised,ProcessedDataUpSampled
                else:
                    FolderNameforSave= 'ProcessedData'#'ProcessedData'byProcessType ,ProcessedDataRevised,ProcessedDataUpSampled


            print(FolderNameforSave)

            NoHRCases = self.LoadNoHRfILES(FolderNameforSave)

            #  Load Data from path and Process Data
            self.GenerateResultsfromParticipants(self.LoadBinaryData(SaveFileName,UpSampleData),FolderNameforSave,UpSampleData,CaseListExists,NoHRCases,AttemptType)#FOR Window processing

UpSampleDataList = [False,True]  # UpsampleDATA?
skinGroup = ['OtherAsian_OtherSkin_Group','SouthAsian_BrownSkin_Group','Europe_WhiteSkin_Group' ]#
CaseListExists = []
combinedRun=True
for skintype in skinGroup:
    loadedGeneratedCaseList = False
    print('Program started for ' +skintype)
    objMain = Main(skintype)  # Add none here to process all skin types [Europe_WhiteSkin_Group,SouthAsian_BrownSkin_Group,OtherAsian_OtherSkin_Group]
    generateBinaryData = False
    if(not loadedGeneratedCaseList):
        ExistingCasesDT = objMain.objSQLConfig.getProcessedCases(skintype,'1')#AttemptType
        for fullRow in ExistingCasesDT.iterrows():
            rowData = fullRow[1]
            case = rowData.get('CaseProcessed')
            participant_number = rowData.get('ParticipantId')
            position = rowData.get('HeartRateStatus')
            IsUpsampled = rowData.get('UpSampled')
            CaseListExists.append(case + '+' + participant_number + '+' + position + str(IsUpsampled))
        loadedGeneratedCaseList = True
    for UpSampleData in UpSampleDataList:
        print('for upsample: ' + str(UpSampleData))
        objMain.mainMethod(UpSampleData,generateBinaryData,skintype,CaseListExists,combinedRun)  # Send true to generate binary object data holding images in arrays meaned
    del objMain
    print('Program Ended')
