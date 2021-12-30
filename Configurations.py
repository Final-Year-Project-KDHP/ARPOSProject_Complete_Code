import os

from matplotlib import pyplot as plt

from FileIO import FileIO

"""
Configuration:
Global and configuration parameters defined in this class
"""
class Configurations:
    # change path here for uncompressed dataset
    def __init__(self, isMain=False, skinGroup='None'):
        if(isMain):
            # Get participant Ids
            self.getParticipantNumbers(skinGroup)
            self.setDiskPath(skinGroup)

    #Global parameters
    DiskPath = ""
    SavePath = ""
    UnCompressed_dataPath = ""
    # Skin_Group_Types = ["Europe_WhiteSkin_Group", "OtherAsian_OtherSkin_Group", "SouthAsian_BrownSkin_Group"]
    current_Skin_Group = ""

    #Algorithm List
    AlgoList = ["FastICA", "PCA", "ICAPCA", "None","Jade"]#,

    #FFT method types
    fftTypeList = ["M1",  "M3"]#Where M1=M2 and M3=M7 and "M4", "M5", "M6",-> similar in graph  but test later with other types later

    # with butter filter try ##,3,4,5,6,7,8,9,10 rest are all same result
    # Old with butter filter methods generate same values (polynomial order does not make a differnce for end result
    filtertypeList = [6]#4, --> really bad result, 3, -> no result, FL1 and FL6 are same 1, --> , 2, 5,  7 as the required freq

    #Pre processing techniques
    preprocesses = [1, 3, 4, 5]#2 (same as 3), 6, 7, 8
    ##preprcess 5 does not produce good results for after excersize

    #Generating result methods (peak identification and frequency value identification) and getting bpm
    resulttypeList = [1, 2, 3] #, 4 similar

    #geting heart rate by different methods in bpm
    hrTypeList = [1,2,3]

    #geting signal to noise ratio
    SNRTypeList = [1,2]

    #Smoothen Curve after filtering frequency
    Smoothen = [False, True]

    # Compressed filtered fft result (remove zeros) before caluclating snr
    # to check if this creates better snr or not
    Compressed = [False, True]

    #region of interests, add or reduce here.. (make sure it matches foldername is same as roi region name holding the data)
    roiregions = ["lips", "forehead", "leftcheek", "rightcheek"]

    """
    Participants numbers list
    30 FPS IDS
    "PIS-1118","PIS-2212","PIS-4497","PIS-8308" (one color is 29 frame),"PIS-8343","PIS-3186"
    15 color fps and 30 ir fps ids
    "PIS-3186","PIS-6888"
    Variable with mostly 30 and 29/28
    "PIS-3807","PIS-2169","PIS-9219","PIS-7728","PIS-8308P2","PIS-5868P2","PIS-3252P2","PIS-7381","PIS-6729"
    variable 15fps and 30 fps color and ir
    "PIS-5868", "PIS-3252"
    fully variable
    "PIS-6327","PIS-4709"
    """
    ParticipantNumbers = ["PIS-4497", "PIS-2212"]
    Participantnumbers_SkinGroupTypes = {}

    # Processed_participants_data = {}

    # heart rate status example resting state and after small workout "Resting1","Resting2","AfterExcersize"
    hearratestatus = ["Resting1","Resting2","AfterExcersize"]

    #Generate HTML Summary #TODO : ADD detail
    GenerateSummary = False

    #Ignore gray when processing signals (only process r,g,b and ir)
    ignoregray = False

    #Generate graphs when processing signals (only process r,g,b and ir)
    GenerateGraphs = True

    #Run for window or for entire signal
    RunAnalysisForEntireSignalData = True

    # setup highpass filter
    ignore_freq_below_bpm = 40
    ignore_freq_below = ignore_freq_below_bpm / 60

    # setup low pass filter
    ignore_freq_above_bpm = 200
    ignore_freq_above = ignore_freq_above_bpm / 60

    def setDiskPath(self, current_Skin_Group):
        self.DiskPath = 'E:\\ARPOS_Server_Data\\Server_Study_Data\\' + current_Skin_Group + "\\"
        self.UnCompressed_dataPath = self.DiskPath + 'UnCompressed\\'

    """
    HidePlots:
    Hide plots and save them to disk only
    so it runs in background
    """
    def hidePlots(self):
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})

    """
    GetSavePath:
    Store all the generated graphs and files to this path
    """
    def setSavePath(self,participantNumber,position):
        self.setDiskPath(self.Participantnumbers_SkinGroupTypes.get(participantNumber))
        self.SavePath = self.DiskPath + '\\Result\\' + participantNumber + '\\' + position + '\\'
        #Create save path if it does not exists
        if not os.path.exists(self.SavePath):
            os.makedirs(self.SavePath)

        # if(self.GenerateGraphs):
        #     graphPath = self.SavePath + "Graphs\\"
        #     if not os.path.exists(graphPath):
        #         os.makedirs(graphPath)

    """
    getLoadPath:
    Get all paths where color, ir, and distance of participants is stored 
    Only uncompressed data path,Change path accordingly
    Requires participant number, position (heart rate status), and the region (lips, forehead etc)
    """
    def getLoadPath(self,participantNumber,position,region):
        LoadColordataPath = self.DiskPath + '\\UnCompressed\\' + participantNumber + '\\' + position + 'Cropped\\' + 'Color\\' + region + '\\'  ## Loading path for color data
        LoadIRdataPath = self.DiskPath + '\\UnCompressed\\' + participantNumber + '\\' + position + 'Cropped\\' + 'IR\\' + region + '\\'  ## Loading path for IR data
        LoadDistancePath = self.DiskPath + '\\UnCompressed\\' + participantNumber + '\\' + position + '\\ParticipantInformation.txt'  ## Loading path for depth and other information
        return LoadColordataPath,LoadIRdataPath,LoadDistancePath

    """
    getParticipantNumbers:
    Store all the participant ids to variable [ParticipantNumbers]
    """
    def getParticipantNumbers(self,skinGroup):
        #Read participantid file to get list of participants
        AppDataPath = os.getcwd() + '\\' + 'AppData' + '\\'
        objFile = FileIO()
        participantIds = objFile.ReaddatafromFile(AppDataPath,'ParticipantIds')

        self.ParticipantNumbers = []
        for Line in participantIds:
            Lineid = Line.split(', ')
            if(Lineid[len(Lineid)-1].__contains__('Yes')): #Is participating
                if(Lineid[len(Lineid)-2] != 'UNOCCUPIED'): #Is occupied
                    if(skinGroup == 'None'):
                        piId = Lineid[1] #participantId
                        self.ParticipantNumbers.append(piId)
                        self.Participantnumbers_SkinGroupTypes[piId] = Lineid[len(Lineid)-2]
                    else:
                        if (Lineid[len(Lineid)-2] == skinGroup):
                            piId = Lineid[1]  # participantId
                            self.ParticipantNumbers.append(piId)
                            self.Participantnumbers_SkinGroupTypes[piId] = Lineid[len(Lineid) - 2]
                        else:
                            skip=True
