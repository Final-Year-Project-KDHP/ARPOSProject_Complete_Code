import os

from matplotlib import pyplot as plt

"""
Configuration:
Global and configuration parameters defined in this class
"""
class Configurations:
    # change path here for uncompressed dataset
    def __init__(self):
        self.DiskPath = 'E:\\ARPOS_Server_Data\\Server_Study_Data\\Europe_WhiteSkin_Group\\'
        self.UnCompressed_dataPath = self.DiskPath + 'UnCompressed\\'
    #Global parameters
    DiskPath = ""
    SavePath = ""
    UnCompressed_dataPath = ""

    #Algorithm List
    AlgoList = ["FastICA", "PCA", "ICAPCA", "None", "Jade"]

    #FFT method types
    fftTypeList = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]

    # with butter filter try ##,3,4,5,6,7,8,9,10 rest are all same result
    # Old with butter filter methods generate same values (polynomial order does not make a differnce for end result
    filtertypeList = [1, 2, 3, 4, 5, 6, 7]

    #Pre processing techniques
    preprocesses = [1, 2, 3, 4, 5, 6, 7, 8]

    #Generating result methods (peak identification and frequency value identification) and getting bpm
    resulttypeList = [1, 2, 3, 4]

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

    # Processed_participants_data = {}

    # heart rate status example resting state and after small workout "Resting1","Resting2","AfterExcersize"
    hearratestatus = ["Resting1","Resting2","AfterExcersize"]

    #Generate HTML Summary #TODO : ADD detail
    GenerateSummary = False

    #Ignore gray when processing signals (only process r,g,b and ir)
    ignoregray = False

    #Generate graphs when processing signals (only process r,g,b and ir)
    GenerateGraphs = False

    #Run for window or for entire signal
    RunAnalysisForEntireSignalData = True

    # setup highpass filter
    ignore_freq_below_bpm = 40
    ignore_freq_below = ignore_freq_below_bpm / 60

    # setup low pass filter
    ignore_freq_above_bpm = 200
    ignore_freq_above = ignore_freq_above_bpm / 60

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
        self.SavePath = self.DiskPath + '\\Result\\' + participantNumber + '\\' + position + '\\'
        #Create save path if it does not exists
        if not os.path.exists(self.SavePath):
            os.makedirs(self.SavePath)

        if(self.GenerateGraphs):
            graphPath = self.SavePath + "Graphs\\"
            if not os.path.exists(graphPath):
                os.makedirs(graphPath)

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
