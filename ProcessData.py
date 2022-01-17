import pickle
from datetime import datetime
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from sklearn import preprocessing
from Algorithm import AlgorithmCollection
from HeartRateAndSPO.ComputeHeartRate import ComputerHeartRate
from SaveGraphs import Plots
import sys
from WindowData import Window_Data, LogItems


class ProcessFaceData:
    # Hold Current Window Region Data
    regionStore = []
    # regionSignalData = []
    regionBlueData = []
    regionGreenData = []
    regionRedData = []
    regionGreyData = []
    regionIRData = []
    distanceM = []
    ColorfpswithTime = []
    IRfpswithTime = []
    time_list_color = []
    timecolorCount = []
    time_list_ir = []
    timeirCount = []
    Frametime_list_ir = []
    Frametime_list_color = []
    region = ''

    # For window size
    Window_count = 0
    # regionWindowSignalData = []
    regionWindowBlueData = []
    regionWindowGreenData = []
    regionWindowRedData = []
    regionWindowGreyData = []
    regionWindowIRData = []
    WindowdistanceM = []
    Windowtime_list_color = []
    WindowtimecolorCount = []
    Windowtime_list_ir = []
    WindowtimeirCount = []
    WindowFrametime_list_ir = []
    WindowFrametime_list_color = []
    WindowColorfpswithTime = []
    WindowIRfpswithTime = []

    timeinSeconds = 0
    Colorfrequency = []
    IRfrequency = []

    # Constants
    ColorEstimatedFPS = 0
    IREstimatedFPS = 0
    TotalWindows = 0
    timeinSecondsWindow = 0
    components = 5
    IRIndex = 4
    grayIndex = 3
    ramp_end_bpm = 55
    ramp_start_percentage = 0.5
    ramp_end_percentage = 1
    ramp_end_hz = 0
    freq_bpmColor = []
    freq_bpmIR = []
    ignore_freq_index_below = 0
    ignore_freq_index_above = 0
    ramp_start = 0
    ramp_end = 0
    rampDesignLength = 0
    ramp_design = []
    ramplooprange = 0

    # setup highpass filter
    ignore_freq_below_bpm = 40
    ignore_freq_below = 0

    # setup low pass filter
    ignore_freq_above_bpm = 200
    ignore_freq_above = 0

    # Input Parameters
    Algorithm_type = ''
    FFT_type = ''
    DumpToDisk = ''
    Filter_type = 0
    Result_type = 0
    HrType = 0
    isCompressed = False
    Preprocess_type = 0
    SavePath = ''
    ignoreGray = False
    isSmoothen = False
    GenerateGraphs = False
    snrType = 0

    # ResultData
    IrSnr = 0.0
    GreySnr = 0.0
    RedSnr = 0.0
    GreenSnr = 0.0
    BlueSnr = 0.0
    IrBpm = 0.0
    GreyBpm = 0.0
    RedBpm = 0.0
    GreenBpm = 0.0
    BlueBpm = 0.0
    IrFreqencySamplingError = 0.0
    GreyFreqencySamplingError = 0.0
    RedFreqencySamplingError = 0.0
    GreenFreqencySamplingError = 0.0
    BlueFreqencySamplingError = 0.0
    bestHeartRateSnr = 0.0
    beatsPerSecond = 0.0
    bestBpm = 0.0
    channeltype = ''
    heartRatePeriod = 0.0
    SNRSummary = ''
    fileName = ''

    # USE ondly for entiere signal
    WindowProcessStartTime = None
    WindowProcessEndTime = None
    WindowProcessDifferenceTime = None
    #SPO
    SPOWindowProcessStartTime = None
    SPOWindowProcessEndTime = None
    FullLog = ''
    SPOWindowProcessDifferenceTime = None
    SPOstd= None
    SPOerr= None
    SPOoxylevl= None

    # Objects
    objPlots = Plots()
    objAlgorithm = AlgorithmCollection()

    # constructor
    def __init__(self, Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type, SavePath, ignoreGray,
                 isSmoothen, GenerateGraphs, timeinSeconds, DumpToDisk, fileName):
        self.fileName = fileName
        self.Algorithm_type = Algorithm_type
        self.FFT_type = FFT_type
        self.DumpToDisk = DumpToDisk
        self.Filter_type = Filter_type
        self.Result_type = Result_type
        self.Preprocess_type = Preprocess_type
        self.SavePath = SavePath
        self.ignoreGray = ignoreGray
        self.isSmoothen = isSmoothen
        self.GenerateGraphs = GenerateGraphs
        self.timeinSeconds = timeinSeconds
        # self.snrType = snrType

        # setup highpass filter
        self.ignore_freq_below_bpm = 40
        self.ignore_freq_below = self.ignore_freq_below_bpm / 60
        # setup low pass filter
        self.ignore_freq_above_bpm = 200
        self.ignore_freq_above = self.ignore_freq_above_bpm / 60
        self.ramp_end_hz = self.ramp_end_bpm / 60

        if (self.ignoreGray):
            self.components = 4
            self.IRIndex = self.components - 2

    def SetFromDataParameters(self, region, IRfpswithTime, ColorfpswithTime, TotalWindows, timeinSeconds,
                              WindowIRfpswithTime, WindowColorfpswithTime, WindowCount, ColorEstimatedFPS,
                              IREstimatedFPS, regionStore, blue, green, red, grey, Irchannel, distanceM, regionBlueData,
                              regionGreenData,
                              regionRedData, regionGreyData, regionIRData,timecolorCount,timeirCount,Frametime_list_ir,Frametime_list_color,Colorfrequency,IRfrequency):
        self.Colorfrequency = Colorfrequency
        self.IRfrequency = IRfrequency
        self.region = region
        self.IRfpswithTime = IRfpswithTime
        self.ColorfpswithTime = ColorfpswithTime
        self.TotalWindows = TotalWindows
        self.timeinSecondsWindow = timeinSeconds
        self.WindowIRfpswithTime = WindowIRfpswithTime
        self.WindowColorfpswithTime = WindowColorfpswithTime
        # self.regionStore = regionStore
        self.regionBlueData = blue
        self.regionGreenData = green
        self.regionRedData = red
        self.regionGreyData = grey
        self.regionIRData = Irchannel
        self.distanceM = distanceM
        # self.time_list_color = self.regionStore.time_list_color
        self.timecolorCount = timecolorCount
        # self.time_list_ir = self.regionStore.time_list_ir
        self.timeirCount = timeirCount
        self.Frametime_list_ir = Frametime_list_ir
        self.Frametime_list_color = Frametime_list_color

        # all to window size
        self.Window_count = WindowCount
        # self.regionWindowSignalData = self.regionSignalData
        self.regionWindowBlueData = regionBlueData
        self.regionWindowGreenData = regionGreenData
        self.regionWindowRedData = regionRedData
        self.regionWindowGreyData = regionGreyData
        self.regionWindowIRData = regionIRData
        self.WindowdistanceM = self.distanceM
        # self.Windowtime_list_color = self.time_list_color
        self.WindowtimecolorCount = self.timecolorCount
        # self.Windowtime_list_ir = self.time_list_ir
        self.WindowtimeirCount = self.timeirCount
        self.WindowFrametime_list_ir = self.Frametime_list_ir
        self.WindowFrametime_list_color = self.Frametime_list_color
        # self.ColorfpswithTime=self.ColorfpswithTime
        # self.IRfpswithTime=self.IRfpswithTime

        self.ColorEstimatedFPS = ColorEstimatedFPS
        self.IREstimatedFPS = IREstimatedFPS
        self.objAlgorithm.ColorEstimatedFPS = self.ColorEstimatedFPS
        self.objAlgorithm.IREstimatedFPS = self.IREstimatedFPS
        self.objPlots.ColorEstimatedFPS = self.ColorEstimatedFPS
        self.objPlots.IREstimatedFPS = self.IREstimatedFPS

    # region Signal Source Data to respective data arrays methods
    '''
    setSignalSourceData: 
    A method that takes ROIStore data and region name such as lips, forehead etc
    and stores specific region data to local object to process it
    '''

    def setSignalSourceData(self, ROIStore, region):
        # Split ROI Store region data
        # where region can be lips, forehead etc
        self.regionStore = ROIStore.get(region)
        # self.regionSignalData = self.regionStore.getAllData()
        self.regionBlueData = self.regionStore.blue
        self.regionGreenData = self.regionStore.green
        self.regionRedData = self.regionStore.red
        self.regionGreyData = self.regionStore.grey
        self.regionIRData = self.regionStore.Irchannel
        self.distanceM = self.regionStore.distanceM
        # self.time_list_color = self.regionStore.time_list_color
        self.timecolorCount = self.regionStore.timecolorCount
        # self.time_list_ir = self.regionStore.time_list_ir
        self.timeirCount = self.regionStore.timeirCount
        self.Frametime_list_ir = self.regionStore.Frametime_list_ir
        self.Frametime_list_color = self.regionStore.Frametime_list_color

    '''
    getSingalDataWindow: 
    This method records original data by calling previously defined method 'setSignalSourceData(ROIStore, region)'
    and splits data to window size to process it
    '''

    def getSingalDataWindow(self, ROIStore, region, WindowCount, TotalWindows, timeinSeconds):

        self.IRfpswithTime = ROIStore.get(region).IRfpswithTime
        self.ColorfpswithTime = ROIStore.get(region).ColorfpswithTime
        self.region = region
        self.TotalWindows = TotalWindows
        self.timeinSecondsWindow = timeinSeconds
        ###CALCULATE NEW FPS
        # calculate window slider as per fps?
        # stepinSecond = 1
        # WindowtimeinSeconds = 10
        # WindowSliderinFrame = self.getWindowSliderinFrame()
        self.WindowIRfpswithTime, WindowSliderIR, stepIR = self.reCalculateFPS(self.IRfpswithTime, WindowCount,
                                                                               self.timeinSecondsWindow)  ##CHECK on how to do this for each window size
        self.WindowColorfpswithTime, WindowSliderColor, stepColor = self.reCalculateFPS(self.ColorfpswithTime,
                                                                                        WindowCount,
                                                                                        self.timeinSecondsWindow)
        # Split ROI Store region data
        if (WindowCount == 0):
            self.setSignalSourceData(ROIStore, region)
        else:
            OrignialBlue = self.regionBlueData
            OrignialGreen = self.regionGreenData
            OrignialRed = self.regionRedData
            OrignialGrey = self.regionGreyData
            OrignialIr = self.regionIRData

            OrignialBlue = OrignialBlue[stepColor:]  # from steps till end and disacard rest
            OrignialGreen = OrignialGreen[stepColor:]
            OrignialRed = OrignialRed[stepColor:]
            OrignialGrey = OrignialGrey[stepColor:]
            OrignialIr = OrignialIr[stepIR:]
            self.regionBlueData = OrignialBlue  # np.c_[OrignialBlue, OrignialGreen, OrignialRed, OrignialGrey, OrignialIr]
            self.regionGreenData = OrignialGreen
            self.regionRedData = OrignialRed
            self.regionGreyData = OrignialGrey
            self.regionIRData = OrignialIr
            self.distanceM = self.distanceM[stepIR:]
            # self.time_list_color = self.time_list_color[step:]
            self.timecolorCount = self.timecolorCount[stepColor:]
            # self.time_list_ir = self.time_list_ir[step:]
            self.timeirCount = self.timeirCount[stepColor:]
            self.Frametime_list_ir = self.Frametime_list_ir[stepIR:]
            self.Frametime_list_color = self.Frametime_list_color[stepColor:]

        # split to window size
        self.Window_count = WindowCount
        # self.regionWindowSignalData = self.regionSignalData[:WindowSlider]
        self.regionWindowBlueData = self.regionBlueData[:WindowSliderColor]
        self.regionWindowGreenData = self.regionGreenData[:WindowSliderColor]
        self.regionWindowRedData = self.regionRedData[:WindowSliderColor]
        self.regionWindowGreyData = self.regionGreyData[:WindowSliderColor]
        self.regionWindowIRData = self.regionIRData[:WindowSliderIR]
        self.WindowdistanceM = self.distanceM[:WindowSliderIR]
        # self.Windowtime_list_color = self.time_list_color[:WindowSlider]
        self.WindowtimecolorCount = self.timecolorCount[:WindowSliderColor]
        # self.Windowtime_list_ir = self.time_list_ir[:WindowSlider]
        self.WindowtimeirCount = self.timeirCount[:WindowSliderIR]
        self.WindowFrametime_list_ir = self.Frametime_list_ir[:WindowSliderIR]
        self.WindowFrametime_list_color = self.Frametime_list_color[:WindowSliderColor]
        # self.ColorfpswithTime=self.ColorfpswithTime[:WindowSlider]
        # self.IRfpswithTime=self.IRfpswithTime[:WindowSlider]

        self.ColorEstimatedFPS = self.getDuplicateValue(self.WindowColorfpswithTime)  # Only one time
        self.IREstimatedFPS = self.getDuplicateValue(self.WindowIRfpswithTime)  # Only one time
        # set estimated fps
        # self.ColorEstimatedFPS = ROIStore.get(region).ColorEstimatedFPS  # IREstimatedFPS
        # self.IREstimatedFPS = ROIStore.get(region).IREstimatedFPS
        # Set in algorithm class
        self.objAlgorithm.ColorEstimatedFPS = self.ColorEstimatedFPS
        self.objAlgorithm.IREstimatedFPS = self.IREstimatedFPS
        # Set in plot class
        self.objPlots.ColorEstimatedFPS = self.ColorEstimatedFPS
        self.objPlots.IREstimatedFPS = self.IREstimatedFPS

    '''
    getSingalData: 
    This method records original data without spliting it into widow size (so entire signal data)
    '''

    def getSingalData(self, ROIStore, region, WindowCount, TotalWindows, timeinSeconds):
        self.IRfpswithTime = ROIStore.get(region).IRfpswithTime
        self.ColorfpswithTime = ROIStore.get(region).ColorfpswithTime
        self.region = region
        self.TotalWindows = TotalWindows
        self.timeinSecondsWindow = timeinSeconds
        ###CALCULATE NEW FPS
        # calculate window slider as per fps?
        # stepinSecond = 1
        # WindowtimeinSeconds = 10
        # WindowSliderinFrame = self.getWindowSliderinFrame()
        self.WindowIRfpswithTime, WindowSliderIR, stepIR = self.reCalculateFPS(self.IRfpswithTime, WindowCount,
                                                                               self.timeinSecondsWindow)  ##CHECK on how to do this for each window size
        self.WindowColorfpswithTime, WindowSliderColor, stepColor = self.reCalculateFPS(self.ColorfpswithTime,
                                                                                        WindowCount,
                                                                                        self.timeinSecondsWindow)
        #  ROI Store region data
        self.setSignalSourceData(ROIStore, region)

        # all to window size
        self.Window_count = WindowCount
        # self.regionWindowSignalData = self.regionSignalData
        self.regionWindowBlueData = self.regionBlueData
        self.regionWindowGreenData = self.regionGreenData
        self.regionWindowRedData = self.regionRedData
        self.regionWindowGreyData = self.regionGreyData
        self.regionWindowIRData = self.regionIRData
        self.WindowdistanceM = self.distanceM
        # self.Windowtime_list_color = self.time_list_color
        self.WindowtimecolorCount = self.timecolorCount
        # self.Windowtime_list_ir = self.time_list_ir
        self.WindowtimeirCount = self.timeirCount
        self.WindowFrametime_list_ir = self.Frametime_list_ir
        self.WindowFrametime_list_color = self.Frametime_list_color
        # self.ColorfpswithTime=self.ColorfpswithTime
        # self.IRfpswithTime=self.IRfpswithTime

        self.ColorEstimatedFPS =  self.getDuplicateValue(self.WindowColorfpswithTime)  # Only one time
        self.IREstimatedFPS = self.getDuplicateValue(self.WindowIRfpswithTime)  # Only one time
        # set estimated fps
        # self.ColorEstimatedFPS = ROIStore.get(region).ColorEstimatedFPS  # IREstimatedFPS
        # self.IREstimatedFPS = ROIStore.get(region).IREstimatedFPS
        # Set in algorithm class
        self.objAlgorithm.ColorEstimatedFPS = self.ColorEstimatedFPS
        self.objAlgorithm.IREstimatedFPS = self.IREstimatedFPS
        # Set in plot class
        self.objPlots.ColorEstimatedFPS = self.ColorEstimatedFPS
        self.objPlots.IREstimatedFPS = self.IREstimatedFPS

    # endregion

    def getDuplicateValue(self, ini_dict):
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

    def reCalculateFPS(self, dataDictionary, currentWindow, WindowLength):
        initialindex = currentWindow + 1
        fpsList = {}
        endIndex = WindowLength + currentWindow  # -1#initialindex + WindowLength
        count = 1
        WindowSlider = 0
        step = 0
        for key, val in dataDictionary.items():
            # if(initialindex == 1 and step == 0):#if its first window
            #     step = val
            if (count == initialindex):
                fpsList[key] = val
                WindowSlider = WindowSlider + val
                if (step == 0):
                    step = val
            else:
                if (count <= endIndex):
                    if (len(fpsList) > 0):
                        fpsList[key] = val
                        WindowSlider = WindowSlider + val
                else:
                    break
            count = count + 1

        return fpsList, WindowSlider, step

    # region Graph path and name methods
    '''
    getGraphPath: Returns graph path where images of graphs can be strored and viewed
    '''

    def getGraphPath(self):
        return self.SavePath  # + "Graphs\\"

    '''
    defineGraphName: Returns image name of a graph, adds various types of techniques and filters for identification purpose
    '''

    def defineGraphName(self, graphName):
        imageName = graphName + "_" + self.region + "_W" + str(
            self.Window_count) + "_" + self.Algorithm_type + "_FFT-" + str(self.FFT_type) + "_FL-" + str(
            self.Filter_type) + "_RS-" + str(self.Result_type) + "_HR-" + str(self.HrType) + "_PR-" + str(
            self.Preprocess_type) + "_SM-" + str(
            self.isSmoothen) + "_CP-" + str(self.isCompressed)
        return imageName

    '''
    GenerateGrapth: Plot graph and save to disk for signal data
    '''

    def GenerateGrapth(self, graphName, B_Signal, G_Signal, R_Signal, Gy_Signal, IR_Signal):
        # SingalData
        processedBlue = B_Signal
        processedGreen = G_Signal
        processedRed = R_Signal
        if (not self.ignoreGray):
            processedGrey = Gy_Signal  # [:, self.grayIndex]
        else:
            processedGrey = []
        processedIR = IR_Signal  # [:, self.IRIndex]

        if (graphName == "RawData"):
            imageName = self.defineGraphName("1-" + graphName)
            self.objPlots.plotGraphAllWithoutTimewithParam(self.getGraphPath(), imageName,
                                                           processedBlue, processedGreen, processedRed, processedGrey,
                                                           processedIR,
                                                           "No of Frames", "Intensity")
            # imageName = self.defineGraphName(graphName + "All")
            # self.objPlots.plotAllinOneWithoutTime(self.getGraphPath(), imageName, processedBlue, processedGreen, processedRed,
            #                                       processedGrey, processedIR, "No of Frames", "Intensity")
        elif (graphName == "PreProcessed"):
            imageName = self.defineGraphName("2-" + graphName + "All")
            self.objPlots.plotAllinOne(self.getGraphPath(),
                                       processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                                       imageName, self.timeinSeconds, "Time(s)", "Amplitude")
            imageName = self.defineGraphName("2-" + graphName + "TimeAll")
            self.objPlots.plotGraphAllwithParam(self.getGraphPath(), imageName, self.time_list_color, self.time_list_ir,
                                                processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                                                "Time(s)", "Amplitude")
        elif (graphName == "Algorithm"):

            imageName = self.defineGraphName("3-" + graphName + "Time")
            self.objPlots.plotGraphAllwithParam(self.getGraphPath(), imageName, self.time_list_color,
                                                self.time_list_ir,
                                                processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                                                "Time(s)",
                                                "Amplitude")
            # imageName = self.defineGraphName(graphName + "TimeAll")
            # self.objPlots.plotAllinOne(self.getGraphPath(),
            #                            processedBlue, processedGreen, processedRed, processedGrey, processedIR,
            #                            imageName,
            #                            self.EstimatedFPS,
            #                            self.timeinSeconds, "Time(s)",
            #                            "Amplitude")
        elif (graphName == "Smooth"):
            # imageName = self.defineGraphName("4-" + graphName + "TimeAll")
            # self.objPlots.plotAllinOne(self.getGraphPath(),
            #                            processedBlue, processedGreen, processedRed, processedGrey, processedIR,
            #                            imageName, self.EstimatedFPS,
            #                            self.timeinSeconds,
            #                            "Time(s)",
            #                            "Amplitude")

            imageName = self.defineGraphName("4-" + graphName + "Time")
            self.objPlots.plotGraphAllwithParam(self.getGraphPath(), imageName, self.time_list_color,
                                                self.time_list_ir,
                                                processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                                                "Time(s)",
                                                "Amplitude")
        elif (graphName == "FFT"):
            imageName = self.defineGraphName("5-" + graphName)
            self.objPlots.PlotFFT(processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                                  self.Colorfrequency, self.IRfrequency, imageName, self.getGraphPath(),
                                  imageName)
        elif (graphName == "Filtered"):
            imageName = self.defineGraphName("6-" + graphName)
            self.objPlots.PlotFFT(processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                                  self.Colorfrequency, self.IRfrequency,
                                  imageName, self.getGraphPath(), imageName)

    # endregion

    # region pre process signal data
    def preprocessdataType3(self, bufferArray, FPS):
        """remove NaN and Inf values"""
        output = bufferArray[(np.isnan(bufferArray) == 0) & (np.isinf(bufferArray) == 0)]
        # UpSample
        # L = len(output) #forplloting
        # max_time = L / FPS #forplloting
        # t = np.linspace(0, max_time, L) forplloting
        output = signal.resample(output, len(output) * 2)
        # t2 = np.linspace(0, max_time, len(output))  #forplloting# time_steps

        detrended_data = signal.detrend(output)

        try:
            '''interpolation data buffer to make the signal become more periodic (advoid spectral leakage) '''
            L = len(detrended_data)
            max_time = L / FPS
            timeCount = np.linspace(0, max_time, L)  # time_steps
            even_times = np.linspace(timeCount[0], timeCount[-1],
                                     len(timeCount))
            interp = np.interp(even_times, timeCount, detrended_data)
            interpolated_data = np.hamming(L) * interp
            # plt.plot(timeCount, detrended_data, 'blue')
            # plt.plot(timeCount, interpolated_data, 'red')
            # plt.show()
        except:
            interpolated_data = detrended_data

        '''removes noise'''
        # N = 3
        # """ x == an array of data. N == number of samples per average """
        # cumsum = np.cumsum(np.insert(interpolated_data, [0, 0, 0], 0))
        # rm = (cumsum[N:] - cumsum[:-N]) / float(N)

        '''normalize the input data buffer '''
        normalized_data = interpolated_data / np.linalg.norm(interpolated_data)
        return normalized_data

    def preprocessdataType1(self, bufferArray, isDetrend, FPS):
        """remove NaN and Inf values"""
        output = bufferArray[(np.isnan(bufferArray) == 0) & (np.isinf(bufferArray) == 0)]
        detrended_data = output
        if (isDetrend):
            detrended_data = signal.detrend(output)

        try:
            '''interpolation data buffer to make the signal become more periodic (advoid spectral leakage) '''
            L = len(detrended_data)
            max_time = L / FPS
            timeCount = np.linspace(0, max_time, L)  # time_steps
            even_times = np.linspace(timeCount[0], timeCount[-1],
                                     len(timeCount))
            interp = np.interp(even_times, timeCount, detrended_data)
            interpolated_data = np.hamming(L) * interp
        except:
            interpolated_data = detrended_data

        '''removes noise'''
        # N = 3
        # """ x == an array of data. N == number of samples per average """
        # cumsum = np.cumsum(np.insert(interpolated_data, [0, 0, 0], 0))
        # rm = (cumsum[N:] - cumsum[:-N]) / float(N)

        '''normalize the input data buffer '''
        normalized_data = interpolated_data / np.linalg.norm(interpolated_data)
        return normalized_data

    def preprocessdataType2(self, bufferArray, isDetrend, FPS):
        """remove NaN and Inf values"""
        output = bufferArray[(np.isnan(bufferArray) == 0) & (np.isinf(bufferArray) == 0)]
        detrended_data = output
        if (isDetrend):
            detrended_data = signal.detrend(output)
        try:
            '''interpolation data buffer to make the signal become more periodic (advoid spectral leakage) '''
            L = len(detrended_data)
            max_time = L / FPS
            timeCount = np.linspace(0, max_time, L)  # time_steps
            even_times = np.linspace(timeCount[0], timeCount[-1],
                                     len(timeCount))
            interp = np.interp(even_times, timeCount, detrended_data)
            interpolated_data = np.hamming(L) * interp

        except:
            interpolated_data = detrended_data

        '''removes noise'''
        smoothed_data = signal.medfilt(interpolated_data, 15)

        '''normalize the input data buffer '''
        normalized_data_med = smoothed_data / np.linalg.norm(smoothed_data)
        return normalized_data_med

    '''
    preprocessSignalData: Preprocess techniques to apply on signal data
    '''

    def preprocessSignalData(self, blue, green, red, grey, Irchannel):
        # Processed channel data
        processedBlue = blue
        processedGreen = green
        processedRed = red
        processedGrey = grey
        processedIR = Irchannel

        if (self.Preprocess_type == 7):
            if (len(processedGrey) > 0):
                processedBlue = self.preprocessdataType3(np.array(processedBlue), self.ColorEstimatedFPS)
                processedGreen = self.preprocessdataType3(np.array(processedGreen), self.ColorEstimatedFPS)
                processedRed = self.preprocessdataType3(np.array(processedRed), self.ColorEstimatedFPS)
                processedGrey = self.preprocessdataType3(np.array(processedGrey), self.ColorEstimatedFPS)
            if (len(processedIR) > 0):
                processedIR = self.preprocessdataType3(np.array(processedIR), self.IREstimatedFPS)

        elif (self.Preprocess_type == 6):
            if (len(processedGrey) > 0):
                processedBlue = self.preprocessdataType1(np.array(processedBlue), True, self.ColorEstimatedFPS)
                processedGreen = self.preprocessdataType1(np.array(processedGreen), True, self.ColorEstimatedFPS)
                processedRed = self.preprocessdataType1(np.array(processedRed), True, self.ColorEstimatedFPS)
                processedGrey = self.preprocessdataType1(np.array(processedGrey), True, self.ColorEstimatedFPS)
            if (len(processedIR) > 0):
                processedIR = self.preprocessdataType1(np.array(processedIR), True, self.IREstimatedFPS)

        elif (self.Preprocess_type == 5):
            if (len(processedGrey) > 0):
                processedBlue = self.preprocessdataType1(np.array(processedBlue), False, self.ColorEstimatedFPS)
                processedGreen = self.preprocessdataType1(np.array(processedGreen), False, self.ColorEstimatedFPS)
                processedRed = self.preprocessdataType1(np.array(processedRed), False, self.ColorEstimatedFPS)
                processedGrey = self.preprocessdataType1(np.array(processedGrey), False, self.ColorEstimatedFPS)
            if (len(processedIR) > 0):
                processedIR = self.preprocessdataType1(np.array(processedIR), False, self.IREstimatedFPS)

        elif (self.Preprocess_type == 4):
            if (len(processedGrey) > 0):
                processedBlue = self.preprocessdataType2(np.array(processedBlue), True, self.ColorEstimatedFPS)
                processedGreen = self.preprocessdataType2(np.array(processedGreen), True, self.ColorEstimatedFPS)
                processedRed = self.preprocessdataType2(np.array(processedRed), True, self.ColorEstimatedFPS)
                processedGrey = self.preprocessdataType2(np.array(processedGrey), True, self.ColorEstimatedFPS)
            if (len(processedIR) > 0):
                processedIR = self.preprocessdataType2(np.array(processedIR), True, self.IREstimatedFPS)

        elif (self.Preprocess_type == 3):
            if (len(processedGrey) > 0):
                processedBlue = self.preprocessdataType2(np.array(processedBlue), False, self.ColorEstimatedFPS)
                processedGreen = self.preprocessdataType2(np.array(processedGreen), False, self.ColorEstimatedFPS)
                processedRed = self.preprocessdataType2(np.array(processedRed), False, self.ColorEstimatedFPS)
                processedGrey = self.preprocessdataType2(np.array(processedGrey), False, self.ColorEstimatedFPS)
            if (len(processedIR) > 0):
                processedIR = self.preprocessdataType2(np.array(processedIR), False, self.IREstimatedFPS)

        elif (self.Preprocess_type == 2):
            processedBlue = processedBlue / np.linalg.norm(processedBlue)
            processedGreen = processedGreen / np.linalg.norm(processedGreen)
            processedRed = processedRed / np.linalg.norm(processedRed)
            processedGrey = processedGrey / np.linalg.norm(processedGrey)
            processedIR = processedIR / np.linalg.norm(processedIR)
            # SCombined = np.c_[processedBlue, processedGreen, processedRed, processedGrey]
            # if (len(processedIR) == len(processedGrey)):
            #     SCombined = self.getSignalDataCombined(processedBlue, processedGreen, processedRed, processedGrey,
            #                                            processedIR)
            #
            # # SCombined = preprocessing.normalize(SCombined)
            # SCombined = preprocessing.normalize(SCombined)
            #
            # # split
            # if (len(processedGrey) > 0):
            #     processedBlue = SCombined[:, 0]
            # processedGreen = SCombined[:, 1]
            # processedRed = SCombined[:, 2]
            # processedGrey = SCombined[:, 3]
            # if (len(processedIR) == len(processedGrey)):
            #     if (len(processedIR) > 0):
            #         processedIR = SCombined[:, 4]
            # else:
            #     # processedIR = np.array(processedIR).reshape((len(processedIR), 1))
            #     # processedIR = preprocessing.normalize(processedIR)
            #     # processedIR = np.array(processedIR).reshape(1, (len(processedIR)))[0]
        else:  # 1
            processedBlue = blue
            processedGreen = green
            processedRed = red
            processedGrey = grey
            processedIR = Irchannel

        # generate PreProcessed plot
        if (self.GenerateGraphs):
            self.GenerateGrapth("PreProcessed", processedBlue, processedGreen, processedRed, processedGrey, processedIR)

        # wRITE IN THESE ignoring gray was not big differnecein results, but better with gray if gray peak was high
        return processedBlue, processedGreen, processedRed, processedGrey, processedIR

    # endregion

    '''
    getSignalDataCombined: Combine r,g,b,gy,ir to one array
    '''

    def getSignalDataCombined(self, blue, green, red, grey, Irchannel):
        S = []

        if (self.ignoreGray):
            S = np.c_[blue, green, red, Irchannel]
        else:
            S = np.c_[blue, green, red, grey, Irchannel]

        return S

    def ReshapeArray(self, AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR):
        # Change Shape
        # if (len(AlgoprocessedGrey) > 0):
        AlgoprocessedBlue = np.array(AlgoprocessedBlue).reshape(1, (len(AlgoprocessedBlue)))[0]
        AlgoprocessedGreen = np.array(AlgoprocessedGreen).reshape(1, (len(AlgoprocessedGreen)))[0]
        AlgoprocessedRed = np.array(AlgoprocessedRed).reshape(1, (len(AlgoprocessedRed)))[0]
        AlgoprocessedGrey = np.array(AlgoprocessedGrey).reshape(1, (len(AlgoprocessedGrey)))[0]
        # if (len(AlgoprocessedIR) > 0):
        AlgoprocessedIR = np.array(AlgoprocessedIR).reshape(1, (len(AlgoprocessedIR)))[0]

        return AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR

    def ApplyFastICAonIndividualChannels(self, processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                                         components):
        # if (len(processedGrey) > 0):
        AlgoprocessedBlue = self.objAlgorithm.ApplyICA(processedBlue, components)
        AlgoprocessedGreen = self.objAlgorithm.ApplyICA(processedGreen, components)
        AlgoprocessedRed = self.objAlgorithm.ApplyICA(processedRed, components)
        AlgoprocessedGrey = self.objAlgorithm.ApplyICA(processedGrey, components)
        # if (len(processedIR) > 0):
        AlgoprocessedIR = self.objAlgorithm.ApplyICA(processedIR, components)

        # Reshpae
        AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR = self.ReshapeArray(
            AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR)

        return AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR

    def ApplyFastICAonIndividualChannelsWithoutReshape(self, processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                                         components):
        # if (len(processedGrey) > 0):
        AlgoprocessedBlue = self.objAlgorithm.ApplyICA(processedBlue, components)
        AlgoprocessedGreen = self.objAlgorithm.ApplyICA(processedGreen, components)
        AlgoprocessedRed = self.objAlgorithm.ApplyICA(processedRed, components)
        AlgoprocessedGrey = self.objAlgorithm.ApplyICA(processedGrey, components)
        # if (len(processedIR) > 0):
        AlgoprocessedIR = self.objAlgorithm.ApplyICA(processedIR, components)

        return AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR

    def ApplyPCAonIndividualChannels(self, processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                                     components):
        AlgoprocessedBlue = self.objAlgorithm.ApplyPCA(processedBlue, components)
        AlgoprocessedGreen = self.objAlgorithm.ApplyPCA(processedGreen, components)
        AlgoprocessedRed = self.objAlgorithm.ApplyPCA(processedRed, components)
        AlgoprocessedGrey = self.objAlgorithm.ApplyPCA(processedGrey, components)
        AlgoprocessedIR = self.objAlgorithm.ApplyPCA(processedIR, components)


        return AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR

    '''
    ApplyAlgorithm: Applies algorithms on signal data
    '''

    def ApplyAlgorithm(self, processedBlue, processedGreen, processedRed, processedGrey, processedIR):
        S = np.c_[processedBlue, processedGreen, processedRed, processedGrey]
        self.components = 4
        if (len(processedIR) == len(processedGrey)):
            S = self.getSignalDataCombined(processedBlue, processedGreen, processedRed, processedGrey, processedIR)
            self.components = 5
        # colorShape = np.array(processedGrey).shape
        # IRShape = np.array(processedIR).shape
        if (self.Algorithm_type == 'None'):
            skip = 0
        else:
            if (len(processedGrey) > 0):
                processedBlue = np.array(processedBlue).reshape((len(processedBlue), 1))
                processedGreen = np.array(processedGreen).reshape((len(processedGreen), 1))
                processedRed = np.array(processedRed).reshape((len(processedRed), 1))
                processedGrey = np.array(processedGrey).reshape((len(processedGrey), 1))
            if (len(processedIR) > 0):
                processedIR = np.array(processedIR).reshape((len(processedIR), 1))
        # Apply by Algorithm_type
        # self.components = 1  # makes no difference with 1 to 15 components
        if (self.Algorithm_type == "FastICA"):  # ApplyICA for each channel with 1 compoenent
            AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR = self.ApplyFastICAonIndividualChannels(
                processedBlue, processedGreen, processedRed, processedGrey,
                processedIR, 1)  # self.components
        elif (self.Algorithm_type == "FastICAComponents3"):
            AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR = self.ApplyFastICAonIndividualChannels(
                processedBlue, processedGreen, processedRed, processedGrey,
                processedIR, 3)  # self.components
        elif (self.Algorithm_type == "FastICAComponents5"):
            AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR = self.ApplyFastICAonIndividualChannels(
                processedBlue, processedGreen, processedRed, processedGrey,
                processedIR, 5)
        elif (self.Algorithm_type == "FastICAComponents10"):
            AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR = self.ApplyFastICAonIndividualChannels(
                processedBlue, processedGreen, processedRed, processedGrey,
                processedIR, 10)  # self.components
        elif (self.Algorithm_type == "FastICAComponents3Times"):
            # FirstTime
            AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR = self.ApplyFastICAonIndividualChannelsWithoutReshape(
                processedBlue, processedGreen, processedRed, processedGrey,
                processedIR, 1)  # self.components
            # SecondTime
            AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR = self.ApplyFastICAonIndividualChannelsWithoutReshape(
                AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR , 1)  # self.components
            # ThirdTime
            AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR = self.ApplyFastICAonIndividualChannelsWithoutReshape(
                AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR , 1)

            # Reshpae
            AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR = self.ReshapeArray(
                AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR)

        elif (self.Algorithm_type == "FastICACombined"):
            S_ = self.objAlgorithm.ApplyICA(S, self.components)
            if (len(processedGrey) > 0):
                AlgoprocessedBlue = S_[:, 0]
                AlgoprocessedGreen = S_[:, 1]
                AlgoprocessedRed = S_[:, 2]
                AlgoprocessedGrey = S_[:, 3]
            if (len(processedIR) > 0):
                AlgoprocessedIR = S_[:, 4]

        elif (self.Algorithm_type == "PCA"):
            AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR = self.ApplyPCAonIndividualChannels(
                processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                1)

            # Reshpae
            AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR = self.ReshapeArray(
                AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR)

        elif (self.Algorithm_type == "PCACombined"):
            S_ = self.objAlgorithm.ApplyPCA(S, self.components)
            # plt.plot(AlgoprocessedGrey, 'grey')
            # plt.plot(S_[:, 3], 'black')
            # plt.show()
            if (len(S_[:, 3]) > 0):
                AlgoprocessedBlue = S_[:, 0]
                AlgoprocessedGreen = S_[:, 1]
                AlgoprocessedRed = S_[:, 2]
                AlgoprocessedGrey = S_[:, 3]
            if (len(S_[:, 4]) > 0):
                AlgoprocessedIR = S_[:, 4]

        elif (self.Algorithm_type == "PCAICA"):
            component = 1
            AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR = self.ApplyPCAonIndividualChannels(
                processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                component)
            AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR = self.ApplyFastICAonIndividualChannels(
                AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR, component)

        elif (self.Algorithm_type == "PCAICACombined"):
            S = self.objAlgorithm.ApplyPCA(S, self.components)
            S_ = self.objAlgorithm.ApplyICA(S, self.components)

            if (len(S_[:, 3]) > 0):
                AlgoprocessedBlue = S_[:, 0]
                AlgoprocessedGreen = S_[:, 1]
                AlgoprocessedRed = S_[:, 2]
                AlgoprocessedGrey = S_[:, 3]
            if (len(S_[:, 4]) > 0):
                AlgoprocessedIR = S_[:, 4]

        elif (self.Algorithm_type == "Jade"):
            # https://github.com/kellman/heartrate_matlab/blob/master/jadeR.m
            # r4 is slwoer and f5 is faster
            # S = np.c_[processedBlue,processedGreen,processedRed,processedGrey]
            S_ = self.objAlgorithm.jadeOptimised(S,
                                                 self.components)  # Only allows same or less components as array size
            # AlgoprocessedGreen = self.objAlgorithm.jadeOptimised(processedGreen, self.components)
            # Split data
            # newBlue = S_[0].real
            # newGreen = np.array(S_[1])[0].real
            # newRed = np.array(S_[2])[0].real
            # if (not self.ignoreGray):
            #     newGrey = np.array(S_[self.grayIndex])[0].real
            # newIr = np.array(S_[self.IRIndex])[0].real
            if (len(S_[3]) > 0):
                AlgoprocessedBlue = np.array(S_[0])[0].real  # S_[:, 0]
                AlgoprocessedGreen = np.array(S_[1])[0].real
                AlgoprocessedRed = np.array(S_[2])[0].real
                AlgoprocessedGrey = np.array(S_[3])[0].real

            if (self.components == 4):
                if (len(processedIR) > 0):
                    AlgoprocessedIR = self.objAlgorithm.jadeOptimised(processedIR, 1)
                    AlgoprocessedIR = np.array(AlgoprocessedIR[0])[
                        0]  # Only allows same or less components as array size
                    # AlgoprocessedIR = np.array(AlgoprocessedIR).reshape(1, (len(AlgoprocessedIR)))#[0]
            else:
                if (len(S_[4]) > 0):
                    AlgoprocessedIR = np.array(S_[4])[0].real
        else:
            AlgoprocessedBlue = processedBlue
            AlgoprocessedGreen = processedGreen
            AlgoprocessedRed = processedRed
            AlgoprocessedGrey = processedGrey
            AlgoprocessedIR = processedIR

        if (self.GenerateGraphs):
            self.GenerateGrapth("Algorithm", AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey,
                                AlgoprocessedIR)

        return AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR

    def Filterfft_BelowandAbove(self, sig_fft, type):
        sig_fft_filtered = abs(sig_fft.copy())
        if (type == 'color'):
            sig_fft_filtered[np.abs(self.Colorfrequency) <= self.ignore_freq_below] = 0
            sig_fft_filtered[np.abs(self.Colorfrequency) >= self.ignore_freq_above] = 0
        else:
            sig_fft_filtered[np.abs(self.IRfrequency) <= self.ignore_freq_below] = 0
            sig_fft_filtered[np.abs(self.IRfrequency) >= self.ignore_freq_above] = 0

        return sig_fft_filtered

    def Filterfft_Below(self, sig_fft, type):
        sig_fft_filtered = abs(sig_fft.copy())
        if (type == 'color'):
            sig_fft_filtered[np.abs(self.Colorfrequency) <= self.ignore_freq_below] = 0
        else:
            sig_fft_filtered[np.abs(self.IRfrequency) <= self.ignore_freq_below] = 0
        return sig_fft_filtered

    def Filterfft_cuttoff(self, fftarray, type):

        if (type == 'color'):
            bound_low = (np.abs(self.Colorfrequency - self.ignore_freq_below)).argmin()
            bound_high = (np.abs(self.Colorfrequency - self.ignore_freq_above)).argmin()
            fftarray[:bound_low] = 0
            fftarray[bound_high:-bound_high] = 0
            fftarray[-bound_low:] = 0
            return fftarray
        else:

            bound_low = (np.abs(self.IRfrequency - self.ignore_freq_below)).argmin()
            bound_high = (np.abs(self.IRfrequency - self.ignore_freq_above)).argmin()
            fftarray[:bound_low] = 0
            fftarray[bound_high:-bound_high] = 0
            fftarray[-bound_low:] = 0
            return fftarray

    def butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y

    def Filterfft_butterbandpass(self, blue, green, red, grey, Ir, ordno):
        # change for virable estimated fps
        B_bp = self.butter_bandpass_filter(blue, self.ignore_freq_below, self.ignore_freq_above, self.ColorEstimatedFPS,
                                           order=ordno)
        G_bp = self.butter_bandpass_filter(green, self.ignore_freq_below, self.ignore_freq_above,
                                           self.ColorEstimatedFPS,
                                           order=ordno)
        R_bp = self.butter_bandpass_filter(red, self.ignore_freq_below, self.ignore_freq_above, self.ColorEstimatedFPS,
                                           order=ordno)
        if (not self.ignoreGray):
            Gy_bp = self.butter_bandpass_filter(grey, self.ignore_freq_below, self.ignore_freq_above,
                                                self.ColorEstimatedFPS,
                                                order=ordno)
        else:
            Gy_bp = []
        IR_bp = self.butter_bandpass_filter(Ir, self.ignore_freq_below, self.ignore_freq_above, self.IREstimatedFPS,
                                            order=ordno)
        return B_bp, G_bp, R_bp, Gy_bp, IR_bp

    def Filterfft_LimitFreq_Belowfilter(self, signal, type):
        if (type == 'color'):
            interest_idx = np.where((self.Colorfrequency >= self.ignore_freq_below))[0]
            interest_idx_sub = interest_idx.copy()  # [:-1] advoid the indexing error
            fft_of_interest = signal[interest_idx_sub]
            self.Colorfrequency = self.Colorfrequency[interest_idx_sub]
            return fft_of_interest
        else:
            interest_idx = np.where((self.IRfrequency >= self.ignore_freq_below))[0]
            interest_idx_sub = interest_idx.copy()  # [:-1] advoid the indexing error
            fft_of_interest = signal[interest_idx_sub]
            self.IRfrequency = self.IRfrequency[interest_idx_sub]
            return fft_of_interest

    def Filterfft_LimitFreq_BelowAbovefilter(self, signal, type):

        if (type == 'color'):
            interest_idx = \
                np.where(
                    (self.Colorfrequency >= self.ignore_freq_below) & (self.Colorfrequency <= self.ignore_freq_above))[
                    0]
            interest_idx_sub = interest_idx.copy()  # [:-1] advoid the indexing error
            fft_of_interest = signal[interest_idx_sub]
            self.Colorfrequency = self.Colorfrequency[interest_idx_sub]
            return fft_of_interest
        else:
            interest_idx = \
                np.where((self.IRfrequency >= self.ignore_freq_below) & (self.IRfrequency <= self.ignore_freq_above))[0]
            interest_idx_sub = interest_idx.copy()  # [:-1] advoid the indexing error
            fft_of_interest = signal[interest_idx_sub]
            self.IRfrequency = self.IRfrequency[interest_idx_sub]
            return fft_of_interest

    '''
    FilterTechniques: Applies filters on signal data
    '''

    def FilterTechniques(self, B_fft, G_fft, R_fft, Gy_fft, IR_fft):
        # cuttoff,  butterworth and other
        if (self.ignoreGray):
            Gy_filtered = []

        if (self.Filter_type == 1):  # Not very good with rampstuff method, very high heart rate
            B_filtered = self.Filterfft_BelowandAbove(B_fft, 'color')
            G_filtered = self.Filterfft_BelowandAbove(G_fft, 'color')
            R_filtered = self.Filterfft_BelowandAbove(R_fft, 'color')
            if (not self.ignoreGray):
                Gy_filtered = self.Filterfft_BelowandAbove(Gy_fft, 'color')
            IR_filtered = self.Filterfft_BelowandAbove(IR_fft, 'ir')

        elif (self.Filter_type == 2):  #

            B_filtered = self.Filterfft_Below(B_fft, 'color')
            G_filtered = self.Filterfft_Below(G_fft, 'color')
            R_filtered = self.Filterfft_Below(R_fft, 'color')
            if (not self.ignoreGray):
                Gy_filtered = self.Filterfft_Below(Gy_fft, 'color')
            IR_filtered = self.Filterfft_Below(IR_fft, 'ir')

        elif (self.Filter_type == 3):
            B_filtered = self.Filterfft_cuttoff(B_fft, 'color')
            G_filtered = self.Filterfft_cuttoff(G_fft, 'color')
            R_filtered = self.Filterfft_cuttoff(R_fft, 'color')
            if (not self.ignoreGray):
                Gy_filtered = self.Filterfft_cuttoff(Gy_fft, 'color')
            IR_filtered = self.Filterfft_cuttoff(IR_fft, 'ir')

        elif (self.Filter_type == 4):  # Not very good with rampstuff method
            B_filtered, G_filtered, R_filtered, Gy_filtered, IR_filtered = self.Filterfft_butterbandpass(B_fft, G_fft,
                                                                                                         R_fft, Gy_fft,
                                                                                                         IR_fft, 6)

        elif (self.Filter_type == 5):
            # No Filter
            B_filtered = B_fft
            G_filtered = G_fft
            R_filtered = R_fft
            if (not self.ignoreGray):
                Gy_filtered = Gy_fft
            IR_filtered = IR_fft

        elif (self.Filter_type == 6):

            B_filtered = self.Filterfft_LimitFreq_BelowAbovefilter(B_fft, 'color')
            G_filtered = self.Filterfft_LimitFreq_BelowAbovefilter(G_fft, 'color')
            R_filtered = self.Filterfft_LimitFreq_BelowAbovefilter(R_fft, 'color')
            if (not self.ignoreGray):
                Gy_filtered = self.Filterfft_LimitFreq_BelowAbovefilter(Gy_fft, 'color')
            IR_filtered = self.Filterfft_LimitFreq_BelowAbovefilter(IR_fft, 'ir')


        elif (self.Filter_type == 7):

            B_filtered = self.Filterfft_LimitFreq_Belowfilter(B_fft, 'color')
            G_filtered = self.Filterfft_LimitFreq_Belowfilter(G_fft, 'color')
            R_filtered = self.Filterfft_LimitFreq_Belowfilter(R_fft, 'color')
            if (not self.ignoreGray):
                Gy_filtered = self.Filterfft_LimitFreq_Belowfilter(Gy_fft, 'color')
            IR_filtered = self.Filterfft_LimitFreq_Belowfilter(IR_fft, 'ir')

        return B_filtered, G_filtered, R_filtered, Gy_filtered, IR_filtered  # as differnt fps changes array length

    '''
    SmoothenData: Smooth data
    '''

    def smooth(self, x, window_len=11, window='hamming'):
        # """smooth the data using a window with requested size.
        #
        # This method is based on the convolution of a scaled window with the signal.
        # The signal is prepared by introducing reflected copies of the signal
        # (with the window size) in both ends so that transient parts are minimized
        # in the begining and end part of the output signal.
        #
        # input:
        #     x: the input signal
        #     window_len: the dimension of the smoothing window; should be an odd integer
        #     window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
        #     flat window will produce a moving average smoothing.
        # output:
        #     the smoothed signal
        #
        # example:
        #     t=linspace(-2,2,0.1)
        #     x=sin(t)+randn(len(t))*0.1
        #     y=smooth(x)
        #
        # see also:
        #
        #     numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
        #     scipy.signal.lfilter
        #
        # NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
        # """

        if x.ndim != 1:
            raise ValueError('Smooth only accepts 1 dimension arrays.')

        if x.size < window_len:
            raise ValueError('Input vector needs to be bigger than window size.')

        if window_len < 3:
            return x

        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

        s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
        # print(len(s))
        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.' + window + '(window_len)')

        y = np.convolve(w / w.sum(), s, mode='valid')
        return y

    def SmoothenData(self, Signal):
        SmoothenSignal = self.smooth(np.array(Signal))
        # if (not self.ignoreGray):
        #     Smoothengrey = self.smooth(S_[:, self.grayIndex])
        # else:
        #     Smoothengrey = []
        return SmoothenSignal

    '''
    ApplyFFT: types of fft on signal data
    '''

    def ApplyFFT(self, AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR):
        if (self.FFT_type == "M1"):
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, Colorfreq, IRfreq = self.objAlgorithm.Apply_FFT_WithoutPower_M4_eachsignal(
                AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR,
                self.ignoreGray)  # rfft

        elif (self.FFT_type == "M2"):  # with fft shift
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, Colorfreq, IRfreq = self.objAlgorithm.Apply_FFT_M2_eachsignal(
                AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR,
                self.ignoreGray)  # rfft

        if (self.FFT_type == "M3"):
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, Colorfreq, IRfreq = self.objAlgorithm.Apply_FFT_M1_byeachsignal(
                AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR,
                self.ignoreGray)  # rfft

        elif (self.FFT_type == "M4"):
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, Colorfreq, IRfreq = self.objAlgorithm.ApplyFFT9(AlgoprocessedBlue,
                                                                                                  AlgoprocessedGreen,
                                                                                                  AlgoprocessedRed,
                                                                                                  AlgoprocessedGrey,
                                                                                                  AlgoprocessedIR,
                                                                                                  self.ignoreGray)  # fft


        elif (self.FFT_type == "M5"):
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, Colorfreq, IRfreq = self.objAlgorithm.Apply_FFT_M6_Individual(
                AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR,
                self.ignoreGray)  # sqrt

        elif (self.FFT_type == "M6"):  # with fft shift
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, Colorfreq, IRfreq = self.objAlgorithm.Apply_FFT_M5_Individual(
                AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR,
                self.ignoreGray)  # sqrt

        # elif (self.FFT_type == "M7"):
        #     B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, Colorfreq, IRfreq = self.objAlgorithm.Apply_FFT_forpreprocessed(
        #         AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR,
        #         self.ignoreGray)  # sqrt

        if (np.iscomplex(B_fft.any())):
            B_fft = B_fft.real

        if (np.iscomplex(Gr_fft.any())):
            Gr_fft = Gr_fft.real

        if (np.iscomplex(R_fft.any())):
            R_fft = R_fft.real

        if (not self.ignoreGray):
            if (np.iscomplex(Gy_fft.any())):
                Gy_fft = Gy_fft.real
        else:
            Gy_fft = []

        if (np.iscomplex(IR_fft.any())):
            IR_fft = IR_fft.real

        self.Colorfrequency = Colorfreq
        self.IRfrequency = IRfreq

        return B_fft, Gr_fft, R_fft, Gy_fft, IR_fft

    def WritetoDisk(self, location, filename, data):
        ##STORE Data
        with open(location + filename, 'wb') as filehandle:
            pickle.dump(data, filehandle)

    def StoreData(self, filename, blue, green, red, grey, Ir, startTime, endTime, calculateDiff=False):
        diffTime = None
        objProcessedData = ProcessedData()
        objProcessedData.B_signal = blue
        objProcessedData.G_signal = green
        objProcessedData.R_signal = red
        objProcessedData.Gy_signal = grey
        objProcessedData.IR_signal = Ir
        objProcessedData.ColorFPSwithTime = self.WindowColorfpswithTime
        objProcessedData.IRFPSwithTime = self.WindowIRfpswithTime
        objProcessedData.IREstimatedFPS = self.IREstimatedFPS
        objProcessedData.ColorEstimatedFPS = self.ColorEstimatedFPS
        objProcessedData.WindowCount = self.Window_count
        objProcessedData.startTime = startTime
        objProcessedData.endTime = endTime
        if (calculateDiff):
            diffTime = (endTime - startTime)
        objProcessedData.diffTime = diffTime

        # Save
        self.WritetoDisk(self.SavePath + self.Window_count + '\\', 'objWindowProcessedData_' + filename,
                         objProcessedData)  #

    def getSignalforStoring(self, processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                            Colorfreq=False, IRfreq=False):
        Signal = {}
        Signal['blue'] = processedBlue
        Signal['green'] = processedGreen
        Signal['red'] = processedRed
        Signal['grey'] = processedGrey
        Signal['Ir'] = processedIR
        if (Colorfreq):
            Signal['Colorfrequency'] = self.Colorfrequency
        if (IRfreq):
            Signal['IRfrequency'] = self.IRfrequency
        return Signal

    '''
    Process_EntireSignalData: Process signal data
    '''

    def Process_EntireSignalData(self, IsEntireSignal=True):  # TODO : Implement without Gray
        # window data object
        windowList = Window_Data()
        windowList.WindowNo = self.Window_count
        windowList.LogTime(LogItems.Start_Total)
        windowList.isSmooth = self.isSmoothen
        windowList.fileName = self.fileName

        blue = self.regionWindowBlueData
        green = self.regionWindowGreenData
        red = self.regionWindowRedData
        grey = self.regionWindowGreyData
        Irchannel = self.regionWindowIRData
        distanceM = self.WindowdistanceM

        # Record Window Raw data
        # self.StoreData('RawSignal', blue, green, red, grey, Irchannel, None, None)

        # generate raw data plot
        if (self.GenerateGraphs):
            self.GenerateGrapth("RawData", blue, green, red, grey,
                                Irchannel)  # todo: put condition for if array len is 0

        # Log Start Time preprcessing signal data
        startlogTime = windowList.LogTime(LogItems.Start_PreProcess)
        # PreProcess Signal
        processedBlue, processedGreen, processedRed, processedGrey, processedIR = self.preprocessSignalData(blue, green,
                                                                                                            red, grey,
                                                                                                            Irchannel)
        endlogTime = windowList.LogTime(LogItems.End_PreProcess)

        # # Record Window Raw data
        windowList.SignalWindowPreProcessed = self.getSignalforStoring(processedBlue, processedGreen, processedRed,
                                                                       processedGrey, processedIR)
        # self.StoreData('PreProcess', processedBlue, processedGreen, processedRed, processedGrey, processedIR,
        #                startlogTime, endlogTime, True)

        # Apply Algorithm
        startlogTime = windowList.LogTime(LogItems.Start_Algorithm)
        AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey, AlgoprocessedIR = self.ApplyAlgorithm(
            processedBlue, processedGreen, processedRed, processedGrey, processedIR)
        endlogTime = windowList.LogTime(LogItems.End_Algorithm)

        # Record Window Raw data
        windowList.SignalWindowAfterAlgorithm = self.getSignalforStoring(AlgoprocessedBlue, AlgoprocessedGreen,
                                                                         AlgoprocessedRed, AlgoprocessedGrey,
                                                                         AlgoprocessedIR)
        # self.StoreData('Algorithm', AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey,
        #                AlgoprocessedIR, startlogTime, endlogTime, True)

        # Apply smoothen only before fft
        # self.isSmoothen= True
        if (self.isSmoothen):
            # Smooth data
            startlogTime = windowList.LogTime(LogItems.Start_Smooth)
            if (len(processedGrey) > 0):
                AlgoprocessedBlue = self.SmoothenData(AlgoprocessedBlue)
                AlgoprocessedGreen = self.SmoothenData(AlgoprocessedGreen)
                AlgoprocessedRed = self.SmoothenData(AlgoprocessedRed)
                AlgoprocessedGrey = self.SmoothenData(AlgoprocessedGrey)
            if (len(processedIR) > 0):
                AlgoprocessedIR = self.SmoothenData(AlgoprocessedIR)
            endlogTime = windowList.LogTime(LogItems.End_Smooth)

            if (self.GenerateGraphs):
                self.GenerateGrapth("Smooth", AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed,
                                    AlgoprocessedGrey, AlgoprocessedIR)

            # Record Window  data
            windowList.SignalWindowSmoothed = self.getSignalforStoring(AlgoprocessedBlue, AlgoprocessedGreen,
                                                                       AlgoprocessedRed, AlgoprocessedGrey,
                                                                       AlgoprocessedIR)
            # Record Window Raw data
            # self.StoreData('Smoothed', AlgoprocessedBlue, AlgoprocessedGreen, AlgoprocessedRed, AlgoprocessedGrey,
            #                AlgoprocessedIR, startlogTime, endlogTime, True)

        # Apply fft
        startlogTime = windowList.LogTime(LogItems.Start_FFT)
        FFTBlue, FFTGreen, FFTRed, FFTGrey, FFTIR = self.ApplyFFT(AlgoprocessedBlue,
                                                                  AlgoprocessedGreen,
                                                                  AlgoprocessedRed,
                                                                  AlgoprocessedGrey, AlgoprocessedIR)
        endlogTime = windowList.LogTime(LogItems.End_FFT)

        # Record Window  data
        windowList.SignalWindowFFT = self.getSignalforStoring(FFTBlue, FFTGreen, FFTRed, FFTGrey, FFTIR, True, True)
        # # Record Window Raw data
        # self.StoreData('FFT', FFTBlue, FFTGreen, FFTRed, FFTGrey, FFTIR, startlogTime, endlogTime, True)

        if (self.GenerateGraphs):
            self.GenerateGrapth("FFT", FFTBlue, FFTGreen, FFTRed, FFTGrey, FFTIR)

        startlogTime = windowList.LogTime(LogItems.Start_Filter)
        B_filtered, G_filtered, R_filtered, Gy_filtered, IR_filtered = self.FilterTechniques(FFTBlue, FFTGreen, FFTRed,
                                                                                             FFTGrey,
                                                                                             FFTIR)  ##Applyfiltering
        endlogTime = windowList.LogTime(LogItems.End_Filter)

        if (self.GenerateGraphs):
            self.GenerateGrapth("Filtered", B_filtered, G_filtered, R_filtered, Gy_filtered, IR_filtered)

        # Record Window Raw data
        # self.StoreData('Filtered', B_filtered, G_filtered, R_filtered, Gy_filtered, IR_filtered, startlogTime, endlogTime, True)
        windowList.SignalWindowFiltered = self.getSignalforStoring(B_filtered, G_filtered, R_filtered, Gy_filtered,
                                                                   IR_filtered, True, True)

        startlogTime = windowList.LogTime(LogItems.Start_ComputerHRSNR)
        self.generateHeartRateandSNR(B_filtered, G_filtered, R_filtered, Gy_filtered, IR_filtered, self.Result_type)

        # get best bpm and heart rate period in one region
        self.bestHeartRateSnr = 0.0
        self.bestBpm = 0.0
        self.GetBestBpm()
        endlogTime = windowList.LogTime(LogItems.End_ComputerHRSNR) #TODO:RERUN -> THIS WAS BEFORE BEST BPM

        # calculate SPO
        windowList.LogTime(LogItems.Start_SPO)

        Gy_filteredCopy = Gy_filtered
        greyCopy = grey
        # redCopy = red
        if (IsEntireSignal):
            if (len(grey) > len(self.regionIRData)):
                greyCopy = grey.copy()
                lengthDiff = len(greyCopy) - len(self.regionIRData)
                for i in range(lengthDiff):
                    greyCopy.pop()
            if (len(Gy_filtered) > len(self.regionIRData)):
                Gy_filteredCopy = Gy_filtered.copy()
                Gy_filteredCopy = Gy_filteredCopy[0:len(self.regionIRData)]  # all but the first and last element
            # if (len(red) > len(self.regionIRData)):
            #     redCopy = red.copy()
            #     lengthDiff = len(redCopy) - len(self.regionIRData)
            #     for i in range(lengthDiff):
            #         redCopy.pop()
        else:
            Gy_filteredCopy = Gy_filtered
            greyCopy = grey
            # redCopy = red

        std, err, oxylevl = self.getSpo(greyCopy, Gy_filteredCopy, self.regionIRData, red,
                                        self.distanceM)  # Irchannel and distanceM as IR channel lengh can be smaller so passing full array
        windowList.LogTime(LogItems.End_SPO)

        windowList.LogTime(LogItems.End_Total)

        # SPO
        windowList.SignalWindowSPOgrey = grey
        windowList.SignalWindowSPOIrchannel = Irchannel
        windowList.SignalWindowSPOred = red
        windowList.SignalWindowSPOdistance = distanceM
        windowList.SignalWindowSPOGy_filtered = Gy_filtered

        # HR
        windowList.SNRSummary = self.SNRSummary
        windowList.channeltype = self.channeltype
        windowList.WindowNo = self.Window_count
        windowList.BestBPM = self.bestBpm
        windowList.BestSnR = self.bestHeartRateSnr
        windowList.IrSnr = self.IrSnr
        windowList.GreySnr = self.GreySnr
        windowList.RedSnr = self.RedSnr
        windowList.GreenSnr = self.GreenSnr
        windowList.BlueSnr = self.BlueSnr
        windowList.BlueBpm = self.BlueBpm
        windowList.IrBpm = self.IrBpm
        windowList.GreyBpm = self.GreyBpm
        windowList.RedBpm = self.RedBpm
        windowList.GreenBpm = self.GreenBpm
        windowList.regiontype = self.region
        windowList.IrFreqencySamplingError = self.IrFreqencySamplingError
        windowList.GreyFreqencySamplingError = self.GreyFreqencySamplingError
        windowList.RedFreqencySamplingError = self.RedFreqencySamplingError
        windowList.GreenFreqencySamplingError = self.GreenFreqencySamplingError
        windowList.BlueFreqencySamplingError = self.BlueFreqencySamplingError
        windowList.oxygenSaturationSTD = std  # std
        windowList.oxygenSaturationValueError = err  # err
        windowList.oxygenSaturationValueValue = oxylevl  # oxylevl
        windowList.timeDifferences()

        return windowList

    # region CalcualateSPO
    '''
    CalcualateSPO: Calculate SPO using ifft channel, original red and ir channels as input
    '''

    #
    # def CalcualateSPO(self, Gy_bp, heartRatePeriod, IR_bp, R_bp, G_bp, distanceM, smallestOxygenError, region):
    #
    #     self.OxygenSaturation = 0.0
    #     self.OxygenSaturationError = 0.0
    #
    #     greyPeeks, properties, filteredgrey = self.FindGreyPeak(Gy_bp, freqs, self.ignore_freq_below,
    #                                                             self.ignore_freq_above)
    #     # find the max peeks in each sample window
    #     grayMaxPeekIndice = []
    #
    #     formid = len(filteredgrey) - int(heartRatePeriod)  # grey channel orinigal before
    #     for x in range(0, formid, int(heartRatePeriod)):
    #         maxGreyPeekInPeriod = self.FindMaxPeekInPeriod(greyPeeks, filteredgrey, int(heartRatePeriod), x)
    #         grayMaxPeekIndice.append((maxGreyPeekInPeriod))
    #
    #     # get the values of the ir and red channels at the grey peeks
    #     oxygenLevels = []
    #     for index in grayMaxPeekIndice:
    #         # if self.ignore_freq_below <= freqs[index] <= self.ignore_freq_above:
    #         irValue = np.abs(IR_bp[index].real)  # ir cahnnel original before
    #         redValue = R_bp[index].real  # red cahnnel original before
    #         if (irValue > 0):
    #             # irValue = Infra_fftMaxVal
    #             # redValue = red_fftMaxVal  # red cahnnel original before
    #             # greenValue = gy_fftMaxVal  # green cahnnel original before
    #
    #             distValue = distanceM[index]  # 0.7 #self.distanceMm[index]
    #             # distValue = distValue*1000 #convert to mm
    #             # Absorption values calculated of oxy and deoxy heamoglobin
    #             # where red defined as: 600-700nm
    #             # where ir defined as: 858-860nm.
    #             red_deoxy_mean = 4820.4361
    #             red_oxy_mean = 667.302
    #             ir_deoxy_mean = 694.32  # 693.38
    #             ir_oxy_mean = 1092  # 1087.2
    #
    #             # Depth-resultion blood oxygen satyuration measuremnet by dual-wavelength photothermal (DWP) optical coherence tomography
    #             # Biomed Opt Express
    #             # 2011 P491-504
    #             irToRedRatio = (red_oxy_mean / ir_oxy_mean) * (
    #             (irValue * ((distValue * distValue) / 1000000) / redValue)) / 52
    #             oxygenLevel = 100 * (red_deoxy_mean - (irToRedRatio * ir_deoxy_mean)) / (
    #                         ir_oxy_mean + red_deoxy_mean - ir_deoxy_mean - red_oxy_mean) - 2
    #
    #             # irToRedRatio = (red_oxy_mean / ir_oxy_mean) * ((irValue * ((
    #             #         distValue / 100000)) / redValue)) / 52  # /1000 at a distance of one meter using 99% white reflections standard, 52 is the ir/r scaling factor.
    #             # oxygenLevel = 100 * (red_deoxy_mean - (irToRedRatio * ir_deoxy_mean)) / (
    #             #         ir_oxy_mean + red_deoxy_mean - ir_deoxy_mean - red_oxy_mean) - 6  # -2 only for pca
    #             oxygenLevels.append(oxygenLevel)
    #
    #             oxygenLevels.append(round(oxygenLevel))
    #
    #     # compute SD and mean of oxygenLevels
    #     self.OxygenSaturation = np.std(oxygenLevels)
    #     self.OxygenSaturationError = np.std(oxygenLevels, ddof=1) / np.sqrt(
    #         np.size(oxygenLevels))  # MeanStandardDeviation err
    #
    #     if (self.OxygenSaturationError < smallestOxygenError):
    #         self.smallestOxygenError = self.OxygenSaturationError
    #         self.regionToUse = region
    #
    #         # if (self.OxygenSaturation > self.oxygenstd):
    #         #     oxygenstd = self.OxygenSaturation
    #         #     print("STD : " + str(self.OxygenSaturation) + " , error: " + str(self.OxygenSaturationError))
    #         #     print("SPO 0 : " + str(oxygenLevels[0]))
    #
    #     if (len(oxygenLevels) <= 0):
    #         oxygenLevels.append(0)
    #     return self.OxygenSaturation, self.OxygenSaturationError, str(oxygenLevels[0])

    def getSpo(self, OriginalGrey, G_bp, Irchannel, red, distanceM):
        # =====================================SPO =====================================
        # =====================================SPO =====================================
        # we select the one that has the smallest error
        smallestOxygenError = sys.float_info.max  # double.MaxValue
        regionToUse = ''
        oxygenSaturationValueError = 0.0  ## udpate this everytime run through loop
        oxygenSaturationValueValue = 0.0
        oxygenstd = 0.0
        finaloxy = 0.0
        filteredgrey = np.fft.ifft(G_bp)
        filteredir = Irchannel
        filteredred = red
        # if(len(filteredgrey)> )
        std, err, oxylevl = self.CalcualateSPOWithout(OriginalGrey, filteredgrey, self.heartRatePeriod, filteredir,
                                                      filteredred,
                                                      None,
                                                      distanceM, smallestOxygenError,
                                                      self.region)  # CalcualateSPOPart2 ,CalcualateSPOWithout

        oxygenSaturationValueError = self.OxygenSaturationError
        oxygenSaturationValueValue = self.OxygenSaturation

        return std, err, float(oxylevl)

    # endregion

    def CalcualateSPOWithout(self, OriginalGrey, filteredGrey, heartRatePeriod, IR_bp, R_bp, G_bp, distanceM,
                             smallestOxygenError, region):

        self.OxygenSaturation = 0.0
        self.OxygenSaturationError = 0.0

        # do for each region
        numSamples = len(filteredGrey)
        greyPeeks = self.FindPeeks(filteredGrey)

        # find the max peeks in each sample window
        grayMaxPeekIndice = []

        loopTill = len(OriginalGrey) - int(heartRatePeriod)  # grey channel orinigal before

        for x in range(0, loopTill, int(heartRatePeriod)):
            maxGreyPeekInPeriod = self.FindMaxPeekInPeriod(greyPeeks, filteredGrey, int(heartRatePeriod), x)
            grayMaxPeekIndice.append((maxGreyPeekInPeriod))

        # get the values of the ir and red channels at the grey peeks
        oxygenLevels = []
        newgrayMaxPeekIndice = []
        # if(len(grayMaxPeekIndice)> len(IR_bp)):
        #     #remove zeros from gray peak array
        #     grayMaxPeekIndice = [i for i in grayMaxPeekIndice if i != 0]
        #     for item in grayMaxPeekIndice:
        #         newIndex = item/len(IR_bp)
        #         newgrayMaxPeekIndice.append(int(round(newIndex)))
        # else:
        #     newgrayMaxPeekIndice =grayMaxPeekIndice

        for index in grayMaxPeekIndice:
            # if self.ignore_freq_below <= freqs[index] <= self.ignore_freq_above:

            irValue = IR_bp[index].real  # ir cahnnel original before
            redValue = R_bp[index].real  # red cahnnel original before
            # greenValue = G_bp[index].real  # green cahnnel original before
            if (irValue > 0):
                # irValue = Infra_fftMaxVal
                # redValue = red_fftMaxVal  # red cahnnel original before
                # greenValue = gy_fftMaxVal  # green cahnnel original before

                distValue = distanceM[index]  # self.distanceMm[index]
                # if (distanceM[index] >= 2):
                #     distValue = distValue - 1

                # distValue = distValue*1000
                # distValue = distValue*1000 #convert to mm
                # Absorption values calculated of oxy and deoxy heamoglobin
                # where red defined as: 600-700nm
                # where ir defined as: 858-860nm.
                # red_deoxy_mean = 4820.4361
                # red_oxy_mean = 667.302
                ir_deoxy_mean = 694.32  # 693.38
                ir_oxy_mean = 1092  # 1087.2
                red_deoxy_mean = 4820.4361
                red_oxy_mean = 667.302
                # ir_deoxy_mean = 693.38
                # ir_oxy_mean = 1087.2
                # Depth-resultion blood oxygen satyuration measuremnet by dual-wavelength photothermal (DWP) optical coherence tomography
                # Biomed Opt Express
                # 2011 P491-504
                irToRedRatio = (red_oxy_mean / ir_oxy_mean) * (
                    (irValue * ((distValue * distValue) / 1000000) / redValue)) / 52
                oxygenLevel = 100 * (red_deoxy_mean - (irToRedRatio * ir_deoxy_mean)) / (
                        ir_oxy_mean + red_deoxy_mean - ir_deoxy_mean - red_oxy_mean) - 6
                oxygenLevels.append(oxygenLevel)

        # compute SD and mean of oxygenLevels
        self.OxygenSaturation = np.std(oxygenLevels)
        self.OxygenSaturationError = np.std(oxygenLevels, ddof=1) / np.sqrt(
            np.size(oxygenLevels))  # MeanStandardDeviation err

        if (len(oxygenLevels) <= 0):
            oxygenLevels.append(0)
        return self.OxygenSaturation, self.OxygenSaturationError, str(oxygenLevels[0])

    def FindMaxPeekInPeriod(self, indicesOfPeeks, samples, period, startIdx):

        maxPeek = float('-inf')
        maxPeekIdx = 0
        for x in range(0, len(indicesOfPeeks)):
            index = indicesOfPeeks[x]
            if (index >= startIdx and index < startIdx + period):  # make sure it's within the window
                value = samples[index].real
                if (value > maxPeek):
                    maxPeek = value
                    maxPeekIdx = index

        return maxPeekIdx

    def FindPeeks(self, arry):
        peeks = []
        # for every consecutive triple in samples
        for x in range(1, len(arry) - 1):
            if (arry[x - 1].real <= arry[x].real and arry[x].real >= arry[x + 1].real):
                peeks.append(x)

        return peeks

    # def FindGreyPeak(self, fftarray, freqs, freq_min, freq_max):
    #     fft_maximums = []
    #
    #     for i in range(fftarray.shape[0]):
    #         if freq_min <= freqs[i] <= freq_max:
    #             fftMap = abs(fftarray[i].real)
    #             fft_maximums.append(fftMap.max())
    #         else:
    #             fft_maximums.append(0)
    #
    #     filteredgrey = ifft(fft_maximums)
    #     peaks, properties = signal.find_peaks(filteredgrey)
    #
    #     return peaks, properties, filteredgrey

    def GetBestBpm(self):
        if (self.IrSnr > self.bestHeartRateSnr):
            self.bestHeartRateSnr = self.IrSnr
            self.bestBpm = self.IrBpm
            self.channeltype = 'IR'

        if (self.GreySnr > self.bestHeartRateSnr):
            self.bestHeartRateSnr = self.GreySnr
            self.bestBpm = self.GreyBpm
            self.channeltype = 'Grey'

        if (self.RedSnr > self.bestHeartRateSnr):
            self.bestHeartRateSnr = self.RedSnr
            self.bestBpm = self.RedBpm
            self.channeltype = 'Red'

        if (self.GreenSnr > self.bestHeartRateSnr):
            self.bestHeartRateSnr = self.GreenSnr
            self.bestBpm = self.GreenBpm
            self.channeltype = 'Green'

        if (self.BlueSnr > self.bestHeartRateSnr):
            self.bestHeartRateSnr = self.BlueSnr
            self.bestBpm = self.BlueBpm
            self.channeltype = 'Blue'

        # work out the length of time of one heart beat - the heart rate period
        if (self.bestBpm > 0):
            self.beatsPerSecond = self.bestBpm / 60.0
        else:
            self.beatsPerSecond = 1

        colorHRperiod = self.ColorEstimatedFPS / self.beatsPerSecond  # window size in sample
        IRHRperiod = self.IREstimatedFPS / self.beatsPerSecond  # window size in sample
        self.heartRatePeriod = IRHRperiod
        self.heartRatePeriod = round(self.heartRatePeriod)
        #
        # if (colorHRperiod == IRHRperiod):
        #     self.heartRatePeriod = IRHRperiod
        # else:
        #     if (IRHRperiod > colorHRperiod):
        #         self.heartRatePeriod = IRHRperiod
        #     else:
        #         self.heartRatePeriod = colorHRperiod

    # Not used
    # def ComputeIndices(self):
    #     self.ignore_freq_index_below = np.rint(
    #         ((self.ignore_freq_below * self.NumSamples) / self.EstimatedFPS))  # high pass
    #     self.ignore_freq_index_above = np.rint(
    #         ((self.ignore_freq_above * self.NumSamples) / self.EstimatedFPS))  # low pass
    #     # compute the ramp filter start and end indices double
    #     self.ramp_start = self.ignore_freq_index_below
    #     self.ramp_end = np.rint(((self.ramp_end_hz * self.NumSamples) / self.EstimatedFPS))
    #     self.rampDesignLength = self.ignore_freq_index_above - self.ignore_freq_index_below
    #     self.ramp_design = [None] * int(self.rampDesignLength)
    #     self.ramplooprange = int(self.ramp_end - self.ramp_start)
    #     # setup linear ramp
    #     for x in range(0, self.ramplooprange):
    #         self.ramp_design[x] = ((((self.ramp_end_percentage - self.ramp_start_percentage) / (
    #                 self.ramp_end - self.ramp_start)) * (
    #                                     x)) + self.ramp_start_percentage)
    #     # setup plateu of linear ramp
    #     for x in range(int(self.ramp_end - self.ramp_start),
    #                    int(self.ignore_freq_index_above - self.ignore_freq_index_below)):
    #         # ramp_design.append(1)
    #         self.ramp_design[x] = 1

    def signaltonoiseDB(self, a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        # return np.where(sd == 0, 0, m / sd)
        return 20 * np.log10(abs(np.where(sd == 0, 0, m / sd)))  # convertin got decibels

    def signaltonoise(self, a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m / sd)

    def getSNR(self, ir_fft_maxVal, ir_fft_realabs,
               grey_fft_maxVal, grey_fft_realabs,
               red_fft_maxVal, red_fft_realabs,
               green_fft_maxVal, green_fft_realabs,
               blue_fft_maxVal, blue_fft_realabs):

        if (self.snrType == 1):
            self.IrSnr = float(ir_fft_maxVal) / np.average(
                ir_fft_realabs) * 1  # * 1 # could artificially increase SNR for IR as provdes higher accuracy readings, enabling higher weighting for readings
            if (not self.ignoreGray):
                self.GreySnr = float(grey_fft_maxVal) / np.average(grey_fft_realabs)
            else:
                self.GreySnr = 0.0
            self.RedSnr = float(red_fft_maxVal) / np.average(red_fft_realabs)
            self.GreenSnr = float(green_fft_maxVal) / np.average(green_fft_realabs)
            self.BlueSnr = float(blue_fft_maxVal) / np.average(blue_fft_realabs)

        if (self.snrType == 2):
            self.IrSnr = self.signaltonoiseDB(ir_fft_realabs) * 1
            if (not self.ignoreGray):
                self.GreySnr = self.signaltonoiseDB(grey_fft_realabs)
            else:
                self.GreySnr = 0.0
            self.RedSnr = self.signaltonoiseDB(red_fft_realabs)
            self.GreenSnr = self.signaltonoiseDB(green_fft_realabs)
            self.BlueSnr = self.signaltonoiseDB(blue_fft_realabs)

        # ir2= self.signaltonoise(ir_fft_realabs)
        # gy2=self.signaltonoise(grey_fft_realabs)
        # red2=self.signaltonoise(red_fft_realabs)
        # green2=self.signaltonoise(green_fft_realabs)

    # def getSamplingError(self, ir_fft_index, grey_fft_index, red_fft_index, green_fft_index, blue_fft_index):
    #
    #     if ((ir_fft_index +1 > (len(self.freq_bpm) - 1))):
    #         ir_fft_index = (len(self.freq_bpm) - 2)
    #     if ((grey_fft_index+1 > (len(self.freq_bpm) - 1))):
    #         grey_fft_index = (len(self.freq_bpm) - 2)
    #     if ((red_fft_index+1 > (len(self.freq_bpm) - 1))):
    #         red_fft_index = (len(self.freq_bpm) - 2)
    #     if ((green_fft_index+1 > (len(self.freq_bpm) - 1))):
    #         green_fft_index = (len(self.freq_bpm) - 2)
    #     if ((blue_fft_index+1 > (len(self.freq_bpm) - 1))):
    #         blue_fft_index = (len(self.freq_bpm) - 2)
    #
    #     self.IrFreqencySamplingError = self.freq_bpm[ir_fft_index + 1] - self.freq_bpm[ir_fft_index - 1]
    #     if (not self.ignoreGray):
    #         self.GreyFreqencySamplingError = self.freq_bpm[grey_fft_index + 1] - self.freq_bpm[grey_fft_index - 1]
    #     else:
    #         self.GreyFreqencySamplingError = 0.0
    #     self.RedFreqencySamplingError = self.freq_bpm[red_fft_index + 1] - self.freq_bpm[red_fft_index - 1]
    #     self.GreenFreqencySamplingError = self.freq_bpm[green_fft_index + 1] - self.freq_bpm[green_fft_index - 1]
    #     self.BlueFreqencySamplingError = self.freq_bpm[blue_fft_index + 1] - self.freq_bpm[blue_fft_index - 1]

    def compressChannels(self, IR_fft, Gy_fft, R_fft, G_fft, B_fft):
        newitem = []
        for x in IR_fft:
            if (x <= 0):
                a = 0
            else:
                newitem.append(x)
        IR_fft = np.array(newitem)

        if (not self.ignoreGray):
            newitem = []
            for x in Gy_fft:
                if (x <= 0):
                    a = 0
                else:
                    newitem.append(x)
            Gy_fft = np.array(newitem)

        newitem = []
        for x in R_fft:
            if (x <= 0):
                a = 0
            else:
                newitem.append(x)
        R_fft = np.array(newitem)

        newitem = []
        for x in G_fft:
            if (x <= 0):
                a = 0
            else:
                newitem.append(x)
        G_fft = np.array(newitem)

        newitem = []
        for x in B_fft:
            if (x <= 0):
                a = 0
            else:
                newitem.append(x)
        B_fft = np.array(newitem)

        return B_fft.real, G_fft.real, R_fft.real, Gy_fft.real, IR_fft.real

    # def getResultHR(self, B_fft_Copy, G_fft_Copy, R_fft_Copy, Gy_fft_Copy, IR_fft_Copy):
    #     # apply ramp filter and find index of maximum frequency (after filter is applied).
    #     ir_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
    #     ir_fft_index = -1
    #     ir_fft_realabs = [0] * (int(self.ignore_freq_index_above) - int(self.ignore_freq_index_below))
    #
    #     grey_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
    #     grey_fft_index = -1
    #     grey_fft_realabs = [0] * (int(self.ignore_freq_index_above) - int(self.ignore_freq_index_below))
    #
    #     red_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
    #     red_fft_index = -1
    #     red_fft_realabs = [0] * (int(self.ignore_freq_index_above) - int(self.ignore_freq_index_below))
    #
    #     green_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
    #     green_fft_index = -1
    #     green_fft_realabs = [0] * (int(self.ignore_freq_index_above) - int(self.ignore_freq_index_below))
    #
    #     blue_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
    #     blue_fft_index = -1
    #     blue_fft_realabs = [0] * (int(self.ignore_freq_index_above) - int(self.ignore_freq_index_below))
    #
    #     realabs_i = 0
    #
    #     for x in range((int(self.ignore_freq_index_below + 1)), (int(self.ignore_freq_index_above + 1))):
    #         # "apply" the ramp to generate the shaped frequency values for IR
    #         # find the max value and the index of the max value.
    #         if(x < len(IR_fft_Copy.real)):
    #             current_irNum = self.ramp_design[realabs_i] * np.abs(IR_fft_Copy[x].real)
    #             ir_fft_realabs[realabs_i] = current_irNum
    #             if ((ir_fft_maxVal is None) or (current_irNum > ir_fft_maxVal)):
    #                 ir_fft_maxVal = current_irNum
    #                 ir_fft_index = x
    #
    #             if (not self.ignoreGray):
    #                 # "apply" the ramp to generate the shaped frequency values for Grey
    #                 current_greyNum = self.ramp_design[realabs_i] * np.abs(Gy_fft_Copy[x].real)
    #                 grey_fft_realabs[realabs_i] = current_greyNum
    #                 if ((grey_fft_maxVal is None) or (current_greyNum > grey_fft_maxVal)):
    #                     grey_fft_maxVal = current_greyNum
    #                     grey_fft_index = x
    #
    #             # "apply" the ramp to generate the shaped frequency values for Red
    #             current_redNum = self.ramp_design[realabs_i] * np.abs(R_fft_Copy[x].real)
    #             red_fft_realabs[realabs_i] = current_redNum
    #             if ((red_fft_maxVal is None) or (current_redNum > red_fft_maxVal)):
    #                 red_fft_maxVal = current_redNum
    #                 red_fft_index = x
    #
    #             # "apply" the ramp to generate the shaped frequency values for Green
    #             current_greenNum = self.ramp_design[realabs_i] * np.abs(G_fft_Copy[x].real)
    #             green_fft_realabs[realabs_i] = current_greenNum
    #             if ((green_fft_maxVal is None) or (current_greenNum > green_fft_maxVal)):
    #                 green_fft_maxVal = current_greenNum
    #                 green_fft_index = x
    #
    #             # "apply" the ramp to generate the shaped frequency values for blue
    #             current_blueNum = self.ramp_design[realabs_i] * np.abs(B_fft_Copy[x].real)
    #             blue_fft_realabs[realabs_i] = current_blueNum
    #             if ((blue_fft_maxVal is None) or (current_blueNum > blue_fft_maxVal)):
    #                 blue_fft_maxVal = current_blueNum
    #                 blue_fft_index = x
    #
    #             realabs_i = realabs_i + 1
    #
    #     # if(ir_fft_realabs.__contains__(None)):
    #     #     count =0
    #     #     for i in ir_fft_realabs:
    #     #         if (i is None):
    #     #             ir_fft_realabs[count] = 0
    #     #         count = count+1
    #     if(not len(self.frequency) == len(ir_fft_realabs)):
    #         self.frequency = self.objAlgorithm.get_Freq(len(ir_fft_realabs))
    #         self.freq_bpm = self.frequency *60
    #
    #     return blue_fft_realabs, blue_fft_index, blue_fft_maxVal, \
    #            green_fft_realabs, green_fft_index, green_fft_maxVal, \
    #            red_fft_realabs, red_fft_index, red_fft_maxVal, \
    #            grey_fft_realabs, grey_fft_index, grey_fft_maxVal, \
    #            ir_fft_realabs, ir_fft_index, ir_fft_maxVal

    # def find_heart_rate(self, fftarray):
    #     freqs = self.frequency
    #     freq_min = self.ignore_freq_below
    #     freq_max = self.ignore_freq_above
    #     fft_maximums = []
    #
    #     for i in range(fftarray.shape[0]):
    #         if freq_min <= freqs[i] <= freq_max:
    #             fftMap = abs(fftarray[i].real)
    #             fft_maximums.append(fftMap.max())
    #         else:
    #             fft_maximums.append(0)
    #
    #     peaks, properties = signal.find_peaks(fft_maximums)
    #     max_peak = -1
    #     max_freq = 0

    ## Find frequency with max amplitude in peaks
    # for peak in peaks:
    #     if fft_maximums[peak] > max_freq:
    #         max_freq = fft_maximums[peak]
    #         max_peak = peak
    #
    # return freqs[max_peak], max_peak

    # def find_heart_rate_byIndex(self,ir_fft_index,grey_fft_index,red_fft_index,green_fft_index,blue_fft_index):
    #     self.IrBpm = self.frequency[ir_fft_index]* 60
    #     if (not self.ignoreGray):
    #         self.GreyBpm = self.frequency[grey_fft_index] * 60
    #     else:
    #         self.GreyBpm = 0
    #     self.RedBpm = self.frequency[red_fft_index]* 60
    #     self.GreenBpm = self.frequency[green_fft_index]* 60
    #     self.BlueBpm = self.frequency[blue_fft_index]* 60
    #
    # def getChannelIndex(self, fft_realabs, frequency):
    #     fft_index = fft_realabs[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
    #     fft_maxVal = frequency[fft_index]  # Get the actual frequency value
    #     return fft_index,fft_maxVal

    # def getBPMbyfrequencyArray(self,ir_fft_realabs,grey_fft_realabs,red_fft_realabs,green_fft_realabs,blue_fft_realabs):
    #     # ind0 = np.abs(self.frequency) <= self.ignore_freq_above
    #     # a = self.frequency * 60
    #     # b = self.freq_bpm
    #     # ind1 = np.abs(a) <= self.ignore_freq_above_bpm
    #     # ind2 = np.abs(b) <= self.ignore_freq_above_bpm
    #     # index = ir_fft_realabs[np.abs(self.frequency) <= self.ignore_freq_above].argmax()
    #     # index = ir_fft_realabs[np.abs(self.freq_bpm) <= self.ignore_freq_above_bpm].argmax()
    #     index = ir_fft_realabs[np.abs(self.frequency) <= self.ignore_freq_above].argmax()
    #     self.IrBpm = np.abs(self.frequency)[index] * 60
    #     if (not self.ignoreGray):
    #         self.GreyBpm = np.abs(self.frequency)[
    #                            grey_fft_realabs[np.abs(self.frequency) <= self.ignore_freq_above].argmax()] * 60
    #     else:
    #         self.GreyBpm = 0
    #     self.RedBpm = np.abs(self.frequency)[red_fft_realabs[np.abs(self.frequency) <= self.ignore_freq_above].argmax()] * 60
    #     self.GreenBpm = np.abs(self.frequency)[green_fft_realabs[np.abs(self.frequency) <= self.ignore_freq_above].argmax()] * 60
    #     self.BlueBpm = np.abs(self.frequency)[blue_fft_realabs[np.abs(self.frequency) <= self.ignore_freq_above].argmax()] * 60

    # def getBPMbyfrqbpmArray(self,ir_fft_index,grey_fft_index,red_fft_index,green_fft_index,blue_fft_index):
    #
    #     self.IrBpm = self.freq_bpm[ir_fft_index]
    #     if (not self.ignoreGray):
    #         self.GreyBpm = self.freq_bpm[grey_fft_index]
    #     else:
    #         self.GreyBpm = 0
    #     self.RedBpm = self.freq_bpm[red_fft_index]
    #     self.GreenBpm = self.freq_bpm[green_fft_index]
    #     self.BlueBpm = self.freq_bpm[blue_fft_index]

    def CopyCalculatedata_fromHRComputerobject(self, IrSnr, GreySnr, RedSnr, GreenSnr, BlueSnr,
                                               IrBpm, GreyBpm, RedBpm, GreenBpm, BlueBpm, IrFreqencySamplingError,
                                               GreyFreqencySamplingError, RedFreqencySamplingError,
                                               GreenFreqencySamplingError, BlueFreqencySamplingError, SNR):
        # ResultData
        self.IrSnr = IrSnr
        self.GreySnr = GreySnr
        self.RedSnr = RedSnr
        self.GreenSnr = GreenSnr
        self.BlueSnr = BlueSnr
        self.IrBpm = IrBpm
        self.GreyBpm = GreyBpm
        self.RedBpm = RedBpm
        self.GreenBpm = GreenBpm
        self.BlueBpm = BlueBpm
        self.IrFreqencySamplingError = IrFreqencySamplingError
        self.GreyFreqencySamplingError = GreyFreqencySamplingError
        self.RedFreqencySamplingError = RedFreqencySamplingError
        self.GreenFreqencySamplingError = GreenFreqencySamplingError
        self.BlueFreqencySamplingError = BlueFreqencySamplingError
        self.SNRSummary = SNR

    # region calculate result (HR in bpm) using frequency
    def generateHeartRateandSNR(self, B_filtered, G_filtered, R_filtered, Gy_filtered, IR_filtered, type):
        # Create copy of channels
        ir_fft_realabs = IR_filtered.copy()
        if (not self.ignoreGray):
            grey_fft_realabs = Gy_filtered.copy()
        red_fft_realabs = R_filtered.copy()
        green_fft_realabs = G_filtered.copy()
        blue_fft_realabs = B_filtered.copy()

        # Calculate samples
        ColorNumSamples = len(grey_fft_realabs)
        IRNumSamples = len(ir_fft_realabs)
        self.freq_bpmColor = 60 * self.Colorfrequency
        self.freq_bpmIr = 60 * self.IRfrequency

        # Compute heart rate and snr
        objComputerHeartRate = ComputerHeartRate(self.snrType, self.ignoreGray, ColorNumSamples, self.grayIndex,
                                                 self.IRIndex, self.components,
                                                 self.ColorEstimatedFPS, self.IREstimatedFPS, self.ramp_end_bpm,
                                                 self.ramp_start_percentage, self.ramp_end_percentage,
                                                 self.ignore_freq_below_bpm, self.ignore_freq_above_bpm,
                                                 self.freq_bpmColor, self.freq_bpmIr, self.Colorfrequency,
                                                 self.IRfrequency, self.SavePath, self.region, IRNumSamples)

        SNR = ''
        if (type == 1):
            SNR = objComputerHeartRate.OriginalARPOSmethod(blue_fft_realabs, green_fft_realabs, red_fft_realabs,
                                                           grey_fft_realabs, ir_fft_realabs)
        elif (type == 2):
            SNR = objComputerHeartRate.getHeartRate_fromFrequency(blue_fft_realabs, green_fft_realabs, red_fft_realabs,
                                                                  grey_fft_realabs, ir_fft_realabs)
        elif (type == 3):
            SNR = objComputerHeartRate.getHearRate_fromFrequencyWithFilter_Main(blue_fft_realabs, green_fft_realabs,
                                                                                red_fft_realabs, grey_fft_realabs,
                                                                                ir_fft_realabs)

        ##Copy obj compute HR data to local class
        self.CopyCalculatedata_fromHRComputerobject(objComputerHeartRate.IrSnr, objComputerHeartRate.GreySnr,
                                                    objComputerHeartRate.RedSnr,
                                                    objComputerHeartRate.GreenSnr, objComputerHeartRate.BlueSnr,
                                                    objComputerHeartRate.IrBpm,
                                                    objComputerHeartRate.GreyBpm, objComputerHeartRate.RedBpm,
                                                    objComputerHeartRate.GreenBpm, objComputerHeartRate.BlueBpm,
                                                    objComputerHeartRate.IrFreqencySamplingError,
                                                    objComputerHeartRate.GreyFreqencySamplingError,
                                                    objComputerHeartRate.RedFreqencySamplingError,
                                                    objComputerHeartRate.GreenFreqencySamplingError,
                                                    objComputerHeartRate.BlueFreqencySamplingError, SNR)

        ##delete obj compute hr data
        del objComputerHeartRate

        ##Copy for storage
        # ResultData


class ProcessedData:
    ColorEstimatedFPS = None
    IREstimatedFPS = None
    ColorFPSwithTime = []
    IRFPSwithTime = []
    startTime = None
    endTime = None
    diffTime = None
    Window_count = None
    Colorfrequency = None
    IRfrequency = None
    B_signal = None
    G_signal = None
    R_signal = None
    Gy_signal = None
    IR_signal = None
