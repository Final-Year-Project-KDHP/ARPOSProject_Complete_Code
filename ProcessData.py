import os
import numpy as np
from scipy import signal
from sklearn import preprocessing
from Algorithm import AlgorithmCollection
from SaveGraphs import Plots
import sys
from WindowData import Window_Data


class ProcessFaceData:
    # Hold Current Window Region Data
    regionStore = []
    regionSignalData = []
    distanceM = []
    time_list_color = []
    timecolorCount = []
    time_list_ir = []
    timeirCount = []
    Frametime_list_ir = []
    Frametime_list_color = []
    region = ''

    # For window size
    Window_count = 0
    regionWindowSignalData = []
    WindowdistanceM = []
    Windowtime_list_color = []
    WindowtimecolorCount = []
    Windowtime_list_ir = []
    WindowtimeirCount = []
    WindowFrametime_list_ir = []
    WindowFrametime_list_color = []
    timeinSeconds = 0

    # Constants
    EstimatedFPS = 30

    # Input Parameters
    Algorithm_type = ''
    FFT_type = ''
    Filter_type = 0
    Result_type = 0
    Preprocess_type = 0
    SavePath = ''
    ignoreGray = False
    isSmoothen = False
    GenerateGraphs = False

    objPlots = Plots()
    objAlgorithm = AlgorithmCollection()

    # constructor
    def __init__(self, Algorithm_type, FFT_type, Filter_type, Result_type, Preprocess_type, SavePath, ignoreGray,
                 isSmoothen, GenerateGraphs):
        self.Algorithm_type = Algorithm_type
        self.FFT_type = FFT_type
        self.Filter_type = Filter_type
        self.Result_type = Result_type
        self.Preprocess_type = Preprocess_type
        self.SavePath = SavePath
        self.ignoreGray = ignoreGray
        self.isSmoothen = isSmoothen
        self.GenerateGraphs = GenerateGraphs

        #set estimated fps
        self.objAlgorithm.EstimatedFPS = self.EstimatedFPS  # TODO FIX FOR VAIOIURS FPS

    '''
    setSignalSourceData: 
    A method that takes ROIStore data and region name such as lips, forehead etc
    and stores specific region data to local object to process it
    '''
    def setSignalSourceData(self, ROIStore, region):
        # Split ROI Store region data
        # where region can be lips, forehead etc
        self.regionStore = ROIStore.get(region)
        self.regionSignalData = self.regionStore.getAllData()
        self.distanceM = self.regionStore.distanceM
        self.time_list_color = self.regionStore.time_list_color
        self.timecolorCount = self.regionStore.timecolorCount
        self.time_list_ir = self.regionStore.time_list_ir
        self.timeirCount = self.regionStore.timeirCount
        self.Frametime_list_ir = self.regionStore.Frametime_list_ir
        self.Frametime_list_color = self.regionStore.Frametime_list_color
        self.region = region

    '''
    getSingalData: 
    This method records original data by calling previously defined method 'setSignalSourceData(ROIStore, region)'
    and splits data to window size to process it
    '''
    def getSingalData(self, ROIStore, region, WindowSlider, step, WindowCount):
        # Split ROI Store region data
        if (WindowCount == 0):
            self.setSignalSourceData(ROIStore, region)
        else:
            OrignialBlue = self.regionSignalData[:, 0]
            OrignialGreen = self.regionSignalData[:, 1]
            OrignialRed = self.regionSignalData[:, 2]
            OrignialGrey = self.regionSignalData[:, 3]
            OrignialIr = self.regionSignalData[:, 4]

            OrignialBlue = OrignialBlue[step:]  # from steps till end and disacard rest
            OrignialGreen = OrignialGreen[step:]
            OrignialRed = OrignialRed[step:]
            OrignialGrey = OrignialGrey[step:]
            OrignialIr = OrignialIr[step:]
            self.regionSignalData = np.c_[OrignialBlue, OrignialGreen, OrignialRed, OrignialGrey, OrignialIr]
            self.distanceM = self.distanceM[step:]
            self.time_list_color = self.time_list_color[step:]
            self.timecolorCount = self.timecolorCount[step:]
            self.time_list_ir = self.time_list_ir[step:]
            self.timeirCount = self.timeirCount[step:]
            self.Frametime_list_ir = self.Frametime_list_ir[step:]
            self.Frametime_list_color = self.Frametime_list_color[step:]

        # split to window size
        self.Window_count = WindowCount
        self.regionWindowSignalData = self.regionSignalData[:WindowSlider]
        self.WindowdistanceM = self.distanceM[:WindowSlider]
        self.Windowtime_list_color = self.time_list_color[:WindowSlider]
        self.WindowtimecolorCount = self.timecolorCount[:WindowSlider]
        self.Windowtime_list_ir = self.time_list_ir[:WindowSlider]
        self.WindowtimeirCount = self.timeirCount[:WindowSlider]
        self.WindowFrametime_list_ir = self.Frametime_list_ir[:WindowSlider]
        self.WindowFrametime_list_color = self.Frametime_list_color[:WindowSlider]

    '''
    getGraphPath: Returns graph path where images of graphs can be strored and viewed
    '''
    def getGraphPath(self):
        return self.SavePath + "Graphs\\"
    '''
    defineGraphName: Returns image name of a graph, adds various types of techniques and filters for identification purpose
    '''
    def defineGraphName(self, graphName):
        imageName = graphName + "_" + self.region + "_W" + str(
            self.Window_count) + "_" + self.Algorithm_type + "_FFT-" + str(self.FFT_type) + "_FL-" + str(
            self.Filter_type) + "_RS-" + str(self.Result_type) + "_PR-" + str(self.Preprocess_type) + "_SM-" + str(
            self.isSmoothen)
        return imageName

    '''
    ApplyAlgorithm: Applies algorithms on signal data
    '''
    def ApplyAlgorithm(self, S, components):
        #Apply by Algorithm_type
        if (self.Algorithm_type == "FastICA"):
            S_ = self.objAlgorithm.ApplyICA(S, components)
        elif (self.Algorithm_type == "PCA"):
            S_ = self.objAlgorithm.ApplyPCA(S, components)
        elif (self.Algorithm_type == "PCAICA"):
            S = self.objAlgorithm.ApplyPCA(S, components)
            S_ = self.objAlgorithm.ApplyICA(S, components)
        elif (self.Algorithm_type == "Jade"):
            S_ = self.objAlgorithm.jadeR5(S, components)  # r4 is slwoer and f5 is faster
            # https://github.com/kellman/heartrate_matlab/blob/master/jadeR.m
            # Blue
            newBlue = np.array(S_[0])
            newBlue2 = newBlue[0].real
            # Green
            newGreen = np.array(S_[1])
            newGreen2 = newGreen[0].real
            # Red
            newRed = np.array(S_[2])
            newRed2 = newRed[0].real

            if (components == 5):
                # Grey
                newGrey = np.array(S_[3])
                newGrey2 = newGrey[0].real
                # IR
                newIr = np.array(S_[4])
                newir2 = newIr[0].real
                #Compile all channel data
                S_ = np.c_[newBlue2, newGreen2, newRed2, newGrey2, newir2]
            else:
                # IR
                newIr = np.array(S_[3])
                newir2 = newIr[0].real
                #Compile all channel data
                S_ = np.c_[newBlue2, newGreen2, newRed2, newir2]
        else:
            S_ = S

        if (self.GenerateGraphs):
            self.GenerateGrapth("Algorithm",S_)

        return S_

    def FilterTechniques(self, type, IR_fft, Gy_fft, R_fft, G_fft, B_fft, frequency, ignoregray):
        # cuttoff,  butterworth and other

        if (ignoregray):
            Gy_bp = []

            # freq_B = []
            # freq_G = []
            # freq_R= []
            # freq_Gy= []
            # freq_Ir = []

        if (type == 1):  # Not very good with rampstuff method, very high heart rate
            B_bp = self.Filterffq(B_fft, frequency)
            G_bp = self.Filterffq(G_fft, frequency)
            R_bp = self.Filterffq(R_fft, frequency)
            if (not ignoregray):
                Gy_bp = self.Filterffq(Gy_fft, frequency)
            IR_bp = self.Filterffq(IR_fft, frequency)

        elif (type == 2):  #

            B_bp = self.Filterffq2(B_fft, frequency)
            G_bp = self.Filterffq2(G_fft, frequency)
            R_bp = self.Filterffq2(R_fft, frequency)
            if (not ignoregray):
                Gy_bp = self.Filterffq2(Gy_fft, frequency)
            IR_bp = self.Filterffq2(IR_fft, frequency)

        elif (type == 3):
            B_bp = self.fft_filter(frequency, B_fft)
            G_bp = self.fft_filter(frequency, G_fft)
            R_bp = self.fft_filter(frequency, R_fft)
            if (not ignoregray):
                Gy_bp = self.fft_filter(frequency, Gy_fft)
            IR_bp = self.fft_filter(frequency, IR_fft)

        elif (type == 4):  # Not very good with rampstuff method
            B_bp, G_bp, R_bp, Gy_bp, IR_bp = self.ApplyFilterwithParamReturn(B_fft, G_fft, R_fft, Gy_fft, IR_fft, 6,
                                                                             ignoregray)  # self.blue, self.green, self.red, self.grey, self.Irchannel)

        elif (type == 5):  # Not very good with rampstuff method
            # Do no filter (use it when calculating hearrate)
            B_bp = B_fft
            G_bp = G_fft
            R_bp = R_fft
            Gy_bp = Gy_fft
            IR_bp = IR_fft

        elif (type == 6):

            B_bp = self.filterTechq2(frequency, B_fft)
            G_bp = self.filterTechq2(frequency, G_fft)
            R_bp = self.filterTechq2(frequency, R_fft)
            if (not ignoregray):
                Gy_bp = self.filterTechq2(frequency, Gy_fft)
            IR_bp = self.filterTechq2(frequency, IR_fft)

        elif (type == 7):

            B_bp = self.filterTechq3(frequency, B_fft)
            G_bp = self.filterTechq3(frequency, G_fft)
            R_bp = self.filterTechq3(frequency, R_fft)
            if (not ignoregray):
                Gy_bp = self.filterTechq3(frequency, Gy_fft)
            IR_bp = self.filterTechq3(frequency, IR_fft)

        # BElow produce same results
        # elif (type == 4):  #
        #     B_bp, G_bp, R_bp, Gy_bp, IR_bp = self.ApplyFilterwithParamReturn(B_fft, G_fft, R_fft, Gy_fft, IR_fft, 3)
        #
        # elif (type == 5):
        #     B_bp, G_bp, R_bp, Gy_bp, IR_bp = self.ApplyFilterwithParamReturn(B_fft, G_fft, R_fft, Gy_fft, IR_fft, 4)
        #
        # elif (type == 6):
        #     B_bp, G_bp, R_bp, Gy_bp, IR_bp = self.ApplyFilterwithParamReturn(B_fft, G_fft, R_fft, Gy_fft, IR_fft, 5)
        #
        # elif (type == 7):
        #     B_bp, G_bp, R_bp, Gy_bp, IR_bp = self.ApplyFilterwithParamReturn(B_fft, G_fft, R_fft, Gy_fft, IR_fft, 6)
        #
        # elif (type == 8):
        #     B_bp, G_bp, R_bp, Gy_bp, IR_bp = self.ApplyFilterwithParamReturn(B_fft, G_fft, R_fft, Gy_fft, IR_fft, 7)
        #
        # elif (type == 9):
        #     B_bp, G_bp, R_bp, Gy_bp, IR_bp = self.ApplyFilterwithParamReturn(B_fft, G_fft, R_fft, Gy_fft, IR_fft, 8)
        #
        # elif (type == 10):
        #     B_bp, G_bp, R_bp, Gy_bp, IR_bp = self.ApplyFilterwithParamReturn(B_fft, G_fft, R_fft, Gy_fft, IR_fft, 9)

        return B_bp, G_bp, R_bp, Gy_bp, IR_bp

    def CalcualateSPOWithout(self, Gy_bp, heartRatePeriod, IR_bp, R_bp, G_bp, distanceM, smallestOxygenError, region):

        self.OxygenSaturation = 0.0
        self.OxygenSaturationError = 0.0

        freqs = np.fft.fftfreq(len(Gy_bp), 1 / 30)
        greyPeeks, properties, filteredgrey = self.FindGreyPeak(Gy_bp, freqs, self.ignore_freq_below,
                                                                self.ignore_freq_above)
        # find the max peeks in each sample window
        grayMaxPeekIndice = []

        formid = len(filteredgrey) - int(heartRatePeriod)  # grey channel orinigal before
        for x in range(0, formid, int(heartRatePeriod)):
            maxGreyPeekInPeriod = self.FindMaxPeekInPeriod(greyPeeks, filteredgrey, int(heartRatePeriod), x)
            grayMaxPeekIndice.append((maxGreyPeekInPeriod))

        # get the values of the ir and red channels at the grey peeks
        oxygenLevels = []
        for index in grayMaxPeekIndice:
            # if self.ignore_freq_below <= freqs[index] <= self.ignore_freq_above:
            irValue = np.abs(IR_bp[index].real)  # ir cahnnel original before
            redValue = R_bp[index].real  # red cahnnel original before
            greenValue = G_bp[index].real  # green cahnnel original before
            if (irValue > 0):
                # irValue = Infra_fftMaxVal
                # redValue = red_fftMaxVal  # red cahnnel original before
                # greenValue = gy_fftMaxVal  # green cahnnel original before

                distValue = distanceM[index] - 1  # 0.7 #self.distanceMm[index]
                # distValue = distValue*1000 #convert to mm
                # Absorption values calculated of oxy and deoxy heamoglobin
                # where red defined as: 600-700nm
                # where ir defined as: 858-860nm.
                red_deoxy_mean = 4820.4361
                red_oxy_mean = 667.302
                ir_deoxy_mean = 694.32  # 693.38
                ir_oxy_mean = 1092  # 1087.2

                # Depth-resultion blood oxygen satyuration measuremnet by dual-wavelength photothermal (DWP) optical coherence tomography
                # Biomed Opt Express
                # 2011 P491-504

                irToRedRatio = (red_oxy_mean / ir_oxy_mean) * ((irValue * ((
                        distValue / 100000)) / redValue)) / 52  # /1000 at a distance of one meter using 99% white reflections standard, 52 is the ir/r scaling factor.
                oxygenLevel = 100 * (red_deoxy_mean - (irToRedRatio * ir_deoxy_mean)) / (
                        ir_oxy_mean + red_deoxy_mean - ir_deoxy_mean - red_oxy_mean) - 6  # -2 only for pca
                oxygenLevels.append(oxygenLevel)

                oxygenLevels.append(round(oxygenLevel))

        # compute SD and mean of oxygenLevels
        self.OxygenSaturation = np.std(oxygenLevels)
        self.OxygenSaturationError = np.std(oxygenLevels, ddof=1) / np.sqrt(
            np.size(oxygenLevels))  # MeanStandardDeviation err

        if (self.OxygenSaturationError < smallestOxygenError):
            self.smallestOxygenError = self.OxygenSaturationError
            self.regionToUse = region

            # if (self.OxygenSaturation > self.oxygenstd):
            #     oxygenstd = self.OxygenSaturation
            #     print("STD : " + str(self.OxygenSaturation) + " , error: " + str(self.OxygenSaturationError))
            #     print("SPO 0 : " + str(oxygenLevels[0]))

        if (len(oxygenLevels) <= 0):
            oxygenLevels.append(0)
        return self.OxygenSaturation, self.OxygenSaturationError, str(oxygenLevels[0])

    def preprocessSignalData(self):

        # Add interpolation and smoothen of curve
        if (self.Preprocess_type == 1):
            # do nothing
            a = 0
        elif (self.Preprocess_type == 2):
            self.blue = np.array(self.blue)
            Larray = len(self.blue)
            processed = signal.detrend(self.blue)  # detrend the signal to avoid interference of light change
            even_times = np.linspace(self.timecolorCount[0], self.timecolorCount[-1], Larray) # TODO: Fix time color count? instead of maual use other automatic method?
            interpolated = np.interp(even_times, self.timecolorCount, processed)  # interpolation by 1
            interpolated = np.hamming(
                Larray) * interpolated  # make the signal become more periodic (advoid spectral leakage)
            self.blue = interpolated / np.linalg.norm(interpolated)

            self.green = np.array(self.green)
            Larray = len(self.green)
            processed = signal.detrend(self.green)  # detrend the signal to avoid interference of light change
            even_times = np.linspace(self.timecolorCount[0], self.timecolorCount[-1], Larray)
            interpolated = np.interp(even_times, self.timecolorCount, processed)  # interpolation by 1
            interpolated = np.hamming(
                Larray) * interpolated  # make the signal become more periodic (advoid spectral leakage)
            self.green = interpolated / np.linalg.norm(interpolated)

            self.red = np.array(self.red)
            Larray = len(self.red)
            processed = signal.detrend(self.red)  # detrend the signal to avoid interference of light change
            even_times = np.linspace(self.timecolorCount[0], self.timecolorCount[-1], Larray)
            interpolated = np.interp(even_times, self.timecolorCount, processed)  # interpolation by 1
            interpolated = np.hamming(
                Larray) * interpolated  # make the signal become more periodic (advoid spectral leakage)
            self.red = interpolated / np.linalg.norm(interpolated)

            if (not self.ignoreGray):
                self.grey = np.array(self.grey)
                Larray = len(self.grey)
                processed = signal.detrend(self.grey)  # detrend the signal to avoid interference of light change
                even_times = np.linspace(self.timecolorCount[0], self.timecolorCount[-1], Larray)
                interpolated = np.interp(even_times, self.timecolorCount, processed)  # interpolation by 1
                interpolated = np.hamming(
                    Larray) * interpolated  # make the signal become more periodic (advoid spectral leakage)
                self.grey = interpolated / np.linalg.norm(interpolated)

            self.Irchannel = np.array(self.Irchannel)
            Larray = len(self.Irchannel)
            processed = signal.detrend(self.Irchannel)  # detrend the signal to avoid interference of light change
            even_times = np.linspace(self.timeirCount[0], self.timeirCount[-1], Larray)
            interpolated = np.interp(even_times, self.timeirCount, processed)  # interpolation by 1
            interpolated = np.hamming(
                Larray) * interpolated  # make the signal become more periodic (advoid spectral leakage)
            self.Irchannel = interpolated / np.linalg.norm(interpolated)

        elif (self.Preprocess_type == 3):
            self.blue = self.preprocessdata(np.array(self.blue), self.timecolorCount, True)
            self.green = self.preprocessdata(np.array(self.green), self.timecolorCount, True)
            self.red = self.preprocessdata(np.array(self.red), self.timecolorCount, True)
            if (not self.ignoreGray):
                self.grey = self.preprocessdata(np.array(self.grey), self.timecolorCount, True)
            self.Irchannel = self.preprocessdata(np.array(self.Irchannel), self.timeirCount, True)

        elif (self.Preprocess_type == 4):  ##set fft and freq data accordingly
            self.blue = self.preprocessdata2(np.array(self.blue), self.timecolorCount, True)
            self.green = self.preprocessdata2(np.array(self.green), self.timecolorCount, True)
            self.red = self.preprocessdata2(np.array(self.red), self.self.timecolorCount, True)
            if (not self.ignoreGray):
                self.grey = self.preprocessdata2(np.array(self.grey), self.timecolorCount, True)
            self.Irchannel = self.preprocessdata2(np.array(self.Irchannel), self.timeirCount, True)

        elif (self.Preprocess_type == 5):

            if (self.ignoreGray):
                S = np.c_[self.blue, self.green, self.red, self.Irchannel]
            else:
                S = np.c_[self.blue, self.green, self.red, self.grey, self.Irchannel]
            S = preprocessing.normalize(S)

            self.blue = S[:, 0]
            self.green = S[:, 1]
            self.red = S[:, 2]
            index = 3
            if (not self.ignoreGray):
                self.grey = S[:, 3]
                index = 4
            self.Irchannel = S[:, index]

        elif (self.Preprocess_type == 6):
            self.blue = self.preprocessdata(np.array(self.blue), self.timecolorCount, False)
            self.green = self.preprocessdata(np.array(self.green), self.timecolorCount, False)
            self.red = self.preprocessdata(np.array(self.red), self.timecolorCount, False)
            if (not self.ignoreGray):
                self.grey = self.preprocessdata(np.array(self.grey), self.timecolorCount, False)
            self.Irchannel = self.preprocessdata(np.array(self.Irchannel), self.timeirCount, False)

        elif (self.Preprocess_type == 7):  ##set fft and freq data accordingly
            self.blue = self.preprocessdata2(np.array(self.blue), self.timecolorCount, False)
            self.green = self.preprocessdata2(np.array(self.green), self.timecolorCount, False)
            self.red = self.preprocessdata2(np.array(self.red), self.timecolorCount, False)
            if (not self.ignoreGray):
                self.grey = self.preprocessdata2(np.array(self.grey), self.timecolorCount, False)
            self.Irchannel = self.preprocessdata2(np.array(self.Irchannel), self.timeirCount, False)


        #Combine r,g,b,gy,ir in one array
        S = self.getSignalDataCombined()

        #generate PreProcessed plot
        if (self.GenerateGraphs):
            self.GenerateGrapth("PreProcessed",S)

    def GenerateGrapth(self,graphName,S):
        if(graphName == "RawData"):
            imageName = self.defineGraphName(graphName)
            self.objPlots.plotGraphAllWithoutTimewithParam(self.getGraphPath(), imageName, self.blue, self.green, self.red, self.grey, self.Irchannel, "No of Frames", "Intensity")
            imageName = self.defineGraphName(graphName +"All")
            self.objPlots.plotAllinOneWithoutTime(self.getGraphPath(), imageName, self.blue, self.green, self.red, self.grey, self.Irchannel, "No of Frames", "Intensity")

        elif (graphName == "PreProcessed"):
            imageName = self.defineGraphName(graphName+"All")
            self.objPlots.plotAllinOne(self.getGraphPath(), S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4], imageName, 30, self.timeinSeconds, "Time(s)", "Amplitude")
            imageName = self.defineGraphName(graphName+"TimeAll")
            self.objPlots.plotGraphAllwithParam(self.getGraphPath(), imageName, self.time_list_color,self.time_list_ir, S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4], "Time(s)",
                                           "Amplitude", 30, self.timeinSeconds)
        elif(graphName == "Algorithm"):
            imageName = self.defineGraphName(graphName + "Time")
            self.objPlots.plotGraphAllwithParam(self.getGraphPath(), imageName, self.time_list_color,
                                           self.time_list_ir,
                                           S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4], "Time(s)",
                                           "Amplitude", 30, self.timeinSeconds)
            imageName = self.defineGraphName(graphName+"TimeAll")
            self.objPlots.plotAllinOne(self.getGraphPath(), S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4], imageName, 30,
                                  self.timeinSeconds, "Time(s)",
                                  "Amplitude")
        elif(graphName == "Smooth"):
            imageName = self.defineGraphName(graphName+"TimeAll")
            self.objPlots.plotAllinOne(self.getGraphPath(),
                                  S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4], imageName, 30, self.timeinSeconds,
                                  "Time(s)",
                                  "Amplitude")
            imageName = self.defineGraphName(graphName+"Time")
            self.objPlots.plotGraphAllwithParam(self.getGraphPath(), imageName, self.time_list_color,
                                           self.time_list_ir,
                                           S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4], "Time(s)",
                                           "Amplitude", 30, self.timeinSeconds)
    def getSignalDataCombined(self):
        S=[]

        if (self.ignoreGray):
            S = np.c_[self.blue, self.green, self.red, self.Irchannel]
        else:
            S = np.c_[self.blue, self.green, self.red, self.grey, self.Irchannel]

        return S

    def SmoothenData(self,S_,components):
        self.blue = self.smooth(S_[:, 0])
        self.green = self.smooth(S_[:, 1])
        self.red = self.smooth(S_[:, 2])
        if (not self.ignoreGray):
            self.grey = self.smooth(S_[:, components - 2])
        self.Irchannel = self.smooth(S_[:, components - 1])

        S_ = self.getSignalDataCombined()

        if (self.GenerateGraphs):
            self.GenerateGrapth("Smooth", S_)

        return S_

    def Process_EntireSignalData(self):  # TODO : Implement without Gray
        components = 5
        if (self.ignoreGray):
            components = 4

        self.blue = self.regionWindowSignalData[:, 0]
        self.green = self.regionWindowSignalData[:, 1]
        self.red = self.regionWindowSignalData[:, 2]
        self.grey = self.regionWindowSignalData[:, 3]
        self.Irchannel = self.regionWindowSignalData[:, 4]

        #generate raw data plot
        if (self.GenerateGraphs):
            self.GenerateGrapth("RawData",None)

        #PreProcess Signal
        S = self.preprocessSignalData()

        # Apply Algorithm
        S_ = self.ApplyAlgorithm(S, components)

        # Apply smoothen only before fft
        if (self.isSmoothen):
            #Smooth data
            self.SmoothenData(S_,components)

        # Apply fft
        B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, frequency, \
        blue_max_peak, green_max_peak, red_max_peak, grey_max_peak, ir_max_peak, \
        blue_bpm, green_bpm, red_bpm, grey_bpm, ir_bpm = self.ApplyFFT(self.FFT_type, S_, self.objAlgorithm, self.objPlots, self.ignoreGray)

        if (self.GenerateGraphs):
            imageName = self.defineGraphName("FFT")
            self.objPlots.PlotFFT(B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, frequency, imageName, self.getGraphPath(), imageName)

        B_bp, G_bp, R_bp, Gy_bp, IR_bp = self.FilterTechniques(self.Filter_type, IR_fft, Gy_fft, R_fft, Gr_fft, B_fft,
                                                               frequency, self.ignoreGray)  ##Applyfiltering

        if (self.Filter_type == 6):
            interest_idx = np.where((frequency >= self.ignore_freq_below) & (frequency <= self.ignore_freq_above))[0]
            interest_idx_sub = interest_idx[:-1].copy()  # advoid the indexing error
            frequency = frequency[interest_idx_sub]
        elif (self.Filter_type == 7):
            interest_idx = np.where((frequency >= self.ignore_freq_below))[0]
            interest_idx_sub = interest_idx[:-1].copy()  # advoid the indexing error
            frequency = frequency[interest_idx_sub]

        if (self.GenerateGraphs):
            imageName = self.defineGraphName("Filtered")
            self.objPlots.PlotFFT(B_bp, G_bp, R_bp, Gy_bp, IR_bp, frequency, imageName, self.getGraphPath(), imageName)

        NumSamples = len(frequency) # TODO: No of samples is differnet initailly meaning was set to len(self.channel)
        if (self.Result_type == 1):
            self.RampStuff(NumSamples, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency)
        elif (self.Result_type == 2):
            self.RampStuff6(NumSamples, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency)
        elif (self.Result_type == 3):
            self.Ramp2(NumSamples, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency)
        elif (self.Result_type == 4):
            self.Ramp3(NumSamples, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency)
        elif (self.Result_type == 5):
            self.Ramp4(NumSamples, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency)
        elif (self.Result_type == 6):
            self.Ramp5(NumSamples, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency)
        elif (self.Result_type == 7):
            self.Ramp7(NumSamples, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency)
        elif (self.Result_type == 8):
            self.Ramp8(NumSamples, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency)
        elif (self.Result_type == 9):
            self.Ramp9(NumSamples, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency)

        windowList = Window_Data()
        windowList.WindowNo = 1
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
        std, err, oxylevl = self.getSpo(G_bp)
        windowList.oxygenSaturationSTD = std  # std
        windowList.oxygenSaturationValueError = err  # err
        windowList.oxygenSaturationValueValue = oxylevl  # oxylevl

        return windowList

    def getSpo(self,G_bp):
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
        filteredir = self.Irchannel
        filteredred = self.red
        std, err, oxylevl = self.CalcualateSPOWithout(filteredgrey, self.heartRatePeriod, filteredir, filteredred,
                                                      self.green,
                                                      self.distanceM, smallestOxygenError,
                                                      self.region)  # CalcualateSPOPart2 ,CalcualateSPOWithout

        oxygenSaturationValueError = self.OxygenSaturationError
        oxygenSaturationValueValue = self.OxygenSaturation

        return std, err, oxylevl

    def RampStuff(self, NumSamples, fps, IR_fft, Gy_fft, R_fft, G_fft, B_fft, frequency,
                  ignoregray):  # affter fft use this
        IR_fft_Copy = IR_fft.copy()
        if (not ignoregray):
            Gy_fft_Copy = Gy_fft.copy()
        R_fft_Copy = R_fft.copy()
        G_fft_Copy = G_fft.copy()
        B_fft_Copy = B_fft.copy()

        ramp_end_bpm = 55
        ramp_start_percentage = 0.5
        ramp_end_percentage = 1
        ramp_end_hz = ramp_end_bpm / 60

        # freq_bpm = []  # size of NumSamples
        # for x in range(0, NumSamples):
        #     freq_bpm.append(((x * fps) / NumSamples) * 60)  # in beats per minute

        freq_bpm = 60 * frequency
        # unfiltered = R_fft.copy()

        # define index for the low and high pass frequency filter in frequency space that has just been created
        # (depends on number of samples).
        ignore_freq_index_below = np.rint(((self.ignore_freq_below * NumSamples) / fps))  # high pass
        ignore_freq_index_above = np.rint(((self.ignore_freq_above * NumSamples) / fps))  # low pass

        # compute the ramp filter start and end indices double
        ramp_start = ignore_freq_index_below
        ramp_end = np.rint(((ramp_end_hz * NumSamples) / fps))
        rampDesignLength = ignore_freq_index_above - ignore_freq_index_below
        ramp_design = [None] * int(rampDesignLength)

        ramplooprange = int(ramp_end - ramp_start)
        # setup linear ramp
        for x in range(0, ramplooprange):
            ramp_design[x] = ((((ramp_end_percentage - ramp_start_percentage) / (ramp_end - ramp_start)) * (
                x)) + ramp_start_percentage)
            # ramp_design.append((((ramp_end_percentage - ramp_start_percentage) / (ramp_end - ramp_start)) * (x)) + ramp_start_percentage)

        # setup plateu of linear ramp
        for x in range(int(ramp_end - ramp_start), int(ignore_freq_index_above - ignore_freq_index_below)):
            # ramp_design.append(1)
            ramp_design[x] = 1

        # apply ramp filter and find index of maximum frequency (after filter is applied).
        ir_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
        ir_fft_index = -1
        ir_fft_realabs = [None] * (int(ignore_freq_index_above) - int(ignore_freq_index_below))

        grey_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
        grey_fft_index = -1
        grey_fft_realabs = [None] * (int(ignore_freq_index_above) - int(ignore_freq_index_below))

        red_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
        red_fft_index = -1
        red_fft_realabs = [None] * (int(ignore_freq_index_above) - int(ignore_freq_index_below))

        green_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
        green_fft_index = -1
        green_fft_realabs = [None] * (int(ignore_freq_index_above) - int(ignore_freq_index_below))

        blue_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
        blue_fft_index = -1
        blue_fft_realabs = [None] * (int(ignore_freq_index_above) - int(ignore_freq_index_below))

        realabs_i = 0

        for x in range((int(ignore_freq_index_below + 1)), (int(ignore_freq_index_above + 1))):
            # "apply" the ramp to generate the shaped frequency values for IR
            # find the max value and the index of the max value.
            current_irNum = ramp_design[realabs_i] * np.abs(IR_fft_Copy[x].real)
            ir_fft_realabs[realabs_i] = current_irNum
            if ((ir_fft_maxVal is None) or (current_irNum > ir_fft_maxVal)):
                ir_fft_maxVal = current_irNum
                ir_fft_index = x

            if (not ignoregray):
                # "apply" the ramp to generate the shaped frequency values for Grey
                current_greyNum = ramp_design[realabs_i] * np.abs(Gy_fft_Copy[x].real)
                grey_fft_realabs[realabs_i] = current_greyNum
                if ((grey_fft_maxVal is None) or (current_greyNum > grey_fft_maxVal)):
                    grey_fft_maxVal = current_greyNum
                    grey_fft_index = x

            # "apply" the ramp to generate the shaped frequency values for Red
            current_redNum = ramp_design[realabs_i] * np.abs(R_fft_Copy[x].real)
            red_fft_realabs[realabs_i] = current_redNum
            if ((red_fft_maxVal is None) or (current_redNum > red_fft_maxVal)):
                red_fft_maxVal = current_redNum
                red_fft_index = x

            # "apply" the ramp to generate the shaped frequency values for Green
            current_greenNum = ramp_design[realabs_i] * np.abs(G_fft_Copy[x].real)
            green_fft_realabs[realabs_i] = current_greenNum
            if ((green_fft_maxVal is None) or (current_greenNum > green_fft_maxVal)):
                green_fft_maxVal = current_greenNum
                green_fft_index = x

            # "apply" the ramp to generate the shaped frequency values for blue
            current_blueNum = ramp_design[realabs_i] * np.abs(G_fft_Copy[x].real)
            blue_fft_realabs[realabs_i] = current_blueNum
            if ((blue_fft_maxVal is None) or (current_blueNum > blue_fft_maxVal)):
                blue_fft_maxVal = current_blueNum
                blue_fft_index = x

            realabs_i = realabs_i + 1

        self.IrSnr = float(ir_fft_maxVal) / np.average(
            ir_fft_realabs) * 1  # * 1 # could artificially increase SNR for IR as provdes higher accuracy readings, enabling higher weighting for readings
        if (not ignoregray):
            self.GreySnr = float(grey_fft_maxVal) / np.average(grey_fft_realabs)
        else:
            self.GreySnr = 0.0

        self.RedSnr = float(red_fft_maxVal) / np.average(red_fft_realabs)
        self.GreenSnr = float(green_fft_maxVal) / np.average(green_fft_realabs)
        self.BlueSnr = float(blue_fft_maxVal) / np.average(blue_fft_realabs)

        self.IrBpm = freq_bpm[ir_fft_index]

        if (not ignoregray):
            self.GreyBpm = freq_bpm[grey_fft_index]
        else:
            self.GreyBpm = 0

        self.RedBpm = freq_bpm[red_fft_index]
        self.GreenBpm = freq_bpm[green_fft_index]
        self.BlueBpm = freq_bpm[blue_fft_index]
        self.IrFreqencySamplingError = freq_bpm[ir_fft_index + 1] - freq_bpm[ir_fft_index - 1]
        if (not ignoregray):
            self.GreyFreqencySamplingError = freq_bpm[grey_fft_index + 1] - freq_bpm[grey_fft_index - 1]
        else:
            self.GreyFreqencySamplingError = 0.0

        self.RedFreqencySamplingError = freq_bpm[red_fft_index + 1] - freq_bpm[red_fft_index - 1]
        self.GreenFreqencySamplingError = freq_bpm[green_fft_index + 1] - freq_bpm[green_fft_index - 1]
        self.BlueFreqencySamplingError = freq_bpm[blue_fft_index + 1] - freq_bpm[blue_fft_index - 1]

        # unfiltered[np.abs(frequency) < self.ignore_freq_below ] = 0
        # unfiltered[np.abs(frequency) > self.ignore_freq_above ] = 0
        #
        # max_peakBlue = R_fft[np.abs(frequency)  < self.ignore_freq_above].argmax()
        # Bluebpm = np.abs(frequency)[R_fft[np.abs(frequency)  < self.ignore_freq_above].argmax()] * 60
        # freqs = 60 * frequency
        # peakval = R_fft[max_peakBlue]
        # pfreq = freqs[max_peakBlue]
        # snr = peakval/np.average(R_fft)
        # snr2 = peakval/np.average(unfiltered)
        # snr3 = peakval/np.average(R_fft.real)
        # snr4 = peakval/np.average(unfiltered.real)

        t = 0

    def RampStuff6(self, NumSamples, fps, IR_fft, Gy_fft, R_fft, G_fft, B_fft, frequency,
                   ignoregray):  # affter fft use this

        IR_fft_Copy = IR_fft.copy()
        if (not ignoregray):
            Gy_fft_Copy = Gy_fft.copy()
        R_fft_Copy = R_fft.copy()
        G_fft_Copy = G_fft.copy()
        B_fft_Copy = B_fft.copy()

        ramp_end_bpm = 55
        ramp_start_percentage = 0.5
        ramp_end_percentage = 1
        ramp_end_hz = ramp_end_bpm / 60

        # freq_bpm = []  # size of NumSamples
        # for x in range(0, NumSamples):
        #     freq_bpm.append(((x * fps) / NumSamples) * 60)  # in beats per minute

        freq_bpm = 60 * frequency
        # unfiltered = R_fft.copy()

        # define index for the low and high pass frequency filter in frequency space that has just been created
        # (depends on number of samples).
        ignore_freq_index_below = np.rint(((self.ignore_freq_below * NumSamples) / fps))  # high pass
        ignore_freq_index_above = np.rint(((self.ignore_freq_above * NumSamples) / fps))  # low pass

        # compute the ramp filter start and end indices double
        ramp_start = ignore_freq_index_below
        ramp_end = np.rint(((ramp_end_hz * NumSamples) / fps))
        rampDesignLength = ignore_freq_index_above - ignore_freq_index_below
        ramp_design = [None] * int(rampDesignLength)

        ramplooprange = int(ramp_end - ramp_start)
        # setup linear ramp
        for x in range(0, ramplooprange):
            ramp_design[x] = ((((ramp_end_percentage - ramp_start_percentage) / (ramp_end - ramp_start)) * (
                x)) + ramp_start_percentage)
            # ramp_design.append((((ramp_end_percentage - ramp_start_percentage) / (ramp_end - ramp_start)) * (x)) + ramp_start_percentage)

        # setup plateu of linear ramp
        for x in range(int(ramp_end - ramp_start), int(ignore_freq_index_above - ignore_freq_index_below)):
            # ramp_design.append(1)
            ramp_design[x] = 1

        # apply ramp filter and find index of maximum frequency (after filter is applied).
        ir_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
        ir_fft_index = -1
        ir_fft_realabs = [None] * (int(ignore_freq_index_above) - int(ignore_freq_index_below))

        grey_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
        grey_fft_index = -1
        grey_fft_realabs = [None] * (int(ignore_freq_index_above) - int(ignore_freq_index_below))

        red_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
        red_fft_index = -1
        red_fft_realabs = [None] * (int(ignore_freq_index_above) - int(ignore_freq_index_below))

        green_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
        green_fft_index = -1
        green_fft_realabs = [None] * (int(ignore_freq_index_above) - int(ignore_freq_index_below))

        blue_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
        blue_fft_index = -1
        blue_fft_realabs = [None] * (int(ignore_freq_index_above) - int(ignore_freq_index_below))

        realabs_i = 0

        for x in range((int(ignore_freq_index_below + 1)), (int(ignore_freq_index_above + 1))):
            # "apply" the ramp to generate the shaped frequency values for IR
            # find the max value and the index of the max value.
            current_irNum = ramp_design[realabs_i] * np.abs(IR_fft_Copy[x].real)
            ir_fft_realabs[realabs_i] = current_irNum
            if ((ir_fft_maxVal is None) or (current_irNum > ir_fft_maxVal)):
                ir_fft_maxVal = current_irNum
                ir_fft_index = x

            if (not ignoregray):
                # "apply" the ramp to generate the shaped frequency values for Grey
                current_greyNum = ramp_design[realabs_i] * np.abs(Gy_fft_Copy[x].real)
                grey_fft_realabs[realabs_i] = current_greyNum
                if ((grey_fft_maxVal is None) or (current_greyNum > grey_fft_maxVal)):
                    grey_fft_maxVal = current_greyNum
                    grey_fft_index = x

            # "apply" the ramp to generate the shaped frequency values for Red
            current_redNum = ramp_design[realabs_i] * np.abs(R_fft_Copy[x].real)
            red_fft_realabs[realabs_i] = current_redNum
            if ((red_fft_maxVal is None) or (current_redNum > red_fft_maxVal)):
                red_fft_maxVal = current_redNum
                red_fft_index = x

            # "apply" the ramp to generate the shaped frequency values for Green
            current_greenNum = ramp_design[realabs_i] * np.abs(G_fft_Copy[x].real)
            green_fft_realabs[realabs_i] = current_greenNum
            if ((green_fft_maxVal is None) or (current_greenNum > green_fft_maxVal)):
                green_fft_maxVal = current_greenNum
                green_fft_index = x

            # "apply" the ramp to generate the shaped frequency values for blue
            current_blueNum = ramp_design[realabs_i] * np.abs(G_fft_Copy[x].real)
            blue_fft_realabs[realabs_i] = current_blueNum
            if ((blue_fft_maxVal is None) or (current_blueNum > blue_fft_maxVal)):
                blue_fft_maxVal = current_blueNum
                blue_fft_index = x

            realabs_i = realabs_i + 1

        newitem = []
        for x in IR_fft:
            if (x <= 0):
                a = 0
            else:
                newitem.append(x)
        IR_fft = np.array(newitem)

        if (not ignoregray):
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

        self.IrSnr = float(ir_fft_maxVal) / np.average(
            IR_fft.real) * 1  # * 1 # could artificially increase SNR for IR as provdes higher accuracy readings, enabling higher weighting for readings
        if (not ignoregray):
            self.GreySnr = float(grey_fft_maxVal) / np.average(Gy_fft.real)
        else:
            self.GreySnr = 0.0

        self.RedSnr = float(red_fft_maxVal) / np.average(R_fft.real)
        self.GreenSnr = float(green_fft_maxVal) / np.average(G_fft.real)
        self.BlueSnr = float(blue_fft_maxVal) / np.average(B_fft.real)

        self.IrBpm = freq_bpm[ir_fft_index]

        if (not ignoregray):
            self.GreyBpm = freq_bpm[grey_fft_index]
        else:
            self.GreyBpm = 0

        self.RedBpm = freq_bpm[red_fft_index]
        self.GreenBpm = freq_bpm[green_fft_index]
        self.BlueBpm = freq_bpm[blue_fft_index]
        self.IrFreqencySamplingError = freq_bpm[ir_fft_index + 1] - freq_bpm[ir_fft_index - 1]
        if (not ignoregray):
            self.GreyFreqencySamplingError = freq_bpm[grey_fft_index + 1] - freq_bpm[grey_fft_index - 1]
        else:
            self.GreyFreqencySamplingError = 0.0

        self.RedFreqencySamplingError = freq_bpm[red_fft_index + 1] - freq_bpm[red_fft_index - 1]
        self.GreenFreqencySamplingError = freq_bpm[green_fft_index + 1] - freq_bpm[green_fft_index - 1]
        self.BlueFreqencySamplingError = freq_bpm[blue_fft_index + 1] - freq_bpm[blue_fft_index - 1]

        # unfiltered[np.abs(frequency) < self.ignore_freq_below ] = 0
        # unfiltered[np.abs(frequency) > self.ignore_freq_above ] = 0
        #
        # max_peakBlue = R_fft[np.abs(frequency)  < self.ignore_freq_above].argmax()
        # Bluebpm = np.abs(frequency)[R_fft[np.abs(frequency)  < self.ignore_freq_above].argmax()] * 60
        # freqs = 60 * frequency
        # peakval = R_fft[max_peakBlue]
        # pfreq = freqs[max_peakBlue]
        # snr = peakval/np.average(R_fft)
        # snr2 = peakval/np.average(unfiltered)
        # snr3 = peakval/np.average(R_fft.real)
        # snr4 = peakval/np.average(unfiltered.real)

        t = 0

    def Ramp9(self, NumSamples, fps, IR_fft, Gy_fft, R_fft, G_fft, B_fft, frequency, ignoregray):

        freq_bpm = 60. * frequency
        freqs_of_interest2 = frequency
        freqs_of_interest = freq_bpm
        # interest_idx = np.where((freq_bpm > 40) & (freq_bpm < 200))[0]
        # interest_idx_sub = interest_idx[:-1].copy()  # advoid the indexing error
        # freqs_of_interest = freq_bpm[interest_idx_sub]

        # IR_fft = IR_fft[interest_idx_sub]
        # if(not ignoregray):
        #     Gy_fft = Gy_fft[interest_idx_sub]
        # R_fft = R_fft[interest_idx_sub]
        # G_fft = G_fft[interest_idx_sub]
        # B_fft = B_fft[interest_idx_sub]

        # interest_idx2 = np.where((frequency > self.ignore_freq_below) & (frequency < self.ignore_freq_above))[0]
        # interest_idx_sub2 = interest_idx2[:-1].copy()  # advoid the indexing error
        # freqs_of_interest2 = frequency[interest_idx_sub2]

        """ For Hueristic approach of HR detection"""
        IR_sig_max_peak = np.argmax(IR_fft)
        IR_freqValueAtPeak = freqs_of_interest2[IR_sig_max_peak]
        self.IrBpm = freqs_of_interest[IR_sig_max_peak]

        if (not ignoregray):
            Gy_sig_max_peak = np.argmax(Gy_fft)
            Gy_freqValueAtPeak = freqs_of_interest2[Gy_sig_max_peak]
            self.GreyBpm = freqs_of_interest[Gy_sig_max_peak]
        else:
            self.GreyBpm = 0
            Gy_freqValueAtPeak = 0
            Gy_sig_max_peak = 0

        R_sig_max_peak = np.argmax(R_fft)
        R_freqValueAtPeak = freqs_of_interest2[R_sig_max_peak]
        self.RedBpm = freqs_of_interest[R_sig_max_peak]

        G_sig_max_peak = np.argmax(G_fft)
        G_freqValueAtPeak = freqs_of_interest2[G_sig_max_peak]
        self.GreenBpm = freqs_of_interest[G_sig_max_peak]

        B_sig_max_peak = np.argmax(B_fft)
        B_freqValueAtPeak = freqs_of_interest2[B_sig_max_peak]
        self.BlueBpm = freqs_of_interest[B_sig_max_peak]

        self.IrSnr = float(IR_freqValueAtPeak) / np.average(
            IR_fft.real) * 1  # * 1 # could artificially increase SNR for IR as provdes higher accuracy readings, enabling higher weighting for readings

        if (not ignoregray):
            self.GreySnr = float(Gy_freqValueAtPeak) / np.average(Gy_fft.real)
        else:
            self.GreySnr = 0.0

        self.RedSnr = float(R_freqValueAtPeak) / np.average(R_fft.real)
        self.GreenSnr = float(G_freqValueAtPeak) / np.average(G_fft.real)
        self.BlueSnr = float(B_freqValueAtPeak) / np.average(B_fft.real)

        self.IrFreqencySamplingError = freq_bpm[round(np.abs(IR_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(IR_freqValueAtPeak - 1))]
        if (not ignoregray):
            self.GreyFreqencySamplingError = freq_bpm[round(np.abs(Gy_freqValueAtPeak + 1))] - freq_bpm[
                round(np.abs(Gy_freqValueAtPeak - 1))]
        else:
            self.GreyFreqencySamplingError = 0.0

        self.RedFreqencySamplingError = freq_bpm[round(np.abs(R_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(R_freqValueAtPeak - 1))]
        self.GreenFreqencySamplingError = freq_bpm[round(np.abs(G_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(G_freqValueAtPeak - 1))]
        self.BlueFreqencySamplingError = freq_bpm[round(np.abs(B_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(B_freqValueAtPeak - 1))]

    def Ramp8(self, NumSamples, fps, IR_fft, Gy_fft, R_fft, G_fft, B_fft, frequency, ignoregray):

        self.IrBpm, IR_freqValueAtPeak, IR_sig_max_peak = self.find_heart_rate(IR_fft, frequency)

        if (not ignoregray):
            self.GreyBpm, Gy_freqValueAtPeak, Gy_sig_max_peak = self.find_heart_rate(Gy_fft, frequency)
        else:
            self.GreyBpm = 0
            Gy_freqValueAtPeak = 0
            Gy_sig_max_peak = 0

        self.RedBpm, R_freqValueAtPeak, R_sig_max_peak = self.find_heart_rate(R_fft, frequency)
        self.GreenBpm, G_freqValueAtPeak, G_sig_max_peak = self.find_heart_rate(G_fft, frequency)
        self.BlueBpm, B_freqValueAtPeak, B_sig_max_peak = self.find_heart_rate(B_fft, frequency)

        freq_bpm = 60 * frequency

        newitem = []
        for x in IR_fft:
            if (x <= 0):
                a = 0
            else:
                newitem.append(x)
        IR_fft = np.array(newitem)

        if (not ignoregray):
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

        self.IrSnr = float(IR_freqValueAtPeak) / np.average(
            IR_fft.real) * 1  # * 1 # could artificially increase SNR for IR as provdes higher accuracy readings, enabling higher weighting for readings

        if (not ignoregray):
            self.GreySnr = float(Gy_freqValueAtPeak) / np.average(Gy_fft.real)
        else:
            self.GreySnr = 0.0

        self.RedSnr = float(R_freqValueAtPeak) / np.average(R_fft.real)
        self.GreenSnr = float(G_freqValueAtPeak) / np.average(G_fft.real)
        self.BlueSnr = float(B_freqValueAtPeak) / np.average(B_fft.real)

        self.IrFreqencySamplingError = freq_bpm[round(np.abs(IR_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(IR_freqValueAtPeak - 1))]
        if (not ignoregray):
            self.GreyFreqencySamplingError = freq_bpm[round(np.abs(Gy_freqValueAtPeak + 1))] - freq_bpm[
                round(np.abs(Gy_freqValueAtPeak - 1))]
        else:
            self.GreyFreqencySamplingError = 0.0

        self.RedFreqencySamplingError = freq_bpm[round(np.abs(R_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(R_freqValueAtPeak - 1))]
        self.GreenFreqencySamplingError = freq_bpm[round(np.abs(G_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(G_freqValueAtPeak - 1))]
        self.BlueFreqencySamplingError = freq_bpm[round(np.abs(B_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(B_freqValueAtPeak - 1))]

    def Ramp7(self, NumSamples, fps, IR_fft, Gy_fft, R_fft, G_fft, B_fft, frequency, ignoregray):

        self.IrBpm, IR_freqValueAtPeak, IR_sig_max_peak = self.find_heart_rate(IR_fft, frequency)

        if (not ignoregray):
            self.GreyBpm, Gy_freqValueAtPeak, Gy_sig_max_peak = self.find_heart_rate(Gy_fft, frequency)
        else:
            self.GreyBpm = 0
            Gy_freqValueAtPeak = 0
            Gy_sig_max_peak = 0

        self.RedBpm, R_freqValueAtPeak, R_sig_max_peak = self.find_heart_rate(R_fft, frequency)
        self.GreenBpm, G_freqValueAtPeak, G_sig_max_peak = self.find_heart_rate(G_fft, frequency)
        self.BlueBpm, B_freqValueAtPeak, B_sig_max_peak = self.find_heart_rate(B_fft, frequency)

        freq_bpm = 60 * frequency

        self.IrSnr = float(IR_freqValueAtPeak) / np.average(
            IR_fft.real) * 1  # * 1 # could artificially increase SNR for IR as provdes higher accuracy readings, enabling higher weighting for readings

        if (not ignoregray):
            self.GreySnr = float(Gy_freqValueAtPeak) / np.average(Gy_fft.real)
        else:
            self.GreySnr = 0.0

        self.RedSnr = float(R_freqValueAtPeak) / np.average(R_fft.real)
        self.GreenSnr = float(G_freqValueAtPeak) / np.average(G_fft.real)
        self.BlueSnr = float(B_freqValueAtPeak) / np.average(B_fft.real)

        self.IrFreqencySamplingError = freq_bpm[round(np.abs(IR_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(IR_freqValueAtPeak - 1))]
        if (not ignoregray):
            self.GreyFreqencySamplingError = freq_bpm[round(np.abs(Gy_freqValueAtPeak + 1))] - freq_bpm[
                round(np.abs(Gy_freqValueAtPeak - 1))]
        else:
            self.GreyFreqencySamplingError = 0.0

        self.RedFreqencySamplingError = freq_bpm[round(np.abs(R_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(R_freqValueAtPeak - 1))]
        self.GreenFreqencySamplingError = freq_bpm[round(np.abs(G_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(G_freqValueAtPeak - 1))]
        self.BlueFreqencySamplingError = freq_bpm[round(np.abs(B_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(B_freqValueAtPeak - 1))]

    def Ramp5(self, NumSamples, fps, IR_fft, Gy_fft, R_fft, G_fft, B_fft, frequency, ignoregray):

        freq_bpm = 60 * frequency

        IR_sig_max_peak = IR_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
        IR_freqValueAtPeak = frequency[IR_sig_max_peak]  # Get the actual frequency value
        self.IrBpm = np.abs(frequency)[IR_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()] * 60

        if (not ignoregray):
            Gy_sig_max_peak = Gy_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
            Gy_freqValueAtPeak = frequency[Gy_sig_max_peak]  # Get the actual frequency value
            self.GreyBpm = np.abs(frequency)[Gy_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()] * 60
        else:
            Gy_sig_max_peak = 0
            Gy_freqValueAtPeak = 0
            self.GreyBpm = 0

        R_sig_max_peak = R_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
        R_freqValueAtPeak = frequency[R_sig_max_peak]  # Get the actual frequency value
        self.RedBpm = np.abs(frequency)[R_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()] * 60

        G_sig_max_peak = G_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
        G_freqValueAtPeak = frequency[G_sig_max_peak]  # Get the actual frequency value
        self.GreenBpm = np.abs(frequency)[G_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()] * 60

        B_sig_max_peak = B_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
        B_freqValueAtPeak = frequency[B_sig_max_peak]  # Get the actual frequency value
        self.BlueBpm = np.abs(frequency)[B_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()] * 60

        newitem = []
        for x in IR_fft:
            if (x <= 0):
                a = 0
            else:
                newitem.append(x)
        IR_fft = np.array(newitem)

        if (not ignoregray):
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

        self.IrSnr = float(IR_freqValueAtPeak) / np.average(
            IR_fft.real) * 1  # * 1 # could artificially increase SNR for IR as provdes higher accuracy readings, enabling higher weighting for readings

        if (not ignoregray):
            self.GreySnr = float(Gy_freqValueAtPeak) / np.average(Gy_fft.real)
        else:
            self.GreySnr = 0.0

        self.RedSnr = float(R_freqValueAtPeak) / np.average(R_fft.real)
        self.GreenSnr = float(G_freqValueAtPeak) / np.average(G_fft.real)
        self.BlueSnr = float(B_freqValueAtPeak) / np.average(B_fft.real)

        self.IrFreqencySamplingError = freq_bpm[round(np.abs(IR_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(IR_freqValueAtPeak - 1))]
        if (not ignoregray):
            self.GreyFreqencySamplingError = freq_bpm[round(np.abs(Gy_freqValueAtPeak + 1))] - freq_bpm[
                round(np.abs(Gy_freqValueAtPeak - 1))]
        else:
            self.GreyFreqencySamplingError = 0.0

        self.RedFreqencySamplingError = freq_bpm[round(np.abs(R_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(R_freqValueAtPeak - 1))]
        self.GreenFreqencySamplingError = freq_bpm[round(np.abs(G_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(G_freqValueAtPeak - 1))]
        self.BlueFreqencySamplingError = freq_bpm[round(np.abs(B_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(B_freqValueAtPeak - 1))]

    def Ramp4(self, NumSamples, fps, IR_fft, Gy_fft, R_fft, G_fft, B_fft, frequency, ignoregray):

        freq_bpm = 60 * frequency

        IR_sig_max_peak = IR_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
        IR_freqValueAtPeak = frequency[IR_sig_max_peak]  # Get the actual frequency value
        self.IrBpm = freq_bpm[
            IR_sig_max_peak]  # np.abs(frequency)[IR_fft[np.abs(frequency)  <= self.ignore_freq_above].argmax()] * 60

        if (not ignoregray):
            Gy_sig_max_peak = Gy_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
            Gy_freqValueAtPeak = frequency[Gy_sig_max_peak]  # Get the actual frequency value
            self.GreyBpm = freq_bpm[
                Gy_sig_max_peak]  # np.abs(frequency)[Gy_fft[np.abs(frequency)  <= self.ignore_freq_above].argmax()] * 60
        else:
            self.GreyBpm = 0
            self.GreySnr = 0.0
            Gy_sig_max_peak = 0
            Gy_freqValueAtPeak = 0

        R_sig_max_peak = R_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
        R_freqValueAtPeak = frequency[R_sig_max_peak]  # Get the actual frequency value
        self.RedBpm = freq_bpm[
            R_sig_max_peak]  # np.abs(frequency)[R_fft[np.abs(frequency)  <= self.ignore_freq_above].argmax()] * 60

        G_sig_max_peak = G_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
        G_freqValueAtPeak = frequency[G_sig_max_peak]  # Get the actual frequency value
        self.GreenBpm = freq_bpm[
            G_sig_max_peak]  # np.abs(frequency)[G_fft[np.abs(frequency)  <= self.ignore_freq_above].argmax()] * 60

        B_sig_max_peak = B_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
        B_freqValueAtPeak = frequency[B_sig_max_peak]  # Get the actual frequency value
        self.BlueBpm = freq_bpm[
            B_sig_max_peak]  # np.abs(frequency)[B_fft[np.abs(frequency)  <= self.ignore_freq_above].argmax()] * 60

        newitem = []
        for x in IR_fft:
            if (x <= 0):
                a = 0
            else:
                newitem.append(x)
        IR_fft = np.array(newitem)

        if (not ignoregray):
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

        self.BlueSnr = float(B_freqValueAtPeak) / np.average(B_fft.real)
        self.GreenSnr = float(G_freqValueAtPeak) / np.average(G_fft.real)
        self.RedSnr = float(R_freqValueAtPeak) / np.average(R_fft.real)

        if (not ignoregray):
            self.GreySnr = float(Gy_freqValueAtPeak) / np.average(Gy_fft.real)
        else:
            self.GreySnr = 0.0

        self.IrSnr = float(IR_freqValueAtPeak) / np.average(
            IR_fft.real)  # * 1 # could artificially increase SNR for IR as provdes higher accuracy readings, enabling higher weighting for readings

        self.IrFreqencySamplingError = freq_bpm[round(np.abs(IR_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(IR_freqValueAtPeak - 1))]
        if (not ignoregray):
            self.GreyFreqencySamplingError = freq_bpm[round(np.abs(Gy_freqValueAtPeak + 1))] - freq_bpm[
                round(np.abs(Gy_freqValueAtPeak - 1))]
        else:
            self.GreyFreqencySamplingError = 0.0

        self.RedFreqencySamplingError = freq_bpm[round(np.abs(R_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(R_freqValueAtPeak - 1))]
        self.GreenFreqencySamplingError = freq_bpm[round(np.abs(G_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(G_freqValueAtPeak - 1))]
        self.BlueFreqencySamplingError = freq_bpm[round(np.abs(B_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(B_freqValueAtPeak - 1))]

    def Ramp3(self, NumSamples, fps, IR_fft, Gy_fft, R_fft, G_fft, B_fft, frequency, ignoregray):

        freq_bpm = 60 * frequency

        IR_sig_max_peak = IR_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
        IR_freqValueAtPeak = frequency[IR_sig_max_peak]  # Get the actual frequency value
        self.IrBpm = freq_bpm[
            IR_sig_max_peak]  # np.abs(frequency)[IR_fft[np.abs(frequency)  <= self.ignore_freq_above].argmax()] * 60
        self.IrSnr = float(IR_freqValueAtPeak) / np.average(
            IR_fft.real) * 1  # * 1 # could artificially increase SNR for IR as provdes higher accuracy readings, enabling higher weighting for readings

        if (not ignoregray):
            Gy_sig_max_peak = Gy_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
            Gy_freqValueAtPeak = frequency[Gy_sig_max_peak]  # Get the actual frequency value
            self.GreyBpm = freq_bpm[
                Gy_sig_max_peak]  # np.abs(frequency)[Gy_fft[np.abs(frequency)  <= self.ignore_freq_above].argmax()] * 60
            self.GreySnr = float(Gy_freqValueAtPeak) / np.average(Gy_fft.real)
        else:
            self.GreyBpm = 0  # np.abs(frequency)[Gy_fft[np.abs(frequency)  <= self.ignore_freq_above].argmax()] * 60
            self.GreySnr = 0.0
            Gy_freqValueAtPeak = 0
            Gy_sig_max_peak = 0

        R_sig_max_peak = R_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
        R_freqValueAtPeak = frequency[R_sig_max_peak]  # Get the actual frequency value
        self.RedBpm = freq_bpm[
            R_sig_max_peak]  # np.abs(frequency)[R_fft[np.abs(frequency)  <= self.ignore_freq_above].argmax()] * 60
        self.RedSnr = float(R_freqValueAtPeak) / np.average(R_fft.real)

        G_sig_max_peak = G_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
        G_freqValueAtPeak = frequency[G_sig_max_peak]  # Get the actual frequency value
        self.GreenBpm = freq_bpm[
            G_sig_max_peak]  # np.abs(frequency)[G_fft[np.abs(frequency)  <= self.ignore_freq_above].argmax()] * 60
        self.GreenSnr = float(G_freqValueAtPeak) / np.average(G_fft.real)

        B_sig_max_peak = B_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
        B_freqValueAtPeak = frequency[B_sig_max_peak]  # Get the actual frequency value
        self.BlueBpm = freq_bpm[
            B_sig_max_peak]  # np.abs(frequency)[B_fft[np.abs(frequency)  <= self.ignore_freq_above].argmax()] * 60
        self.BlueSnr = float(B_freqValueAtPeak) / np.average(B_fft.real)

        self.IrFreqencySamplingError = freq_bpm[round(np.abs(IR_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(IR_freqValueAtPeak - 1))]
        if (not ignoregray):
            self.GreyFreqencySamplingError = freq_bpm[round(np.abs(Gy_freqValueAtPeak + 1))] - freq_bpm[
                round(np.abs(Gy_freqValueAtPeak - 1))]
        else:
            self.GreyFreqencySamplingError = 0.0

        self.RedFreqencySamplingError = freq_bpm[round(np.abs(R_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(R_freqValueAtPeak - 1))]
        self.GreenFreqencySamplingError = freq_bpm[round(np.abs(G_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(G_freqValueAtPeak - 1))]
        self.BlueFreqencySamplingError = freq_bpm[round(np.abs(B_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(B_freqValueAtPeak - 1))]

    def Ramp2(self, NumSamples, fps, IR_fft, Gy_fft, R_fft, G_fft, B_fft, frequency, ignoregray):

        IR_sig_max_peak = IR_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
        IR_freqValueAtPeak = frequency[IR_sig_max_peak]  # Get the actual frequency value
        self.IrBpm = np.abs(frequency)[IR_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()] * 60
        self.IrSnr = float(IR_freqValueAtPeak) / np.average(
            IR_fft.real) * 1  # * 1 # could artificially increase SNR for IR as provdes higher accuracy readings, enabling higher weighting for readings

        if (not ignoregray):
            Gy_sig_max_peak = Gy_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
            Gy_freqValueAtPeak = frequency[Gy_sig_max_peak]  # Get the actual frequency value
            self.GreyBpm = np.abs(frequency)[Gy_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()] * 60
            self.GreySnr = float(Gy_freqValueAtPeak) / np.average(Gy_fft.real)
        else:
            self.GreySnr = 0.0
            self.GreyBpm = 0
            Gy_sig_max_peak = 0
            Gy_freqValueAtPeak = 0

        R_sig_max_peak = R_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
        R_freqValueAtPeak = frequency[R_sig_max_peak]  # Get the actual frequency value
        self.RedBpm = np.abs(frequency)[R_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()] * 60
        self.RedSnr = float(R_freqValueAtPeak) / np.average(R_fft.real)

        G_sig_max_peak = G_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
        G_freqValueAtPeak = frequency[G_sig_max_peak]  # Get the actual frequency value
        self.GreenBpm = np.abs(frequency)[G_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()] * 60
        self.GreenSnr = float(G_freqValueAtPeak) / np.average(G_fft.real)

        B_sig_max_peak = B_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()  # Find its location
        B_freqValueAtPeak = frequency[B_sig_max_peak]  # Get the actual frequency value
        self.BlueBpm = np.abs(frequency)[B_fft[np.abs(frequency) <= self.ignore_freq_above].argmax()] * 60
        self.BlueSnr = float(B_freqValueAtPeak) / np.average(B_fft.real)

        freq_bpm = 60 * frequency

        self.IrFreqencySamplingError = freq_bpm[round(np.abs(IR_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(IR_freqValueAtPeak - 1))]
        if (not ignoregray):
            self.GreyFreqencySamplingError = freq_bpm[round(np.abs(Gy_freqValueAtPeak + 1))] - freq_bpm[
                round(np.abs(Gy_freqValueAtPeak - 1))]
        else:
            self.GreyFreqencySamplingError = 0.0

        self.RedFreqencySamplingError = freq_bpm[round(np.abs(R_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(R_freqValueAtPeak - 1))]
        self.GreenFreqencySamplingError = freq_bpm[round(np.abs(G_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(G_freqValueAtPeak - 1))]
        self.BlueFreqencySamplingError = freq_bpm[round(np.abs(B_freqValueAtPeak + 1))] - freq_bpm[
            round(np.abs(B_freqValueAtPeak - 1))]
