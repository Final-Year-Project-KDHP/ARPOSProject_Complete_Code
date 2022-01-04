import os
import pickle

import numpy as np
from scipy import signal
from sklearn import preprocessing
from Algorithm import AlgorithmCollection
from FileIO import FileIO
from HeartRateAndSPO.ComputeHeartRate import ComputerHeartRate
from SaveGraphs import Plots
import sys
from WindowData import Window_Data, LogItems
from datetime import datetime

class ProcessDataPartial:

    # region global objects and vairables and construcotr (initialisation)
    # Objects
    objPlots = None
    objAlgorithm = None

    #Global vairbales
    ColorEstimatedFPS = None
    IREstimatedFPS = None
    SavePath = None

    # constructor
    def __init__(self,ColorEstimatedFPS,IREstimatedFPS,SavePath):
        self.objPlots = Plots()
        self.objAlgorithm = AlgorithmCollection()
        # set estimated fps
        self.ColorEstimatedFPS = ColorEstimatedFPS
        self.IREstimatedFPS = IREstimatedFPS
        # Set in algorithm class
        self.objAlgorithm.ColorEstimatedFPS = self.ColorEstimatedFPS
        self.objAlgorithm.IREstimatedFPS = self.IREstimatedFPS
        # Set in plot class
        self.objPlots.ColorEstimatedFPS = self.ColorEstimatedFPS
        self.objPlots.IREstimatedFPS = self.IREstimatedFPS

        #savepath
        self.SavePath = SavePath
        # Create save path if it does not exists
        if not os.path.exists(self.SavePath):
            os.makedirs(self.SavePath)

    # endregion

    # region Graph path and name methods
    '''
    GenerateGrapth: Plot graph and save to disk for signal data
    '''
    def GenerateGrapth(self, graphName, B_Signal, G_Signal, R_Signal, Gy_Signal, IR_Signal,Colorfrequency=None,IRfrequency=None):
        # SingalData
        processedBlue = B_Signal
        processedGreen = G_Signal
        processedRed = R_Signal
        processedGrey = Gy_Signal
        processedIR = IR_Signal

        if (graphName.__contains__("RawData")):
            self.objPlots.plotGraphAllWithoutTimewithParam(self.SavePath, graphName,
                                                           processedBlue, processedGreen, processedRed, processedGrey,
                                                           processedIR,
                                                           "No of Frames", "Intensity")
        elif (graphName.__contains__("PreProcessed")):
            self.objPlots.plotAllinOne(self.SavePath,
                                       processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                                       graphName+"_plotAllinOne", None, "Time(s)", "Amplitude")
            self.objPlots.plotGraphAllwithParam(self.SavePath, graphName+"_plotGraphAllwithParam",None, None,
                                                processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                                                "Time(s)", "Amplitude")
        elif (graphName.__contains__("Algorithm")):
            self.objPlots.plotGraphAllwithParam(self.SavePath, graphName+"_plotGraphAllwithParam",None, None,
                                                processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                                                "Time(s)", "Amplitude")
        elif (graphName.__contains__("Smooth")):
            self.objPlots.plotGraphAllwithParam(self.SavePath, graphName+"_plotGraphAllwithParam",None, None,
                                                processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                                                "Time(s)", "Amplitude")
        elif (graphName.__contains__("Filtered")):
            self.objPlots.PlotFFT(processedBlue, processedGreen, processedRed, processedGrey, processedIR,
                                  Colorfrequency,IRfrequency,
                                  graphName+"_PlotFFT", self.SavePath, graphName+"_PlotFFT")

    # endregion

    # region pre process

    def preprocessdataType1(self, bufferArray, timeCount, isDetrend):
        """remove NaN and Inf values"""
        output = bufferArray[(np.isnan(bufferArray) == 0) & (np.isinf(bufferArray) == 0)]

        detrended_data = output

        if (isDetrend):
            detrended_data = signal.detrend(output)

        try:
            '''interpolation data buffer to make the signal become more periodic (advoid spectral leakage) '''
            L = len(detrended_data)
            even_times = np.linspace(timeCount[0], timeCount[-1], L)
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

    def preprocessdataType2(self, bufferArray, timeCount, isDetrend):
        """remove NaN and Inf values"""
        output = bufferArray[(np.isnan(bufferArray) == 0) & (np.isinf(bufferArray) == 0)]

        detrended_data = output
        if (isDetrend):
            detrended_data = signal.detrend(output)

        try:
            '''interpolation data buffer to make the signal become more periodic (advoid spectral leakage) '''
            L = len(detrended_data)
            even_times = np.linspace(timeCount[0], timeCount[-1], L)
            interp = np.interp(even_times, timeCount, detrended_data)
            interpolated_data = np.hamming(L) * interp
        except:
            interpolated_data = detrended_data

        '''removes noise'''
        smoothed_data = signal.medfilt(interpolated_data, 15)

        '''normalize the input data buffer '''
        normalized_data_med = smoothed_data / np.linalg.norm(smoothed_data)
        return normalized_data_med

    def LogTime(self):
        logTime = datetime(datetime.now().year, datetime.now().month, datetime.now().day,
                           datetime.now().time().hour, datetime.now().time().minute,
                           datetime.now().time().second, datetime.now().time().microsecond)
        return logTime

    '''
    preprocessSignalData: Preprocess techniques to apply on signal data
    '''
    def PreProcessData(self, blue, green, red, grey, Irchannel,GenerateGraphs,region,Preprocess_type,timecolorCount,timeirCount):
        # Log Start Time preprcessing signal data
        startTime = self.LogTime()

        # Processed channel data
        processedBlue = blue
        processedGreen = green
        processedRed = red
        processedGrey = grey
        processedIR = Irchannel

        if (Preprocess_type == 6):
            processedBlue = self.preprocessdataType1(np.array(processedBlue), timecolorCount, True)
            processedGreen = self.preprocessdataType1(np.array(processedGreen), timecolorCount, True)
            processedRed = self.preprocessdataType1(np.array(processedRed), timecolorCount, True)
            processedGrey = self.preprocessdataType1(np.array(processedGrey), timecolorCount, True)
            processedIR = self.preprocessdataType1(np.array(processedIR), timeirCount, True)

        elif (Preprocess_type == 5):
            processedBlue = self.preprocessdataType1(np.array(processedBlue), timecolorCount, False)
            processedGreen = self.preprocessdataType1(np.array(processedGreen), timecolorCount, False)
            processedRed = self.preprocessdataType1(np.array(processedRed), timecolorCount, False)
            processedGrey = self.preprocessdataType1(np.array(processedGrey), timecolorCount, False)
            processedIR = self.preprocessdataType1(np.array(processedIR), timeirCount, False)

        elif (Preprocess_type == 4):
            processedBlue = self.preprocessdataType2(np.array(processedBlue), timecolorCount, True)
            processedGreen = self.preprocessdataType2(np.array(processedGreen), timecolorCount, True)
            processedRed = self.preprocessdataType2(np.array(processedRed), timecolorCount, True)
            processedGrey = self.preprocessdataType2(np.array(processedGrey), timecolorCount, True)
            processedIR = self.preprocessdataType2(np.array(processedIR), timeirCount, True)

        elif (Preprocess_type == 3):
            processedBlue = self.preprocessdataType2(np.array(processedBlue), timecolorCount, False)
            processedGreen = self.preprocessdataType2(np.array(processedGreen), timecolorCount, False)
            processedRed = self.preprocessdataType2(np.array(processedRed), timecolorCount, False)
            processedGrey = self.preprocessdataType2(np.array(processedGrey), timecolorCount, False)
            processedIR = self.preprocessdataType2(np.array(processedIR), timeirCount, False)

        elif (Preprocess_type == 2):
            SCombined = self.getSignalDataCombined(processedBlue, processedGreen, processedRed, processedGrey, processedIR)
            SCombined = preprocessing.normalize(SCombined)
            # split
            processedBlue = SCombined[:, 0]
            processedGreen = SCombined[:, 1]
            processedRed = SCombined[:, 2]
            processedGrey = SCombined[:, 3]
            processedIR = SCombined[:, 4]
        else: #1
            processedBlue = blue
            processedGreen = green
            processedRed = red
            processedGrey = grey
            processedIR = Irchannel

        endTime = self.LogTime()
        diffTime = (endTime - startTime)

        # generate PreProcessed plot
        if (GenerateGraphs):
            self.GenerateGrapth("PreProcessed_" + region + "_type-"+ str(Preprocess_type), processedBlue,processedGreen,processedRed,processedGrey, processedIR)

        #Combbine
        SCombined = self.getSignalDataCombined(processedBlue, processedGreen, processedRed, processedGrey, processedIR)

        objProcessedData = ProcessedData()
        objProcessedData.ProcessedSignalData = SCombined
        objProcessedData.IREstimatedFPS = self.IREstimatedFPS
        objProcessedData.ColorEstimatedFPS = self.ColorEstimatedFPS
        objProcessedData.startTime = startTime
        objProcessedData.endTime = endTime
        objProcessedData.diffTime = diffTime

        #wRITE to dsik
        self.WritetoDisk(self.SavePath,'PreProcessedData_'+ region + "_type-"+ str(Preprocess_type),objProcessedData)

        del objProcessedData
        #Writelog
        # self.WriteLog(startTime,endTime,diffTime,region,Preprocess_type)

    # endregion


    '''
    ApplyAlgorithm: Applies algorithms on signal data
    '''

    def ApplyAlgorithm(self, SCombined,Algorithm_type,components,GenerateGraphs,region,preProcessType):
        # Log Start Time preprcessing signal data
        startTime = self.LogTime()
        # Apply by Algorithm_type
        if (Algorithm_type == "FastICA"):
            SResult_ = self.objAlgorithm.ApplyICA(SCombined, components)
        elif (Algorithm_type == "PCA"):
            SResult_ = self.objAlgorithm.ApplyPCA(SCombined, components)
        elif (Algorithm_type == "PCAICA"):
            SCombined = self.objAlgorithm.ApplyPCA(SCombined, components)
            SResult_ = self.objAlgorithm.ApplyICA(SCombined, components)
        elif (Algorithm_type == "Jade"):
            # https://github.com/kellman/heartrate_matlab/blob/master/jadeR.m
            SResult_ = self.objAlgorithm.jadeOptimised(SCombined, components)  # r4 is slwoer and f5 is faster

            # Split data
            newBlue = np.array(SResult_[0])[0].real
            newGreen = np.array(SResult_[1])[0].real
            newRed = np.array(SResult_[2])[0].real
            newGrey = np.array(SResult_[3])[0].real
            newIr = np.array(SResult_[4])[0].real

            # Combine r,g,b,gy,ir in one array
            SResult_ = self.getSignalDataCombined(newBlue, newGreen, newRed, newGrey, newIr)

        else:
            SResult_ = SCombined

        endTime = self.LogTime()
        diffTime = (endTime - startTime)

        if (GenerateGraphs):
            self.GenerateGrapth("Algorithm_" + region + "_type-" + str(Algorithm_type) + '_PreProcessType-'+str(preProcessType),
                                SResult_[:, 0],SResult_[:, 1],SResult_[:, 2],SResult_[:, 3],SResult_[:, 4])


        objProcessedData = ProcessedData()
        objProcessedData.ProcessedSignalData = SResult_
        objProcessedData.IREstimatedFPS = self.IREstimatedFPS
        objProcessedData.ColorEstimatedFPS = self.ColorEstimatedFPS
        objProcessedData.startTime = startTime
        objProcessedData.endTime = endTime
        objProcessedData.diffTime = diffTime

        #wRITE to dsik
        self.WritetoDisk(self.SavePath,'AlgorithmData_'+ region + "_type-"+ str(Algorithm_type) + '_PreProcessType-'+str(preProcessType),objProcessedData)


    '''
    SmoothenData: Smooth data
    '''
    def smooth(self, x, window_len=11, window='hamming'):
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

    def SmoothenData(self, SResult_,GenerateGraphs,region,preProcessType,Algotype):
        # Log Start Time preprcessing signal data
        startTime = self.LogTime()

        Smoothenblue = self.smooth(SResult_[:, 0])
        Smoothengreen = self.smooth(SResult_[:, 1])
        Smoothenred = self.smooth(SResult_[:, 2])
        Smoothengrey = self.smooth(SResult_[:, 3])
        SmoothenIR = self.smooth(SResult_[:, 4])

        endTime = self.LogTime()
        diffTime = (endTime - startTime)

        # Combine into one array
        SmoothedSignal_ = self.getSignalDataCombined(Smoothenblue, Smoothengreen, Smoothenred, Smoothengrey, SmoothenIR)

        if (GenerateGraphs):
            self.GenerateGrapth("Smoothed_" + region + "_Algotype-" + str(Algotype) + '_PreProcessType-' + str(preProcessType),
                                SmoothedSignal_[:, 0],SmoothedSignal_[:, 1],SmoothedSignal_[:, 2],SmoothedSignal_[:, 3],SmoothedSignal_[:, 4])

        objProcessedData = ProcessedData()
        objProcessedData.ProcessedSignalData = SmoothedSignal_
        objProcessedData.IREstimatedFPS = self.IREstimatedFPS
        objProcessedData.ColorEstimatedFPS = self.ColorEstimatedFPS
        objProcessedData.startTime = startTime
        objProcessedData.endTime = endTime
        objProcessedData.diffTime = diffTime

        # wRITE to dsik
        self.WritetoDisk(self.SavePath,
                         'SmoothedData_' + region + "_Algotype-" + str(Algotype) + '_PreProcessType-' + str(
                             preProcessType), objProcessedData)


    '''
    ApplyFFT: types of fft on signal data
    '''
    def ApplyFFT(self, S,FFT_type,region,GenerateGraphs,isSmoothed , Algotype, preProcessType):
        ignoreGray =False
        # Log Start Time preprcessing signal data
        startTime = self.LogTime()
        if (FFT_type == "M1"):
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, Colorfreq, IRfreq = self.objAlgorithm.Apply_FFT_WithoutPower_M4_eachsignal(S,ignoreGray)  # rfft

        elif (FFT_type == "M2"):  # with fft shift
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, Colorfreq, IRfreq = self.objAlgorithm.Apply_FFT_M2_eachsignal(S,ignoreGray)  # rfft

        if (FFT_type == "M3"):
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, Colorfreq, IRfreq = self.objAlgorithm.Apply_FFT_M1_byeachsignal(S,ignoreGray)  # rfft

        elif (FFT_type == "M4"):
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, Colorfreq, IRfreq = self.objAlgorithm.ApplyFFT9(S,ignoreGray)  # fft

        elif (FFT_type == "M5"):
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, Colorfreq, IRfreq = self.objAlgorithm.Apply_FFT_M6_Individual(S,ignoreGray)  # sqrt

        elif (FFT_type == "M6"):  # with fft shift
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, Colorfreq, IRfreq = self.objAlgorithm.Apply_FFT_M5_Individual(S,ignoreGray)  # sqrt

        elif (FFT_type == "M7"):
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, Colorfreq, IRfreq = self.objAlgorithm.Apply_FFT_forpreprocessed(S,ignoreGray)  # sqrt

        if (np.iscomplex(B_fft.any())):
            B_fft = B_fft.real

        if (np.iscomplex(Gr_fft.any())):
            Gr_fft = Gr_fft.real

        if (np.iscomplex(R_fft.any())):
            R_fft = R_fft.real

        if (np.iscomplex(Gy_fft.any())):
            Gy_fft = Gy_fft.real

        if (np.iscomplex(IR_fft.any())):
            IR_fft = IR_fft.real

        endTime = self.LogTime()
        diffTime = (endTime - startTime)

        # Combine into one array
        S_fft = self.getSignalDataCombined(B_fft, Gr_fft, R_fft, Gy_fft, IR_fft)

        if (GenerateGraphs):
            self.GenerateGrapth("FFT-" + FFT_type + "_" + region + "_Smoothed-" + str(isSmoothed) + "_Algotype-" + str(Algotype) + '_PreProcessType-' + str(preProcessType),
                                S_fft[:, 0],S_fft[:, 1],S_fft[:, 2],S_fft[:, 3],S_fft[:, 4])

        objProcessedData = ProcessedData()
        objProcessedData.ProcessedSignalData = S_fft
        objProcessedData.IREstimatedFPS = self.IREstimatedFPS
        objProcessedData.ColorEstimatedFPS = self.ColorEstimatedFPS
        objProcessedData.Colorfrequency = Colorfreq
        objProcessedData.IRfrequency = IRfreq
        objProcessedData.startTime = startTime
        objProcessedData.endTime = endTime
        objProcessedData.diffTime = diffTime

        # wRITE to dsik
        self.WritetoDisk(self.SavePath,
                         "FFT-" + FFT_type + "_" + region + "_Smoothed-" + str(isSmoothed) + "_Algotype-" + str(Algotype) + '_PreProcessType-' + str(preProcessType),
                         objProcessedData)

    def Filterfft_BelowandAbove(self, sig_fft,type,Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above):
        sig_fft_filtered = abs(sig_fft.copy())
        if(type== 'color'):
            sig_fft_filtered[np.abs(Colorfrequency) <= ignore_freq_below] = 0
            sig_fft_filtered[np.abs(Colorfrequency) >= ignore_freq_above] = 0
        else:
            sig_fft_filtered[np.abs(IRfrequency) <= ignore_freq_below] = 0
            sig_fft_filtered[np.abs(IRfrequency) >= ignore_freq_above] = 0
        return sig_fft_filtered

    def Filterfft_Below(self, sig_fft,type,Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above):
        sig_fft_filtered = abs(sig_fft.copy())
        if (type == 'color'):
            sig_fft_filtered[np.abs(Colorfrequency) <= ignore_freq_below] = 0
        else:
            sig_fft_filtered[np.abs(IRfrequency) <= ignore_freq_below] = 0
        return sig_fft_filtered

    def Filterfft_cuttoff(self, fftarray,type,Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above):
        if(type== 'color'):
            bound_low = (np.abs(Colorfrequency - ignore_freq_below)).argmin()
            bound_high = (np.abs(Colorfrequency - ignore_freq_above)).argmin()
            fftarray[:bound_low] = 0
            fftarray[bound_high:-bound_high] = 0
            fftarray[-bound_low:] = 0
            return fftarray
        else:
            bound_low = (np.abs(IRfrequency - ignore_freq_below)).argmin()
            bound_high = (np.abs(IRfrequency - ignore_freq_above)).argmin()
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

    def Filterfft_butterbandpass(self, blue, green, red, grey, Ir, ordno,ignore_freq_below,ignore_freq_above):
        # change for virable estimated fps
        B_bp = self.butter_bandpass_filter(blue, ignore_freq_below, ignore_freq_above, self.ColorEstimatedFPS,
                                           order=ordno)
        G_bp = self.butter_bandpass_filter(green, ignore_freq_below, ignore_freq_above, self.ColorEstimatedFPS,
                                           order=ordno)
        R_bp = self.butter_bandpass_filter(red, ignore_freq_below, ignore_freq_above, self.ColorEstimatedFPS,
                                           order=ordno)
        Gy_bp = self.butter_bandpass_filter(grey, ignore_freq_below, ignore_freq_above, self.ColorEstimatedFPS,
                                            order=ordno)
        IR_bp = self.butter_bandpass_filter(Ir, ignore_freq_below, ignore_freq_above, self.IREstimatedFPS,
                                            order=ordno)
        return B_bp, G_bp, R_bp, Gy_bp, IR_bp

    def Filterfft_LimitFreq_Belowfilter(self, signal,type,Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above):
        if(type== 'color'):
            interest_idx = np.where((Colorfrequency >= ignore_freq_below))[0]
            interest_idx_sub = interest_idx.copy()  #[:-1] advoid the indexing error
            fft_of_interest = signal[interest_idx_sub]
            Colorfrequency = Colorfrequency[interest_idx_sub]
            return fft_of_interest, Colorfrequency
        else:
            interest_idx = np.where((IRfrequency >= ignore_freq_below))[0]
            interest_idx_sub = interest_idx.copy()  #[:-1] advoid the indexing error
            fft_of_interest = signal[interest_idx_sub]
            IRfrequency = IRfrequency[interest_idx_sub]
            return fft_of_interest, IRfrequency

    def Filterfft_LimitFreq_BelowAbovefilter(self, signal,type,Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above):

        if(type== 'color'):
            interest_idx = np.where((Colorfrequency >= ignore_freq_below) & (Colorfrequency <= ignore_freq_above))[0]
            interest_idx_sub = interest_idx.copy()  #[:-1] advoid the indexing error
            fft_of_interest = signal[interest_idx_sub]
            Colorfrequency = Colorfrequency[interest_idx_sub]
            return fft_of_interest,Colorfrequency
        else:
            interest_idx = np.where((IRfrequency >= ignore_freq_below) & (IRfrequency <= ignore_freq_above))[0]
            interest_idx_sub = interest_idx.copy()  #[:-1] advoid the indexing error
            fft_of_interest = signal[interest_idx_sub]
            IRfrequency = IRfrequency[interest_idx_sub]
            return fft_of_interest,IRfrequency
    '''
    FilterTechniques: Applies filters on signal data
    '''
    def FilterTechniques(self, ProcessedSignalData,region,GenerateGraphs,isSmoothed , Algotype, preProcessType,
                         FFT_type,Filter_type,Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above):

        B_fft = ProcessedSignalData[:, 0]
        G_fft = ProcessedSignalData[:, 1]
        R_fft =ProcessedSignalData[:, 2]
        Gy_fft = ProcessedSignalData[:, 3]
        IR_fft = ProcessedSignalData[:, 4]

        # Log Start Time preprcessing signal data
        startTime = self.LogTime()
        # cuttoff,  butterworth and other
        if (Filter_type == 1):  # Not very good with rampstuff method, very high heart rate
            B_filtered = self.Filterfft_BelowandAbove(B_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            G_filtered = self.Filterfft_BelowandAbove(G_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            R_filtered = self.Filterfft_BelowandAbove(R_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            Gy_filtered = self.Filterfft_BelowandAbove(Gy_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            IR_filtered = self.Filterfft_BelowandAbove(IR_fft,'ir',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
        elif (Filter_type == 2):
            B_filtered = self.Filterfft_Below(B_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            G_filtered = self.Filterfft_Below(G_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            R_filtered = self.Filterfft_Below(R_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            Gy_filtered = self.Filterfft_Below(Gy_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            IR_filtered = self.Filterfft_Below(IR_fft,'ir',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
        elif (Filter_type == 3):
            B_filtered = self.Filterfft_cuttoff(B_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            G_filtered = self.Filterfft_cuttoff(G_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            R_filtered = self.Filterfft_cuttoff(R_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            Gy_filtered = self.Filterfft_cuttoff(Gy_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            IR_filtered = self.Filterfft_cuttoff(IR_fft,'ir',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
        elif (Filter_type == 4):  # Not very good with rampstuff method
            B_filtered, G_filtered, R_filtered, Gy_filtered, IR_filtered = self.Filterfft_butterbandpass(B_fft, G_fft,
                                                                                                         R_fft, Gy_fft,
                                                                                                         IR_fft, 6,ignore_freq_below,ignore_freq_above)
        elif (Filter_type == 5):
            # No Filter
            B_filtered = B_fft
            G_filtered = G_fft
            R_filtered = R_fft
            Gy_filtered = Gy_fft
            IR_filtered = IR_fft
        elif (Filter_type == 6):
            B_filtered,Colorfrequency = self.Filterfft_LimitFreq_BelowAbovefilter(B_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            G_filtered,Colorfrequency = self.Filterfft_LimitFreq_BelowAbovefilter(G_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            R_filtered ,Colorfrequency= self.Filterfft_LimitFreq_BelowAbovefilter(R_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            Gy_filtered,Colorfrequency = self.Filterfft_LimitFreq_BelowAbovefilter(Gy_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            IR_filtered,IRfrequency = self.Filterfft_LimitFreq_BelowAbovefilter(IR_fft,'ir',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
        elif (Filter_type == 7):
            B_filtered ,Colorfrequency= self.Filterfft_LimitFreq_Belowfilter(B_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            G_filtered ,Colorfrequency= self.Filterfft_LimitFreq_Belowfilter(G_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            R_filtered ,Colorfrequency= self.Filterfft_LimitFreq_Belowfilter(R_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            Gy_filtered,Colorfrequency = self.Filterfft_LimitFreq_Belowfilter(Gy_fft,'color',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)
            IR_filtered ,IRfrequency= self.Filterfft_LimitFreq_Belowfilter(IR_fft,'ir',Colorfrequency,IRfrequency,ignore_freq_below,ignore_freq_above)

        endTime = self.LogTime()
        diffTime = (endTime - startTime)

        if (GenerateGraphs):
            #Filtered",Filter_type
            self.GenerateGrapth("Filtered_FL-"+ str(Filter_type)+ "_"+ region+ "_FFTtype-" + str(FFT_type)  + "_algotype-" + str(Algotype) +
                                '_PreProcessType-'+str(preProcessType)+ "_Smoothed-" + str(isSmoothed),
                                B_filtered, G_filtered,R_filtered, Gy_filtered,IR_filtered,Colorfrequency,IRfrequency)


        objProcessedData = ProcessedData()
        # objProcessedData.ProcessedSignalData = SResult_
        objProcessedData.B_signal = B_filtered
        objProcessedData.G_signal = G_filtered
        objProcessedData.R_signal = R_filtered
        objProcessedData.Gy_signal = Gy_filtered
        objProcessedData.IR_signal = IR_filtered
        objProcessedData.Colorfrequency = Colorfrequency
        objProcessedData.IRfrequency = IRfrequency
        objProcessedData.IREstimatedFPS = self.IREstimatedFPS
        objProcessedData.ColorEstimatedFPS = self.ColorEstimatedFPS
        objProcessedData.startTime = startTime
        objProcessedData.endTime = endTime
        objProcessedData.diffTime = diffTime

        #wRITE to dsik
        self.WritetoDisk(self.SavePath,
                         "Filtered_FL-"+ str(Filter_type)+ "_"+ region+ "_FFTtype-" + str(FFT_type)  + "_algotype-" + str(Algotype) +
                         '_PreProcessType-'+str(preProcessType)+ "_Smoothed-" + str(isSmoothed)
                         ,objProcessedData)
        h=0

        # region calculate result (HR in bpm) using frequency

    def generateHeartRateandSNR(self, B_filtered, G_filtered, R_filtered, Gy_filtered, IR_filtered, type,
                                Colorfrequency,IRfrequency,ColorEstimatedFPS,IREstimatedFPS,
                                preProcessType, Algotype, IsSmooth, FFT_type, filterType,
                                ignore_freq_below_bpm, ignore_freq_above_bpm,region):

        # Log Start Time preprcessing signal data
        startTime = self.LogTime()

        # Create copy of channels
        ir_fft_realabs = IR_filtered.copy()
        grey_fft_realabs = Gy_filtered.copy()
        red_fft_realabs = R_filtered.copy()
        green_fft_realabs = G_filtered.copy()
        blue_fft_realabs = B_filtered.copy()

        # Calculate samples
        NumSamples = len(red_fft_realabs)
        freq_bpmColor = 60 * Colorfrequency
        freq_bpmIr = 60 * IRfrequency

        ramp_end_bpm = 55
        ramp_start_percentage = 0.5
        ramp_end_percentage = 1

        # Compute heart rate and snr
        objComputerHeartRate = ComputerHeartRate(0, False, NumSamples, 3,
                                                 4, 5,
                                                 ColorEstimatedFPS, IREstimatedFPS, ramp_end_bpm,
                                                 ramp_start_percentage, ramp_end_percentage,
                                                 ignore_freq_below_bpm, ignore_freq_above_bpm,
                                                 freq_bpmColor, freq_bpmIr, Colorfrequency,
                                                 IRfrequency, self.SavePath, region)

        if (type == 1):
            objComputerHeartRate.OriginalARPOSmethod(blue_fft_realabs, green_fft_realabs, red_fft_realabs,
                                                     grey_fft_realabs, ir_fft_realabs)
        elif (type == 2):
            objComputerHeartRate.getHeartRate_fromFrequency(blue_fft_realabs, green_fft_realabs, red_fft_realabs,
                                                            grey_fft_realabs, ir_fft_realabs)
        elif (type == 3):
            objComputerHeartRate.getHearRate_fromFrequencyWithFilter_Main(blue_fft_realabs, green_fft_realabs,
                                                                          red_fft_realabs, grey_fft_realabs,
                                                                          ir_fft_realabs)

        endTime = self.LogTime()
        diffTime = (endTime - startTime)

        ##Copy obj compute HR data to local class
        objresultProcessedData = ResultProcessedData()

        objresultProcessedData.IrSnr =  objComputerHeartRate.IrSnr
        objresultProcessedData.GreySnr = objComputerHeartRate.GreySnr
        objresultProcessedData.RedSnr = objComputerHeartRate.RedSnr
        objresultProcessedData.GreenSnr = objComputerHeartRate.GreenSnr
        objresultProcessedData.BlueSnr = objComputerHeartRate.BlueSnr
        objresultProcessedData.IrBpm = objComputerHeartRate.IrBpm
        objresultProcessedData.GreyBpm = objComputerHeartRate.GreyBpm
        objresultProcessedData.RedBpm = objComputerHeartRate.RedBpm
        objresultProcessedData.GreenBpm = objComputerHeartRate.GreenBpm
        objresultProcessedData.BlueBpm = objComputerHeartRate.BlueBpm
        objresultProcessedData.IrFreqencySamplingError = objComputerHeartRate.IrFreqencySamplingError
        objresultProcessedData.GreyFreqencySamplingError = objComputerHeartRate.GreyFreqencySamplingError
        objresultProcessedData.RedFreqencySamplingError = objComputerHeartRate.RedFreqencySamplingError
        objresultProcessedData.GreenFreqencySamplingError = objComputerHeartRate.GreenFreqencySamplingError
        objresultProcessedData.BlueFreqencySamplingError = objComputerHeartRate.BlueFreqencySamplingError
        objresultProcessedData.startTime = startTime
        objresultProcessedData.endTime = endTime
        objresultProcessedData.diffTime = diffTime
        objresultProcessedData.blue_fft_realabs = blue_fft_realabs
        objresultProcessedData.green_fft_realabs = green_fft_realabs
        objresultProcessedData.red_fft_realabs = red_fft_realabs
        objresultProcessedData.grey_fft_realabs = grey_fft_realabs
        objresultProcessedData.ir_fft_realabs = ir_fft_realabs

        #Write to dsik
        self.WritetoDisk(self.SavePath,
                         "ResultType_RS-" + str(type) + "_Filtered_FL-"+ str(filterType)+ "_"+ region+ "_FFTtype-" + str(FFT_type)  + "_algotype-" + str(Algotype) +
                         '_PreProcessType-'+str(preProcessType)+ "_Smoothed-" + str(IsSmooth)
                         ,objresultProcessedData)

        ##delete obj compute hr data
        del objComputerHeartRate


    def WriteLog(self,startTime,endTime,diffTime,region,type):
        #LogTime and write to disk
        TimeLog = []
        TimeLog.append('StartTime: '+ str(startTime))
        TimeLog.append('EndTime: '+ str(endTime))
        TimeLog.append('diffTime: '+ str(diffTime))
        objFile = FileIO()
        objFile.WriteListDatatoFile(self.SavePath,'TimeLog_PreProcessedData_'+ region + "_type-"+ str(type), TimeLog)
        del objFile


    '''
    getSignalDataCombined: Combine r,g,b,gy,ir to one array
    '''
    def getSignalDataCombined(self, blue, green, red, grey, Irchannel):
        S = np.c_[blue, green, red, grey, Irchannel]
        return S

    def WritetoDisk(self,location, filename, data):
        ##STORE Data
        with open(location + filename, 'wb') as filehandle:
            pickle.dump(data, filehandle)

    def ReadfromDisk(self,location, filename):
        ##Read data
        with open(location + filename, 'rb') as filehandle:
            data = pickle.load(filehandle)
        return data

class ProcessedData:
    ProcessedSignalData = []
    ColorEstimatedFPS = None
    IREstimatedFPS = None
    startTime = None
    endTime = None
    diffTime = None
    Colorfrequency = None
    IRfrequency = None
    B_signal= None
    G_signal= None
    R_signal= None
    Gy_signal= None
    IR_signal= None


class ResultProcessedData:
    IrSnr = None
    GreySnr = None
    RedSnr = None
    GreenSnr = None
    BlueSnr = None
    IrBpm = None
    GreyBpm = None
    RedBpm = None
    GreenBpm = None
    BlueBpm = None
    IrFreqencySamplingError = None
    GreyFreqencySamplingError = None
    RedFreqencySamplingError = None
    GreenFreqencySamplingError = None
    BlueFreqencySamplingError = None
    startTime = None
    endTime = None
    diffTime = None
    blue_fft_realabs = None
    green_fft_realabs = None
    red_fft_realabs = None
    grey_fft_realabs = None
    ir_fft_realabs = None