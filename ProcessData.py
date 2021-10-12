import os
import numpy as np
from scipy import signal
import datetime
from sklearn import preprocessing
from Algorithm import AlgorithmCollection
from SaveGraphs import Plots
import sys
from scipy.fftpack import fft, ifft, fftfreq
from SingalToNoise import SNR
from WindowData import Window_Data


class ProcessFaceData:
    # Image data constants
    red = []
    blue = []
    green = []
    grey = []
    Irchannel = []
    time_listcolor = []
    time_listir = []
    timecolorCount = []
    timeirCount = []
    HasStartTime = 0
    StartTime = datetime.datetime.now()
    EndTime = datetime.datetime.now()

    # Filtereddata
    B_bp = []
    G_bp = []
    R_bp = []
    Gy_bp = []
    IR_bp = []

    # Ohter constatns
    EstimatedFPS = 30  # sort for variable

    # setup highpass filter
    ignore_freq_below_bpm = 40
    ignore_freq_below = ignore_freq_below_bpm / 60  # ignore_freq_below_bpm / 60; // Hz(was 0.2)0.7 #

    # setup low pass filter
    ignore_freq_above_bpm = 200
    ignore_freq_above = ignore_freq_above_bpm / 60  # Hz(was 0.2) 4.0 #

    # select region with highest signal to noise
    bestHeartRateSnr = 0.0
    bestBpm = 0.0
    freqencySamplingError = 0.0
    oxygenLevels = []

    smallestOxygenError = sys.float_info.max

    def GetFrameTime(self, hr, min, sec, mili):
        year = datetime.datetime.now().year
        month = datetime.datetime.now().month
        day = datetime.datetime.now().day
        FrameTimeStamp = datetime.datetime(year, month, day, int(hr), int(min), int(sec), int(mili))
        return FrameTimeStamp

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

    def ApplyFilterwithParam(self, blue, green, red, grey, Ir):
        # change for virable estimated fps
        self.B_bp = self.butter_bandpass_filter(blue, self.ignore_freq_below, self.ignore_freq_above, self.EstimatedFPS,
                                                order=6)
        self.G_bp = self.butter_bandpass_filter(green, self.ignore_freq_below, self.ignore_freq_above,
                                                self.EstimatedFPS, order=6)
        self.R_bp = self.butter_bandpass_filter(red, self.ignore_freq_below, self.ignore_freq_above, self.EstimatedFPS,
                                                order=6)
        self.Gy_bp = self.butter_bandpass_filter(grey, self.ignore_freq_below, self.ignore_freq_above,
                                                 self.EstimatedFPS, order=6)
        self.IR_bp = self.butter_bandpass_filter(Ir, self.ignore_freq_below, self.ignore_freq_above, self.EstimatedFPS,
                                                 order=6)

    def ApplyFilterwithParamReturn(self, blue, green, red, grey, Ir, ordno, ignoregray):
        # change for virable estimated fps
        B_bp = self.butter_bandpass_filter(blue, self.ignore_freq_below, self.ignore_freq_above, self.EstimatedFPS,
                                           order=ordno)
        G_bp = self.butter_bandpass_filter(green, self.ignore_freq_below, self.ignore_freq_above, self.EstimatedFPS,
                                           order=ordno)
        R_bp = self.butter_bandpass_filter(red, self.ignore_freq_below, self.ignore_freq_above, self.EstimatedFPS,
                                           order=ordno)

        if (not ignoregray):
            Gy_bp = self.butter_bandpass_filter(grey, self.ignore_freq_below, self.ignore_freq_above, self.EstimatedFPS,
                                                order=ordno)
        else:
            Gy_bp = []

        IR_bp = self.butter_bandpass_filter(Ir, self.ignore_freq_below, self.ignore_freq_above, self.EstimatedFPS,
                                            order=ordno)
        return B_bp, G_bp, R_bp, Gy_bp, IR_bp

    def ApplyFilter(self):
        # change for virable estimated fps
        self.B_bp = self.butter_bandpass_filter(self.blue, self.ignore_freq_below, self.ignore_freq_above,
                                                self.EstimatedFPS, order=6)
        self.G_bp = self.butter_bandpass_filter(self.green, self.ignore_freq_below, self.ignore_freq_above,
                                                self.EstimatedFPS, order=6)
        self.R_bp = self.butter_bandpass_filter(self.red, self.ignore_freq_below, self.ignore_freq_above,
                                                self.EstimatedFPS, order=6)
        self.Gy_bp = self.butter_bandpass_filter(self.grey, self.ignore_freq_below, self.ignore_freq_above,
                                                 self.EstimatedFPS, order=6)
        self.IR_bp = self.butter_bandpass_filter(self.Irchannel, self.ignore_freq_below, self.ignore_freq_above,
                                                 self.EstimatedFPS, order=6)

    SavefigPath = ''
    LoadColordataPath = ''
    LoadIRdataPath = ''

    IrSnr = 0.0
    GreySnr = 0.0
    RedSnr = 0.0
    GreenSnr = 0.0
    BlueSnr = 0.0

    IrFreqencySamplingError = 0.0
    GreyFreqencySamplingError = 0.0
    RedFreqencySamplingError = 0.0
    GreenFreqencySamplingError = 0.0
    BlueFreqencySamplingError = 0.0

    OxygenSaturation = 0.0
    OxygenSaturationError = 0.0

    def fft_filter(self, freqs, fftarray):
        bound_low = (np.abs(freqs - self.ignore_freq_below)).argmin()
        bound_high = (np.abs(freqs - self.ignore_freq_above)).argmin()
        fftarray[:bound_low] = 0
        fftarray[bound_high:-bound_high] = 0
        fftarray[-bound_low:] = 0
        # iff = ifft(fftarray, axis=0)
        # result = np.abs(iff)
        # result *= 100  # Amplification factor
        return fftarray

    def fft_filter_3(self, frq, sig):
        '''bandpass filter for fourier'''
        y = sig.copy()
        x = frq.copy()
        y3 = []
        x3 = []

        y1 = y[(np.abs(x) <= self.ignore_freq_below)]
        x1 = x[(np.abs(x) <= self.ignore_freq_below)]

        y2 = y[(np.abs(x) >= self.ignore_freq_above)]
        x2 = x[(np.abs(x) >= self.ignore_freq_above)]

        for value in y1:
            y3.append(value)

        for value in x1:
            x3.append(value)

        for value in y2:
            y3.append(value)

        for value in x2:
            x3.append(value)

        return x3, y3

        # Calculate heart rate from FFT peaks

    def find_heart_rate(self, fftarray, freqs):
        freq_min = self.ignore_freq_below
        freq_max = self.ignore_freq_above
        fft_maximums = []

        for i in range(fftarray.shape[0]):
            if freq_min <= freqs[i] <= freq_max:
                fftMap = abs(fftarray[i].real)
                fft_maximums.append(fftMap.max())
            else:
                fft_maximums.append(0)

        peaks, properties = signal.find_peaks(fft_maximums)
        max_peak = -1
        max_freq = 0

        # Find frequency with max amplitude in peaks
        for peak in peaks:
            if fft_maximums[peak] > max_freq:
                max_freq = fft_maximums[peak]
                max_peak = peak

        return freqs[max_peak] * 60, freqs[max_peak], max_peak

        # Calculate heart rate from FFT peaks

    def find_heart_rate2(self, fftarray, freqs):

        peaks, properties = signal.find_peaks(fftarray)
        max_peak = -1
        max_freq = 0

        # Find frequency with max amplitude in peaks
        for peak in peaks:
            if fftarray[peak] > max_freq:
                max_freq = fftarray[peak]
                max_peak = peak

        return freqs[max_peak] * 60, freqs[max_peak], max_peak

    def signaltonoise(self, a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m / sd)

    def GetChannelFPS(self, channelFPS):
        fps = 30

        return fps

    def find_max(self, fft_maximums, freqs):
        peaks, properties = signal.find_peaks(fft_maximums)
        max_peak = -1
        max_freq = 0

        # Find frequency with max amplitude in peaks
        for peak in peaks:
            if fft_maximums[peak] > max_freq:
                max_freq = fft_maximums[peak]
                max_peak = peak

        return freqs[max_peak] * 60, freqs[max_peak], max_peak

    smallestOxygenError = 0.0
    regionToUse = ''

    def FindGreyPeak(self, fftarray, freqs, freq_min, freq_max):
        fft_maximums = []

        for i in range(fftarray.shape[0]):
            if freq_min <= freqs[i] <= freq_max:
                fftMap = abs(fftarray[i].real)
                fft_maximums.append(fftMap.max())
            else:
                fft_maximums.append(0)

        filteredgrey = ifft(fft_maximums)
        peaks, properties = signal.find_peaks(filteredgrey)

        return peaks, properties, filteredgrey

    def CalcualateSPOPart2(self, Gy_bp, heartRatePeriod, IR_bp, R_bp, G_bp, distanceM, smallestOxygenError, region):

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
            irValue = IR_bp[index].real  # ir cahnnel original before
            redValue = R_bp[index].real  # red cahnnel original before
            greenValue = G_bp[index].real  # green cahnnel original before
            if (irValue > 0):
                # irValue = Infra_fftMaxVal
                # redValue = red_fftMaxVal  # red cahnnel original before
                # greenValue = gy_fftMaxVal  # green cahnnel original before

                distValue = distanceM[index]  # 0.7 #self.distanceMm[index]
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
                # irToRedRatio = (red_oxy_mean / ir_oxy_mean) * ((irValue * (( distValue * distValue) / 1000000) / redValue)) / 52  # at a distance of one meter using 99% white reflections standard, 52 is the ir/r scaling factor.
                val4 = (irValue * ((distValue)) / redValue)
                val5 = (red_oxy_mean / ir_oxy_mean)
                val6 = (val4)  # / 52
                irToRedRatio = val5 * val6  # at a distance of one meter using 99% white reflections standard, 52 is the ir/r scaling factor.

                val3 = (irToRedRatio * ir_deoxy_mean)
                val1 = (red_deoxy_mean - val3)
                val2 = (ir_oxy_mean + red_deoxy_mean - ir_deoxy_mean - red_oxy_mean)
                oxygenLevel = 100 * (val1 / val2)  # - 7 # 2
                oxygenLevels.append(round(oxygenLevel))

        # compute SD and mean of oxygenLevels
        self.OxygenSaturation = np.std(oxygenLevels)
        self.OxygenSaturationError = np.std(oxygenLevels, ddof=1) / np.sqrt(
            np.size(oxygenLevels))  # MeanStandardDeviation err

        oxyval = str(oxygenLevels[0])
        # lowesterr=sys.float_info.max

        # for ox in oxygenLevels:
        #     stdsingle = np.std(ox,oxygenLevels)
        #     if(stdsingle <lowesterr): #if close to the dtta , spreded val
        #         lowesterr=stdsingle
        #         oxyval=stdsingle

        if (self.OxygenSaturationError < smallestOxygenError):
            self.smallestOxygenError = self.OxygenSaturationError
            self.regionToUse = region

            # if(self.smallestOxygenError>10):
            #     a=0
            # else:
            #     rounded = round(self.smallestOxygenError)
            #     oxyval= str(oxygenLevels[rounded])

            # if (self.OxygenSaturation > self.oxygenstd):
            #     oxygenstd = self.OxygenSaturation
            #     print("STD : " + str(self.OxygenSaturation) + " , error: " + str(self.OxygenSaturationError))
            #     print("SPO 0 : " + str(oxygenLevels[0]))

        return self.OxygenSaturation, self.OxygenSaturationError, oxyval

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

    def CalcualateSPO(self, Gy_bp, heartRatePeriod, IR_bp, R_bp, G_bp, distanceM, smallestOxygenError, region):

        filteredgrey = ifft(Gy_bp)
        ###GET SPO
        numSamples = len(filteredgrey)
        greyPeeks = self.FindPeeks(Gy_bp)

        # find the max peeks in each sample window
        grayMaxPeekIndice = []

        formid = len(filteredgrey) - int(heartRatePeriod)  # grey channel orinigal before
        for x in range(0, formid, int(heartRatePeriod)):
            maxGreyPeekInPeriod = self.FindMaxPeekInPeriod(greyPeeks, filteredgrey, int(heartRatePeriod), x)
            grayMaxPeekIndice.append((maxGreyPeekInPeriod))

        # get the values of the ir and red channels at the grey peeks
        oxygenLevels = []
        for index in grayMaxPeekIndice:
            irValue = IR_bp[index].real  # ir cahnnel original before
            redValue = R_bp[index].real  # red cahnnel original before
            greenValue = G_bp[index].real  # green cahnnel original before

            # irValue = Infra_fftMaxVal
            # redValue = red_fftMaxVal  # red cahnnel original before
            # greenValue = gy_fftMaxVal  # green cahnnel original before

            distValue = distanceM[index]  # 0.7 #self.distanceMm[index]

            # Absorption values calculated of oxy and deoxy heamoglobin
            # where red defined as: 600-700nm
            # where ir defined as: 858-860nm.
            red_deoxy_mean = 4820.4361
            red_oxy_mean = 667.302
            ir_deoxy_mean = 693.38
            ir_oxy_mean = 1087.2

            # Depth-resultion blood oxygen satyuration measuremnet by dual-wavelength photothermal (DWP) optical coherence tomography
            # Biomed Opt Express
            # 2011 P491-504

            irToRedRatio = (red_oxy_mean / ir_oxy_mean) * ((irValue * ((
                                                                               distValue * distValue) / 1000000) / redValue)) / 52  # at a distance of one meter using 99% white reflections standard, 52 is the ir/r scaling factor.
            oxygenLevel = 100 * (red_deoxy_mean - (irToRedRatio * ir_deoxy_mean)) / (
                    ir_oxy_mean + red_deoxy_mean - ir_deoxy_mean - red_oxy_mean) - 2
            oxygenLevels.append(oxygenLevel)

        print("SPO 0 : " + str(oxygenLevels[0]))
        # compute SD and mean of oxygenLevels
        self.OxygenSaturation = np.std(oxygenLevels)
        self.OxygenSaturationError = np.std(oxygenLevels, ddof=1) / np.sqrt(
            np.size(oxygenLevels))  # MeanStandardDeviation err

        if (self.OxygenSaturationError < smallestOxygenError):
            self.smallestOxygenError = self.OxygenSaturationError
            self.regionToUse = region

    def ApplyAlgorithm(self, algotype, S, objAlgorithm, components):
        S_ = S

        if (algotype == "FastICA"):
            S_ = objAlgorithm.ApplyICA(S, components)

        elif (algotype == "PCA"):
            S_ = objAlgorithm.ApplyPCA(S, components)

        elif (algotype == "ICAM4"):
            S = objAlgorithm.ApplyPCA(S, components)
            S_ = objAlgorithm.ApplyICA(S, components)
        elif (algotype == "SVD"):
            S_ = objAlgorithm.ApplySVD()  # to do
        elif (algotype == "Jade"):
            S_ = objAlgorithm.jadeR5(S, components)  # r4 is slwoer and f5 is faster
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

                S = np.c_[newBlue2, newGreen2, newRed2, newGrey2, newir2]

            else:
                # IR
                newIr = np.array(S_[3])
                newir2 = newIr[0].real

                S = np.c_[newBlue2, newGreen2, newRed2, newir2]

            S_ = S
        else:
            S_ = S

        return S_

    def ApplyFFT(self, fftype, S, objAlgorithm, objPlots, ignoregray):
        blue_max_peak = 0
        green_max_peak = 0
        red_max_peak = 0
        grey_max_peak = 0
        ir_max_peak = 0
        blue_bpm = 0
        green_bpm = 0
        red_bpm = 0
        grey_bpm = 0
        ir_bpm = 0
        if (fftype == "M1"):
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, frequency, \
            blue_max_peak, green_max_peak, red_max_peak, grey_max_peak, ir_max_peak, \
            blue_bpm, green_bpm, red_bpm, grey_bpm, ir_bpm = objAlgorithm.Apply_FFT_WithoutPower_M4_eachsignal(S,
                                                                                                               objPlots,
                                                                                                               self.ignore_freq_below,
                                                                                                               self.ignore_freq_above,
                                                                                                               ignoregray)  # rfft

        elif (fftype == "M2"):  # with fft shift
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, frequency = objAlgorithm.Apply_FFT_M2_eachsignal(S, objPlots,
                                                                                                   self.ignore_freq_below,
                                                                                                   self.ignore_freq_above,
                                                                                                   ignoregray)  # rfft

        if (fftype == "M3"):
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, frequency, \
            blue_max_peak, green_max_peak, red_max_peak, grey_max_peak, ir_max_peak, \
            blue_bpm, green_bpm, red_bpm, grey_bpm, ir_bpm = objAlgorithm.Apply_FFT_M1_byeachsignal(S, objPlots,
                                                                                                    self.ignore_freq_below,
                                                                                                    self.ignore_freq_above,
                                                                                                    ignoregray)  # rfft

        elif (fftype == "M4"):
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, frequency, \
            blue_max_peak, green_max_peak, red_max_peak, grey_max_peak, ir_max_peak, \
            blue_bpm, green_bpm, red_bpm, grey_bpm, ir_bpm = objAlgorithm.ApplyFFT9(S, objPlots, self.ignore_freq_below,
                                                                                    self.ignore_freq_above,
                                                                                    ignoregray)  # fft

        elif (fftype == "M5"):
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, frequency, \
            blue_max_peak, green_max_peak, red_max_peak, grey_max_peak, ir_max_peak, \
            blue_bpm, green_bpm, red_bpm, grey_bpm, ir_bpm = objAlgorithm.Apply_FFT_M6_Individual(S, objPlots,
                                                                                                  ignoregray)  # sqrt

        elif (fftype == "M6"):  # with fft shift
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, frequency = objAlgorithm.Apply_FFT_M5_Individual(S,
                                                                                                   ignoregray)  # sqrt

        elif (fftype == "M7"):
            B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, frequency = objAlgorithm.Apply_FFT_forpreprocessed(S,
                                                                                                     ignoregray)  # sqrt

        ##Failed Tests
        # elif(fftype == "M2"): ##Fail for joint array
        #     B_fft, Gr_fft, R_fft, Gy_fft, IR_fft,frequency = objAlgorithm.Apply_FFT_M4(S,objPlots) #ffttest3
        # elif(fftype == "M3"): ##generates zero everytime so failed
        #     B_fft, Gr_fft, R_fft, Gy_fft, IR_fft = objAlgorithm.Apply_FFT_M5_Individual(S) #ffttest3 fft shift
        # elif(fftype == "M6"):
        #     B_fft, Gr_fft, R_fft, Gy_fft, IR_fft,frequency = objAlgorithm.Apply_FFT_M5(S) # ffft shift fail
        # elif(fftype == "M5"):
        #     B_fft, Gr_fft, R_fft, Gy_fft, IR_fft,frequency = objAlgorithm.Apply_FFT_M6(S) # All sqt fail

        if (np.iscomplex(B_fft.any())):
            B_fft = B_fft.real

        if (np.iscomplex(Gr_fft.any())):
            Gr_fft = Gr_fft.real

        if (np.iscomplex(R_fft.any())):
            R_fft = R_fft.real

        if (not ignoregray):
            if (np.iscomplex(Gy_fft.any())):
                Gy_fft = Gy_fft.real

        if (np.iscomplex(IR_fft.any())):
            IR_fft = IR_fft.real

        return B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, frequency, \
               blue_max_peak, green_max_peak, red_max_peak, grey_max_peak, ir_max_peak, \
               blue_bpm, green_bpm, red_bpm, grey_bpm, ir_bpm

    def GetHeartRateDetails(self, fft, freq_bpm):
        freqs_max_peak_bpm, freqs_max_peakval, max_peak_index = self.find_heart_rate2(fft, freq_bpm)

        return freqs_max_peak_bpm, freqs_max_peakval, max_peak_index

    def CalculateSNR(self, type, Ir_freqs_max_peakval, Gy_freqs_max_peakval, R_freqs_max_peakval, Gr_freqs_max_peakval,
                     B_freqs_max_peakval, IR_fft, Gy_fft, R_fft, Gr_fft, B_fft):
        objSnr = SNR()

        if (type == 1):
            IrSnr = (float(Ir_freqs_max_peakval) / np.average(IR_fft.real))  # *1
            GreySnr = float(Gy_freqs_max_peakval) / np.average(Gy_fft.real)  ##uncomemnt for other
            RedSnr = float(R_freqs_max_peakval) / np.average(R_fft.real)
            GreenSnr = float(Gr_freqs_max_peakval) / np.average(Gr_fft.real)
            BlueSnr = 0.0  # float(B_freqs_max_peakval) / np.average(B_fft.real)

            # d = decimal.Decimal(self.IrSnr)
            # valueD = d.as_tuple().exponent
            # postiiveval = np.abs(valueD)
            # newval = ''
            #
            # for x in range(1,postiiveval):
            #     newval = newval + '0'

            # newval = 1000 #float(newval)
            # self.IrSnr *= newval
            # self.IrSnr = self.IrSnr*2
            #
            # self.GreySnr *= newval
            # self.RedSnr *= newval
            # self.GreenSnr *= newval
            # self.BlueSnr *= newval


        elif (type == 2):
            self.IrSnr = np.abs(objSnr.signaltonoiseDB(IR_fft))
            # self.IrSnr = self.IrSnr*2
            GreySnr = np.abs(objSnr.signaltonoiseDB(Gy_fft))
            RedSnr = np.abs(objSnr.signaltonoiseDB(R_fft))
            GreenSnr = np.abs(objSnr.signaltonoiseDB(Gr_fft))
            BlueSnr = 0.0  # np.abs(objSnr.signaltonoiseDB(B_fft))

        elif (type == 3):
            IrSnr = np.abs(objSnr.signaltonoise(IR_fft))
            # self.IrSnr = self.IrSnr*2
            GreySnr = np.abs(objSnr.signaltonoise(Gy_fft))
            RedSnr = np.abs(objSnr.signaltonoise(R_fft))
            GreenSnr = np.abs(objSnr.signaltonoise(Gr_fft))
            BlueSnr = 0.0  # np.abs(objSnr.signaltonoise(B_fft))

        return IrSnr, GreySnr, RedSnr, GreenSnr, BlueSnr

    def GetBestBpm(self):
        if (self.IrSnr > self.bestHeartRateSnr):
            self.bestHeartRateSnr = self.IrSnr
            self.bestBpm = self.IrBpm

        if (self.GreySnr > self.bestHeartRateSnr):
            self.bestHeartRateSnr = self.GreySnr
            self.bestBpm = self.GreyBpm

        if (self.RedSnr > self.bestHeartRateSnr):
            self.bestHeartRateSnr = self.RedSnr
            self.bestBpm = self.RedBpm

        if (self.GreenSnr > self.bestHeartRateSnr):
            self.bestHeartRateSnr = self.GreenSnr
            self.bestBpm = self.GreenBpm

        if (self.BlueSnr > self.bestHeartRateSnr):
            self.bestHeartRateSnr = self.BlueSnr
            self.bestBpm = self.BlueBpm

        # work out the length of time of one heart beat - the heart rate period
        if (self.bestBpm > 0):
            self.beatsPerSecond = self.bestBpm / 60.0
        else:
            self.beatsPerSecond = 1

        heartRatePeriod = self.EstimatedFPS / self.beatsPerSecond  # window size in sample
        return heartRatePeriod

    def ProcessSPO_Window(self, heartRatePeriod, distanceM):
        ###GET SPO
        unfilteredGray = ifft(self.Gy_bp)  # rollback inverse fft
        numSamples = len(unfilteredGray)
        greyPeeks = self.FindPeeks(unfilteredGray)

        # find the max peeks in each sample window
        grayMaxPeekIndice = []

        formid = len(unfilteredGray) - int(heartRatePeriod)  # grey channel orinigal before
        for x in range(0, formid, int(heartRatePeriod)):
            maxGreyPeekInPeriod = self.FindMaxPeekInPeriod(greyPeeks, unfilteredGray, int(heartRatePeriod), x)
            grayMaxPeekIndice.append((maxGreyPeekInPeriod))

        # get the values of the ir and red channels at the grey peeks
        for index in grayMaxPeekIndice:
            irValue = self.IR_bp[index]  # ir cahnnel original before
            redValue = self.R_bp[index]  # red cahnnel original before
            greenValue = self.G_bp[index]  # green cahnnel original before
            if (irValue > 0):
                # irValue = Infra_fftMaxVal
                # redValue = red_fftMaxVal  # red cahnnel original before
                # greenValue = gy_fftMaxVal  # green cahnnel original before

                distValue = distanceM[index]  # 0.7 #self.distanceM[index]

                # Absorption values calculated of oxy and deoxy heamoglobin
                # where red defined as: 600-700nm
                # where ir defined as: 858-860nm.
                red_deoxy_mean = 4820.4361  # assumption
                red_oxy_mean = 667.302  # assumption
                ir_deoxy_mean = 693.38
                ir_oxy_mean = 1087.2

                # Check paper below for 52 value
                # Depth-resultion blood oxygen satyuration measuremnet by dual-wavelength photothermal (DWP) optical coherence tomography
                # Biomed Opt Express
                # 2011 P491-504
                # at a distance of one meter using 99% white reflections standard, 52 is the ir/r scaling factor.
                irToRedRatio = (red_oxy_mean / ir_oxy_mean) * (
                    (irValue * ((distValue * distValue) / 1000000) / redValue)) / 52
                oxygenLevel = 100 * (red_deoxy_mean - (irToRedRatio * ir_deoxy_mean)) / (
                        ir_oxy_mean + red_deoxy_mean - ir_deoxy_mean - red_oxy_mean) - 2  # -2 is tweeked value
                self.oxygenLevels.append(oxygenLevel)

        print("SPO 0 : " + str(self.oxygenLevels[0]))
        # compute SD and mean of self.oxygenLevels
        OxygenSaturation = np.std(self.oxygenLevels)
        OxygenSaturationError = np.std(self.oxygenLevels, ddof=1) / np.sqrt(
            np.size(self.oxygenLevels))  # MeanStandardDeviation err

        if (OxygenSaturationError < self.smallestOxygenError):
            smallestOxygenError = OxygenSaturationError
            # regionToUse = region

    def Process_Window_Data_RGBIr(self, Alldata, distanceM, fps, algotype, fftype, region):
        objPlots = Plots()
        objAlgorithm = AlgorithmCollection()

        NumSamples = len(Alldata)
        # # calculate the frequency array
        T = 1 / 30  # timestep
        freq_bpm = fftfreq(NumSamples, T)

        self.blue = Alldata[:, 0]
        self.green = Alldata[:, 1]
        self.red = Alldata[:, 2]
        # self.grey = Alldata[:, 3]
        # self.Irchannel = Alldata[:, 3]

        S = np.c_[self.blue, self.green, self.red]  # self.grey,,  self.Irchannel

        ##preprocess
        # ma = moving_average(arr, 4)  # smooth raw color signals
        # detrended = signal.detrend(S)  # scipy detrend
        # S = (detrended - np.mean(detrended)) / np.std(detrended)

        # # Apply ica
        S_ = self.ApplyAlgorithm(algotype, S, objPlots, objAlgorithm)

        # ApplyFilter
        # before or after?
        # B_bp, G_bp, R_bp, Gy_bp, IR_bp = self.ApplyFilterwithParamReturn(S_[:, 0],S_[:, 1],S_[:, 2],S_[:, 3],S_[:, 4]) #self.blue, self.green, self.red, self.grey, self.Irchannel)
        # B_bp = self.fft_filter(freq_bpm,S_[:, 0])
        # G_bp = self.fft_filter(freq_bpm,S_[:, 1])
        # R_bp = self.fft_filter(freq_bpm,S_[:, 2])
        # Gy_bp = self.fft_filter(freq_bpm,S_[:, 3])
        # IR_bp = self.fft_filter(freq_bpm,S_[:, 4])
        # S_ = np.c_[B_bp, G_bp, R_bp, Gy_bp, IR_bp]

        # S_fft, B_freqs_max_peak_bpm, B_freqs_max_peakval, B_max_peak_index = objAlgorithm.Apply_FFT_M5_Individual(S_)
        # Apply fft
        B_fft, Gr_fft, R_fft, Gy_fft, IR_fft = self.ApplyFFT(fftype, S_, objAlgorithm)

        # B_fft, Gr_fft, R_fft, Gy_fft, IR_fft = self.ApplyFilterwithParamReturn(B_fft, Gr_fft, R_fft, Gy_fft, IR_fft)
        B_fft = self.fft_filter(freq_bpm, B_fft)
        Gr_fft = self.fft_filter(freq_bpm, Gr_fft)
        R_fft = self.fft_filter(freq_bpm, R_fft)
        # Gy_fft = self.fft_filter(freq_bpm, Gy_fft)
        # IR_fft = self.fft_filter(freq_bpm, IR_fft)

        # Get Heart rate details , get max peaks and hr
        B_freqs_max_peak_bpm, B_freqs_max_peakval, B_max_peak_index = self.GetHeartRateDetails(B_fft,
                                                                                               freq_bpm)  # blue channel
        Gr_freqs_max_peak_bpm, Gr_freqs_max_peakval, Gr_max_peak_index = self.GetHeartRateDetails(Gr_fft,
                                                                                                  freq_bpm)  # green channel
        R_freqs_max_peak_bpm, R_freqs_max_peakval, R_max_peak_index = self.GetHeartRateDetails(R_fft,
                                                                                               freq_bpm)  # red channel
        # Gy_freqs_max_peak_bpm, Gy_freqs_max_peakval, Gy_max_peak_index = self.GetHeartRateDetails(Gy_fft,
        #                                                                                           freq_bpm)  # grey channel
        # Ir_freqs_max_peak_bpm, Ir_freqs_max_peakval, Ir_max_peak_index = self.GetHeartRateDetails(IR_fft,
        #                                                                                           freq_bpm)  # Ir channel

        freq_bpm = 60. * freq_bpm  # in beats per minute

        # Set bpm Value
        self.IrBpm = 0  # Ir_freqs_max_peak_bpm
        # self.GreyBpm = Gy_freqs_max_peak_bpm
        self.RedBpm = R_freqs_max_peak_bpm
        self.GreenBpm = Gr_freqs_max_peak_bpm
        self.BlueBpm = B_freqs_max_peak_bpm

        # Find SNRs 2 ways 2nd way better
        self.CalculateSNR(2, None, None, R_freqs_max_peakval, Gr_freqs_max_peakval,
                          B_freqs_max_peakval, IR_fft, Gy_fft, R_fft, Gr_fft, B_fft)
        # print("IR bmp: " + str(Ir_freqs_max_peak_bpm) + " , Grey bmp: " + str(Gy_freqs_max_peak_bpm) + " , Red bmp: " + str(R_freqs_max_peak_bpm) + " , Green bmp: " + str(Gr_freqs_max_peak_bpm) + " , Blue bmp: " + str( B_freqs_max_peak_bpm))

        heartRatePeriod = self.GetBestBpm()
        # print("Best bpm:" + str(self.bestBpm) + ", Best hr snr" + str(self.bestHeartRateSnr))

        windowList = Window_Data()
        windowList.WindowNo = 1
        windowList.BestBPM = self.bestBpm
        windowList.BestSnR = self.bestHeartRateSnr
        # windowList.OtherBpms = ("IR bmp: " + str(Ir_freqs_max_peak_bpm) + " , Grey bmp: " + str(Gy_freqs_max_peak_bpm) + " , Red bmp: " + str(R_freqs_max_peak_bpm) + " , Green bmp: " + str(Gr_freqs_max_peak_bpm) + " , Blue bmp: " + str( B_freqs_max_peak_bpm))
        windowList.IrSnr = self.IrSnr
        windowList.GreySnr = self.GreySnr
        windowList.RedSnr = self.RedSnr
        windowList.GreenSnr = self.GreenSnr
        windowList.BlueSnr = self.BlueSnr
        windowList.BlueBpm = self.BlueBpm
        windowList.IrBpm = self.IrBpm
        windowList.GreyBpm = 0  # self.GreyBpm
        windowList.RedBpm = self.RedBpm
        windowList.GreenBpm = self.GreenBpm
        windowList.regiontype = region

        return windowList

        # ApplyFilter

    def filterTechq2(self, freqs, signal):
        # freqs = 60. * freq

        interest_idx = np.where((freqs >= self.ignore_freq_below) & (freqs <= self.ignore_freq_above))[0]
        interest_idx_sub = interest_idx[:-1].copy()  # advoid the indexing error
        # freqs_of_interest = freqs_in_minute[interest_idx_sub]
        fft_of_interest = signal[interest_idx_sub]

        return fft_of_interest

    def filterTechq3(self, freqs, signal):
        # freqs = 60. * freq

        interest_idx = np.where((freqs >= self.ignore_freq_below))[0]
        interest_idx_sub = interest_idx[:-1].copy()  # advoid the indexing error
        # freqs_of_interest = freqs_in_minute[interest_idx_sub]
        fft_of_interest = signal[interest_idx_sub]

        return fft_of_interest

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

    def RampStuff(self, NumSamples, fps, IR_fft, Gy_fft, R_fft, G_fft, B_fft, frequency,
                  ignoregray):  # affter fft use this
        # if (len(ramp_design) > len(IR_fft)):
        #     ramp_design = [None] * int(len(IR_fft))
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

    def smooth(self, x, window_len=11, window='hanning'):
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

    def Filterffq(self, sig_fft, freq):

        # copy the FFT results
        sig_fft_filtered = abs(sig_fft.copy())
        # high-pass filter by assign zeros to the
        # FFT amplitudes where the absolute
        # frequencies smaller than the cut-off
        sig_fft_filtered[np.abs(freq) <= self.ignore_freq_below] = 0
        sig_fft_filtered[np.abs(freq) >= self.ignore_freq_above] = 0
        #
        # sig_max_peak = sig_fft_filtered[np.abs(freq)  <= self.ignore_freq_above].argmax()  # Find its location
        # freqValueAtPeak = freq[sig_max_peak]  # Get the actual frequency value
        # sig_bpm = np.abs(freq)[sig_fft_filtered[np.abs(freq)  <= self.ignore_freq_above].argmax()] * 60
        return sig_fft_filtered  # ,sig_max_peak, sig_bpm,freqValueAtPeak

    def Filterffq2(self, sig_fft, freq):

        # copy the FFT results
        sig_fft_filtered = abs(sig_fft.copy())
        # high-pass filter by assign zeros to the
        # FFT amplitudes where the absolute
        # frequencies smaller than the cut-off
        sig_fft_filtered[np.abs(freq) <= self.ignore_freq_below] = 0
        # sig_fft_filtered[np.abs(freq) > self.ignore_freq_above ] = 0
        #
        # sig_max_peak = sig_fft_filtered[np.abs(freq)  <= self.ignore_freq_above].argmax()  # Find its location
        # freqValueAtPeak = freq[sig_max_peak]  # Get the actual frequency value
        # sig_bpm = np.abs(freq)[sig_fft_filtered[np.abs(freq)  <= self.ignore_freq_above].argmax()] * 60
        return sig_fft_filtered  # ,sig_max_peak, sig_bpm,freqValueAtPeak

    def preprocessdata(self, bufferArray, timecolorcountLips, isDetrend):
        """remove NaN and Inf values"""
        output = bufferArray[(np.isnan(bufferArray) == 0) & (np.isinf(bufferArray) == 0)]

        detrended_data = output

        if (isDetrend):
            detrended_data = signal.detrend(output)

        try:
            '''interpolation data buffer to make the signal become more periodic (advoid spectral leakage) '''
            L = len(detrended_data)
            even_times = np.linspace(timecolorcountLips[0], timecolorcountLips[-1], L)
            interp = np.interp(even_times, timecolorcountLips, detrended_data)
            interpolated_data = np.hamming(L) * interp
        except:
            interpolated_data = detrended_data

        '''removes noise'''
        smoothed_data = signal.medfilt(interpolated_data, 15)
        N = 3
        """ x == an array of data. N == number of samples per average """
        cumsum = np.cumsum(np.insert(interpolated_data, [0, 0, 0], 0))
        rm = (cumsum[N:] - cumsum[:-N]) / float(N)

        '''normalize the input data buffer '''
        normalized_data = interpolated_data / np.linalg.norm(interpolated_data)
        normalized_data_med = smoothed_data / np.linalg.norm(smoothed_data)
        return normalized_data

    def preprocessdata2(self, bufferArray, timecolorcountLips, isDetrend):
        """remove NaN and Inf values"""
        output = bufferArray[(np.isnan(bufferArray) == 0) & (np.isinf(bufferArray) == 0)]

        detrended_data = output
        if (isDetrend):
            detrended_data = signal.detrend(output)

        try:
            '''interpolation data buffer to make the signal become more periodic (advoid spectral leakage) '''
            L = len(detrended_data)
            even_times = np.linspace(timecolorcountLips[0], timecolorcountLips[-1], L)
            interp = np.interp(even_times, timecolorcountLips, detrended_data)
            interpolated_data = np.hamming(L) * interp
        except:
            interpolated_data = detrended_data

        '''removes noise'''
        smoothed_data = signal.medfilt(interpolated_data, 15)
        N = 3
        """ x == an array of data. N == number of samples per average """
        cumsum = np.cumsum(np.insert(interpolated_data, [0, 0, 0], 0))
        rm = (cumsum[N:] - cumsum[:-N]) / float(N)

        '''normalize the input data buffer '''
        normalized_data = interpolated_data / np.linalg.norm(interpolated_data)
        normalized_data_med = smoothed_data / np.linalg.norm(smoothed_data)
        return normalized_data_med

    def Process_EntireSignalData(self, Alldata, distanceM, fps, algotype, fftype, region, Algopath, timecolorLips,
                            timecolorcountLips, timeirLips,
                            timeircountLips, WindowCount, filtertype, resulttype, ignoregray, isSmoothen, IsDetrend,
                            preprocess, savepath, GenerateGraphs, timeinSeconds):

        imagepath = savepath + "Graphs\\" + str(WindowCount) + "\\" + region + "\\"
        if not os.path.exists(imagepath):
            os.makedirs(imagepath)

        grappath = savepath + "dataChannelConent\\" + str(WindowCount) + "\\" + region + "\\"
        if not os.path.exists(grappath):
            os.makedirs(grappath)

        components = 5
        if (ignoregray):
            components = 4

        objPlots = Plots()
        objAlgorithm = AlgorithmCollection()
        objAlgorithm.EstimatedFPS = fps

        NumSamples = len(Alldata)
        T = 1 / 30  # timestep

        self.blue = Alldata[:, 0]
        self.green = Alldata[:, 1]
        self.red = Alldata[:, 2]
        self.grey = Alldata[:, 3]
        self.Irchannel = Alldata[:, 4]

        if (GenerateGraphs):
            imageName = "Originaldata_Algotype_" + algotype + "_fft_" + fftype + "_Fl_" + str(
                filtertype) + "_Rs_" + str(resulttype) + "_Pr_" + \
                        str(preprocess) + "_Sm_" + str(isSmoothen) + \
                        "_Reg_" + region + "_C_" + str(WindowCount)
            objPlots.plotGraphAllWithoutTimewithParam(imagepath, imageName,
                                                      self.blue, self.green, self.red, self.grey, self.Irchannel,
                                                      "No of Frames", "Intensity")
            imageName = "Originaldata_AlgotypeAll_" + algotype + "_fft_" + fftype + "_Fl_" + str(
                filtertype) + "_Rs_" + str(
                resulttype) + "_Pr_" + \
                        str(preprocess) + "_Sm_" + str(isSmoothen) + \
                        "_Reg_" + region + "_C_" + str(WindowCount)
            objPlots.plotAllinOneWithoutTime(imagepath, imageName,
                                             self.blue, self.green, self.red, self.grey, self.Irchannel, "No of Frames",
                                             "Intensity")

        if (ignoregray):
            S = np.c_[self.blue, self.green, self.red, self.Irchannel]
        else:
            S = np.c_[self.blue, self.green, self.red, self.grey, self.Irchannel]

        # Add interpolation and smoothen of curve?
        if (preprocess == 1):
            # do nothing
            a = 0
        elif (preprocess == 2):
            self.blue = np.array(self.blue)
            Larray = len(self.blue)
            processed = signal.detrend(self.blue)  # detrend the signal to avoid interference of light change
            even_times = np.linspace(timecolorcountLips[0], timecolorcountLips[-1], Larray)
            interpolated = np.interp(even_times, timecolorcountLips, processed)  # interpolation by 1
            interpolated = np.hamming(
                Larray) * interpolated  # make the signal become more periodic (advoid spectral leakage)
            self.blue = interpolated / np.linalg.norm(interpolated)

            self.green = np.array(self.green)
            Larray = len(self.green)
            processed = signal.detrend(self.green)  # detrend the signal to avoid interference of light change
            even_times = np.linspace(timecolorcountLips[0], timecolorcountLips[-1], Larray)
            interpolated = np.interp(even_times, timecolorcountLips, processed)  # interpolation by 1
            interpolated = np.hamming(
                Larray) * interpolated  # make the signal become more periodic (advoid spectral leakage)
            self.green = interpolated / np.linalg.norm(interpolated)

            self.red = np.array(self.red)
            Larray = len(self.red)
            processed = signal.detrend(self.red)  # detrend the signal to avoid interference of light change
            even_times = np.linspace(timecolorcountLips[0], timecolorcountLips[-1], Larray)
            interpolated = np.interp(even_times, timecolorcountLips, processed)  # interpolation by 1
            interpolated = np.hamming(
                Larray) * interpolated  # make the signal become more periodic (advoid spectral leakage)
            self.red = interpolated / np.linalg.norm(interpolated)

            if (not ignoregray):
                self.grey = np.array(self.grey)
                Larray = len(self.grey)
                processed = signal.detrend(self.grey)  # detrend the signal to avoid interference of light change
                even_times = np.linspace(timecolorcountLips[0], timecolorcountLips[-1], Larray)
                interpolated = np.interp(even_times, timecolorcountLips, processed)  # interpolation by 1
                interpolated = np.hamming(
                    Larray) * interpolated  # make the signal become more periodic (advoid spectral leakage)
                self.grey = interpolated / np.linalg.norm(interpolated)

            self.Irchannel = np.array(self.Irchannel)
            Larray = len(self.Irchannel)
            processed = signal.detrend(self.Irchannel)  # detrend the signal to avoid interference of light change
            even_times = np.linspace(timeircountLips[0], timeircountLips[-1], Larray)
            interpolated = np.interp(even_times, timeircountLips, processed)  # interpolation by 1
            interpolated = np.hamming(
                Larray) * interpolated  # make the signal become more periodic (advoid spectral leakage)
            self.Irchannel = interpolated / np.linalg.norm(interpolated)

        elif (preprocess == 3):
            self.blue = self.preprocessdata(np.array(self.blue), timecolorcountLips, True)
            self.green = self.preprocessdata(np.array(self.green), timecolorcountLips, True)
            self.red = self.preprocessdata(np.array(self.red), timecolorcountLips, True)
            if (not ignoregray):
                self.grey = self.preprocessdata(np.array(self.grey), timecolorcountLips, True)
            self.Irchannel = self.preprocessdata(np.array(self.Irchannel), timeircountLips, True)

        elif (preprocess == 4):  ##set fft and freq data accordingly
            self.blue = self.preprocessdata2(np.array(self.blue), timecolorcountLips, True)
            self.green = self.preprocessdata2(np.array(self.green), timecolorcountLips, True)
            self.red = self.preprocessdata2(np.array(self.red), timecolorcountLips, True)
            if (not ignoregray):
                self.grey = self.preprocessdata2(np.array(self.grey), timecolorcountLips, True)
            self.Irchannel = self.preprocessdata2(np.array(self.Irchannel), timeircountLips, True)

        elif (preprocess == 5):

            S = preprocessing.normalize(S)

            self.blue = S[:, 0]
            self.green = S[:, 1]
            self.red = S[:, 2]
            index = 3
            if (not ignoregray):
                self.grey = S[:, 3]
                index = 4
            self.Irchannel = S[:, index]

        elif (preprocess == 6):
            self.blue = self.preprocessdata(np.array(self.blue), timecolorcountLips, False)
            self.green = self.preprocessdata(np.array(self.green), timecolorcountLips, False)
            self.red = self.preprocessdata(np.array(self.red), timecolorcountLips, False)
            if (not ignoregray):
                self.grey = self.preprocessdata(np.array(self.grey), timecolorcountLips, False)
            self.Irchannel = self.preprocessdata(np.array(self.Irchannel), timeircountLips, False)

        elif (preprocess == 7):  ##set fft and freq data accordingly
            self.blue = self.preprocessdata2(np.array(self.blue), timecolorcountLips, False)
            self.green = self.preprocessdata2(np.array(self.green), timecolorcountLips, False)
            self.red = self.preprocessdata2(np.array(self.red), timecolorcountLips, False)
            if (not ignoregray):
                self.grey = self.preprocessdata2(np.array(self.grey), timecolorcountLips, False)
            self.Irchannel = self.preprocessdata2(np.array(self.Irchannel), timeircountLips, False)

        if (ignoregray):
            S = np.c_[self.blue, self.green, self.red, self.Irchannel]
        else:
            S = np.c_[self.blue, self.green, self.red, self.grey, self.Irchannel]

        if (GenerateGraphs):

            imageName = "AfterPreProcessdataAll_Algotype_" + algotype + "_fft_" + fftype + "_Fl_" + str(
                filtertype) + "_Rs_" + str(resulttype) + "_Pr_" + \
                        str(preprocess) + "_Sm_" + str(isSmoothen) + \
                        "_Reg_" + region + "_C_" + str(WindowCount)

            objPlots.plotAllinOne(imagepath, S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4], imageName, fps, timeinSeconds,
                                  "Time(s)", "Amplitude")

            imageName = "AfterPreProcessdataTimeAll_Algotype_" + algotype + "_fft_" + fftype + "_Fl_" + str(
                filtertype) + "_Rs_" + str(resulttype) + "_Pr_" + \
                        str(preprocess) + "_Sm_" + str(isSmoothen) + \
                        "_Reg_" + region + "_C_" + str(WindowCount)

            objPlots.plotGraphAllwithParam(imagepath, imageName, timecolorLips,
                                           timeirLips,
                                           S[:, 0], S[:, 1], S[:, 2], S[:, 3], S[:, 4], "Time(s)",
                                           "Amplitude", fps, timeinSeconds)

        # # Apply ica
        S_ = self.ApplyAlgorithm(algotype, S, objAlgorithm, components)

        if (GenerateGraphs):

            imageName = "AfterAlgorithmTime_Algotype_" + algotype + "_fft_" + fftype + "_Fl_" + str(
                filtertype) + "_Rs_" + str(resulttype) + "_Pr_" + \
                        str(preprocess) + "_Sm_" + str(isSmoothen) + \
                        "_Reg_" + region + "_C_" + str(WindowCount)

            objPlots.plotGraphAllwithParam(imagepath, imageName, timecolorLips,
                                           timeirLips,
                                           S_[:, 0], S_[:, 1], S_[:, 2], S_[:, 3], S_[:, 4], "Time(s)",
                                           "Amplitude", fps, timeinSeconds)

            imageName = "AfterAlgorithmAllTime_Algotype_" + algotype + "_fft_" + fftype + "_Fl_" + str(
                filtertype) + "_Rs_" + str(resulttype) + "_Pr_" + \
                        str(preprocess) + "_Sm_" + str(isSmoothen) + \
                        "_Reg_" + region + "_C_" + str(WindowCount)

            objPlots.plotAllinOne(imagepath, S_[:, 0], S_[:, 1], S_[:, 2], S_[:, 3], S_[:, 4], imageName, fps,
                                  timeinSeconds, "Time(s)",
                                  "Amplitude")

        # Apply smoothen only before fft
        if (isSmoothen):
            ##Smooth data
            self.blue = self.smooth(S_[:, 0])
            self.green = self.smooth(S_[:, 1])
            self.red = self.smooth(S_[:, 2])
            if (not ignoregray):
                self.grey = self.smooth(S_[:, components - 2])
            self.Irchannel = self.smooth(S_[:, components - 1])

            if (ignoregray):
                S_ = np.c_[self.blue, self.green, self.red, self.Irchannel]
            else:
                S_ = np.c_[self.blue, self.green, self.red, self.grey, self.Irchannel]

            if (GenerateGraphs):
                imageName = "AfterSmoothTimeAll_Algotype_" + algotype + "_fft_" + fftype + "_Fl_" + str(
                    filtertype) + "_Rs_" + str(
                    resulttype) + "_Pr_" + \
                            str(preprocess) + "_Sm_" + str(isSmoothen) + \
                            "_Reg_" + region + "_C_" + str(WindowCount)

                objPlots.plotAllinOne(imagepath,
                                      S_[:, 0], S_[:, 1], S_[:, 2], S_[:, 3], S_[:, 4], imageName, fps, timeinSeconds,
                                      "Time(s)",
                                      "Amplitude")

                imageName = "AfterSmoothTime_Algotype_" + algotype + "_fft_" + fftype + "_Fl_" + str(
                    filtertype) + "_Rs_" + str(
                    resulttype) + "_Pr_" + \
                            str(preprocess) + "_Sm_" + str(isSmoothen) + \
                            "_Reg_" + region + "_C_" + str(WindowCount)

                objPlots.plotGraphAllwithParam(imagepath, imageName, timecolorLips,
                                               timeirLips,
                                               S_[:, 0], S_[:, 1], S_[:, 2], S_[:, 3], S_[:, 4], "Time(s)",
                                               "Amplitude", fps, timeinSeconds)

        # Apply fft
        B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, frequency, \
        blue_max_peak, green_max_peak, red_max_peak, grey_max_peak, ir_max_peak, \
        blue_bpm, green_bpm, red_bpm, grey_bpm, ir_bpm = self.ApplyFFT(fftype, S_, objAlgorithm, objPlots, ignoregray)

        if (GenerateGraphs):
            imageName = "AfterFFT_Algotype_" + algotype + "_fft_" + fftype + "_Fl_" + str(filtertype) + "_Rs_" + str(
                resulttype) + "_Pr_" + \
                        str(preprocess) + "_Sm_" + str(isSmoothen) + \
                        "_Reg_" + region + "_C_" + str(WindowCount)
            objPlots.PlotFFT(B_fft, Gr_fft, R_fft, Gy_fft, IR_fft, frequency, imageName, imagepath, imageName)

        B_bp, G_bp, R_bp, Gy_bp, IR_bp = self.FilterTechniques(filtertype, IR_fft, Gy_fft, R_fft, Gr_fft, B_fft,
                                                               frequency, ignoregray)  ##Applyfiltering

        if (filtertype == 6):
            interest_idx = np.where((frequency >= self.ignore_freq_below) & (frequency <= self.ignore_freq_above))[0]
            interest_idx_sub = interest_idx[:-1].copy()  # advoid the indexing error
            frequency = frequency[interest_idx_sub]
        elif (filtertype == 7):
            interest_idx = np.where((frequency >= self.ignore_freq_below))[0]
            interest_idx_sub = interest_idx[:-1].copy()  # advoid the indexing error
            frequency = frequency[interest_idx_sub]

        if (GenerateGraphs):
            imageName = "AfterFilter_Algotype_" + algotype + "_fft_" + fftype + "_Fl_" + str(filtertype) + "_Rs_" + str(
                resulttype) + "_Pr_" + \
                        str(preprocess) + "_Sm_" + str(isSmoothen) + \
                        "_Reg_" + region + "_C_" + str(WindowCount)
            objPlots.PlotFFT(B_bp, G_bp, R_bp, Gy_bp, IR_bp, frequency, imageName, imagepath, imageName)

        NumSamples = len(frequency)
        if (resulttype == 1):
            self.RampStuff(NumSamples, 30, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency, ignoregray)
        elif (resulttype == 2):
            self.RampStuff6(NumSamples, 30, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency, ignoregray)
        elif (resulttype == 3):
            self.Ramp2(NumSamples, 30, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency, ignoregray)
        elif (resulttype == 4):
            self.Ramp3(NumSamples, 30, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency, ignoregray)
        elif (resulttype == 5):
            self.Ramp4(NumSamples, 30, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency, ignoregray)
        elif (resulttype == 6):
            self.Ramp5(NumSamples, 30, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency, ignoregray)
        elif (resulttype == 7):
            self.Ramp7(NumSamples, 30, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency, ignoregray)
        elif (resulttype == 8):
            self.Ramp8(NumSamples, 30, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency, ignoregray)
        elif (resulttype == 9):
            self.Ramp9(NumSamples, 30, IR_bp, Gy_bp, R_bp, G_bp, B_bp, frequency, ignoregray)

        self.bestBpm = 0
        self.bestHeartRateSnr = 0.0
        heartRatePeriod = self.GetBestBpm()

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
        windowList.regiontype = region
        windowList.IrFreqencySamplingError = self.IrFreqencySamplingError
        windowList.GreyFreqencySamplingError = self.GreyFreqencySamplingError
        windowList.RedFreqencySamplingError = self.RedFreqencySamplingError
        windowList.GreenFreqencySamplingError = self.GreenFreqencySamplingError
        windowList.BlueFreqencySamplingError = self.BlueFreqencySamplingError
        FileName = "HRdataSingle_Algotype_" + algotype + "_fft_" + fftype + "_Fl_" + str(filtertype) + "_Rs_" + str(
            resulttype) + "_Pr_" + \
                   str(preprocess) + "_Sm_" + str(isSmoothen) + \
                   "_Reg_" + region + "_C_" + str(WindowCount)


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
        filteredir = np.fft.ifft(self.Irchannel)
        filteredred = np.fft.ifft(self.red)
        std, err, oxylevl = self.CalcualateSPOWithout(filteredgrey, heartRatePeriod, filteredir, filteredred,
                                                      self.green,
                                                      distanceM, smallestOxygenError,
                                                      region)  # CalcualateSPOPart2 ,CalcualateSPOWithout

        oxygenSaturationValueError = self.OxygenSaturationError
        oxygenSaturationValueValue = self.OxygenSaturation

        windowList.oxygenSaturationSTD = 0  # std
        windowList.oxygenSaturationValueError = 0  # err
        windowList.oxygenSaturationValueValue = 0  # oxylevl

        return windowList


    def FindPeeks(self, arry):
        peeks = []
        # for every consecutive triple in samples
        for x in range(1, len(arry) - 1):
            if (arry[x - 1].real <= arry[x].real and arry[x].real >= arry[x + 1].real):
                peeks.append(x)

        return peeks

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

    def FindMaxPeekInPeriodPart2(self, indicesOfPeeks, samples, period, startIdx, freq_min, freqs, freq_max):
        maxPeek = float('-inf')
        maxPeekIdx = 0
        for x in range(0, len(indicesOfPeeks)):
            if freq_min <= freqs[x] <= freq_max:
                index = indicesOfPeeks[x]
                if (index >= startIdx and index < startIdx + period):  # make sure it's within the window
                    value = samples[index].real
                    if (value > maxPeek):
                        maxPeek = value
                        maxPeekIdx = index

        return maxPeekIdx
