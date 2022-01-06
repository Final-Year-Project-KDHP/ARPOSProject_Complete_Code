import numpy as np
from scipy import signal

from Configurations import Configurations
from FileIO import FileIO
from SingalToNoise import SNR


class ComputerHeartRate:
    # Constants
    ColorEstimatedFPS = 0
    IREstimatedFPS = 0
    components = 0
    IRIndex = 0
    grayIndex = 0
    ramp_end_bpm = 0
    ramp_start_percentage = 0.0
    ramp_end_percentage = 0
    ramp_end_hz = 0.0
    freq_bpmColor= []
    freq_bpmIr= []
    Colorfrequency= []
    IRfrequency= []
    ColorNumSamples = 0
    IRNumSamples = 0
    ignore_freq_index_below = 0
    ignore_freq_index_above = 0
    ramp_start = 0
    ramp_end = 0
    rampDesignLength = 0
    ramp_design = []
    ramplooprange = 0

    # setup highpass filter
    ignore_freq_below_bpm = 0
    ignore_freq_below = 0

    # setup low pass filter
    ignore_freq_above_bpm = 0
    ignore_freq_above = 0

    # Input Parameters
    ignoreGray = False
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
    SavePath =''
    region = ''

    # Constructor
    def __init__(self, snrType, ignoreGray,ColorNumSamples,grayIndex,IRIndex,components,ColorEstimatedFPS,IREstimatedFPS,ramp_end_bpm,ramp_start_percentage,
                 ramp_end_percentage,ignore_freq_below_bpm,ignore_freq_above_bpm,freq_bpmColor,freq_bpmIr,Colorfrequency,IRfrequency,SavePath,region,IRNumSamples):

        # Constants
        self.region = region
        self.SavePath =SavePath
        self.ColorEstimatedFPS = ColorEstimatedFPS
        self.IREstimatedFPS = IREstimatedFPS
        self.components = components
        self.IRIndex = IRIndex
        self.grayIndex = grayIndex
        self.ramp_end_bpm = ramp_end_bpm
        self.ramp_start_percentage = ramp_start_percentage
        self.ramp_end_percentage = ramp_end_percentage
        self.ramp_end_hz = self.ramp_end_bpm / 60
        self.freq_bpmColor = freq_bpmColor
        self.freq_bpmIr = freq_bpmIr
        self.Colorfrequency = Colorfrequency
        self.IRfrequency = IRfrequency
        self.ColorNumSamples = ColorNumSamples
        self.IRNumSamples =IRNumSamples
        self.ignore_freq_index_below = 0
        self.ignore_freq_index_above = 0
        self.ramp_start = 0
        self.ramp_end = 0
        self.rampDesignLength = 0
        self.ramp_design = []
        self.ramplooprange = 0

        # setup highpass filter
        self.ignore_freq_below_bpm = ignore_freq_below_bpm
        self.ignore_freq_below = self.ignore_freq_below_bpm / 60

        # setup low pass filter
        self.ignore_freq_above_bpm = ignore_freq_above_bpm
        self.ignore_freq_above = self.ignore_freq_above_bpm / 60

        # Input Parameters
        self.ignoreGray = ignoreGray
        self.snrType = snrType

        # ResultData
        self.IrSnr = 0.0
        self.GreySnr = 0.0
        self.RedSnr = 0.0
        self.GreenSnr = 0.0
        self.BlueSnr = 0.0
        self.IrBpm = 0.0
        self.GreyBpm = 0.0
        self.RedBpm = 0.0
        self.GreenBpm = 0.0
        self.BlueBpm = 0.0
        self.IrFreqencySamplingError = 0.0
        self.GreyFreqencySamplingError = 0.0
        self.RedFreqencySamplingError = 0.0
        self.GreenFreqencySamplingError = 0.0
        self.BlueFreqencySamplingError = 0.0

    def getHeartRate_fromFrequencyWithFilter(self,fftarray, freqs):
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

    def getHearRate_fromFrequencyWithFilter_Main(self, blue_fft_realabs,green_fft_realabs,red_fft_realabs,grey_fft_realabs,ir_fft_realabs):

        grey_fft_maxVal = 0
        grey_fft_index = 0

        irbpm, ir_fft_maxVal, ir_fft_index = self.getHeartRate_fromFrequencyWithFilter(ir_fft_realabs, self.IRfrequency)
        if (not self.ignoreGray):
            greybpm, grey_fft_maxVal, grey_fft_index = self.getHeartRate_fromFrequencyWithFilter(grey_fft_realabs, self.Colorfrequency)
        else:
            grey_fft_maxVal = 0
            grey_fft_index = 0
        redbpm, red_fft_maxVal, red_fft_index = self.getHeartRate_fromFrequencyWithFilter(red_fft_realabs, self.Colorfrequency)
        greenbpm, green_fft_maxVal, green_fft_index = self.getHeartRate_fromFrequencyWithFilter(green_fft_realabs, self.Colorfrequency)
        bluebpm, blue_fft_maxVal, blue_fft_index = self.getHeartRate_fromFrequencyWithFilter(blue_fft_realabs, self.Colorfrequency)

        #####GET BPM#####
        self.getBPM(ir_fft_index, grey_fft_index, red_fft_index, green_fft_index, blue_fft_index)

        #### GET SAMPLING ERROR ####
        self.getSamplingError(ir_fft_index, grey_fft_index, red_fft_index, green_fft_index, blue_fft_index)

        ### GET SNR ###
        SNR = self.getSNR(ir_fft_maxVal, ir_fft_realabs, grey_fft_maxVal, grey_fft_realabs, red_fft_maxVal,
                    red_fft_realabs, green_fft_maxVal, green_fft_realabs, blue_fft_maxVal, blue_fft_realabs)
        return SNR
        # self.WriteDetails(SNR)

    def WriteDetails(self, SNR):
        # Write to file
        objFileIO = FileIO()

        Bpm = 'IrBpm: ' + str(self.IrBpm) + ', GreyBpm: ' + str(self.GreyBpm) + ', RedBpm: ' + str(self.RedBpm) \
               + ', GreenBpm: ' + str(self.GreenBpm) + ', BlueBpm: ' + str(self.BlueBpm) + '\n'

        objFileIO.WritedatatoFile(self.SavePath, 'SNR_Data_' + self.region, SNR)
        objFileIO.WritedatatoFile(self.SavePath, 'Bpm_Data_' + self.region, Bpm)

        del objFileIO

    def getHeartRate_fromFrequency(self, blue_fft_realabs,green_fft_realabs,red_fft_realabs,grey_fft_realabs,ir_fft_realabs):

        IR_sig_max_peak = ir_fft_realabs.argmax()  # Find its location [np.abs(self.frequency) <= self.ignore_freq_above]
        R_sig_max_peak = red_fft_realabs.argmax()  # Find its location [np.abs(self.frequency) <= self.ignore_freq_above]
        G_sig_max_peak = green_fft_realabs.argmax()  # Find its location [np.abs(self.frequency) <= self.ignore_freq_above]
        B_sig_max_peak = blue_fft_realabs.argmax()  # Find its location [np.abs(self.frequency) <= self.ignore_freq_above]

        Gy_sig_max_peak = 0
        Gy_freqValueAtPeak = 0

        if (not self.ignoreGray):
            Gy_sig_max_peak = grey_fft_realabs.argmax()  # Find its location [np.abs(self.frequency)  <= self.ignore_freq_above]
            Gy_freqValueAtPeak = self.Colorfrequency[Gy_sig_max_peak]  # Get the actual frequency value
            #####GET BPM#####
            # self.GreyBpm = np.abs(self.frequency)[grey_fft_realabs[np.abs(self.frequency)  <= self.ignore_freq_above].argmax()] * 60
        else:
            self.GreySnr=0.0
            self.GreyBpm=0

        IR_freqValueAtPeak = self.IRfrequency[IR_sig_max_peak]  # Get the actual frequency value
        R_freqValueAtPeak = self.Colorfrequency[R_sig_max_peak]  # Get the actual frequency value
        G_freqValueAtPeak = self.Colorfrequency[G_sig_max_peak]  # Get the actual frequency value
        B_freqValueAtPeak = self.Colorfrequency[B_sig_max_peak]  # Get the actual frequency value

        #####GET BPM#####
        self.getBPM(IR_sig_max_peak, Gy_sig_max_peak, R_sig_max_peak, G_sig_max_peak, B_sig_max_peak)

        #####TODO: Comprared and result is same as method above
        # self.BlueBpm = np.abs(self.frequency)[blue_fft_realabs[np.abs(self.frequency) <= self.ignore_freq_above].argmax()] * 60
        # self.GreenBpm = np.abs(self.frequency)[green_fft_realabs[np.abs(self.frequency) <= self.ignore_freq_above].argmax()] * 60
        # self.RedBpm = np.abs(self.frequency)[red_fft_realabs[np.abs(self.frequency) <= self.ignore_freq_above].argmax()] * 60
        # self.IrBpm = np.abs(self.frequency)[ir_fft_realabs[np.abs(self.frequency) <= self.ignore_freq_above].argmax()] * 60
        # self.GreyBpm = np.abs(self.frequency)[grey_fft_realabs[np.abs(self.frequency) <= self.ignore_freq_above].argmax()] * 60

        ### GET SNR ###
        SNR =self.getSNR(IR_freqValueAtPeak, ir_fft_realabs, Gy_freqValueAtPeak, grey_fft_realabs, R_freqValueAtPeak,
                    red_fft_realabs, G_freqValueAtPeak, green_fft_realabs, B_freqValueAtPeak, blue_fft_realabs)

        #### GET SAMPLING ERROR ####
        self.getSamplingError(IR_sig_max_peak, Gy_sig_max_peak, R_sig_max_peak, G_sig_max_peak, B_sig_max_peak)
        return SNR
        # self.WriteDetails(SNR)

    def OriginalARPOSmethod(self, B_fft_Copy, G_fft_Copy, R_fft_Copy, Gy_fft_Copy, IR_fft_Copy):

        # region COLOR
        # ##FOR COLOR###
        # define index for the low and high pass frequency filter in frequency space that has just been created
        # (depends on number of samples).
        self.ignore_freq_index_below = np.rint(((self.ignore_freq_below * self.ColorNumSamples) / self.ColorEstimatedFPS))  # high pass
        self.ignore_freq_index_above = np.rint(((self.ignore_freq_above * self.ColorNumSamples) / self.ColorEstimatedFPS))  # low pass

        # compute the ramp filter start and end indices double
        self.ramp_start = self.ignore_freq_index_below
        self.ramp_end = np.rint(((self.ramp_end_hz * self.ColorNumSamples) / self.ColorEstimatedFPS))
        self.rampDesignLength = self.ignore_freq_index_above - self.ignore_freq_index_below
        self.ramp_design = [None] * int(self.rampDesignLength)
        self.ramplooprange = int(self.ramp_end - self.ramp_start)

        # setup linear ramp
        for x in range(0, self.ramplooprange):
            self.ramp_design[x] = ((((self.ramp_end_percentage - self.ramp_start_percentage) / (self.ramp_end - self.ramp_start)) * (x)) + self.ramp_start_percentage)

        # setup plateu of linear ramp
        for x in range(int(self.ramp_end - self.ramp_start), int(self.ignore_freq_index_above - self.ignore_freq_index_below)):
            # ramp_design.append(1)
            self.ramp_design[x] = 1

        # apply ramp filter and find index of maximum frequency (after filter is applied).
        # nullable so this works even if you have all super-low negatives
        # Calculate and process signal data
        grey_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
        grey_fft_index = -1
        grey_fft_realabs = [0] * (int(self.ignore_freq_index_above) - int(self.ignore_freq_index_below))

        red_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
        red_fft_index = -1
        red_fft_realabs = [0] * (int(self.ignore_freq_index_above) - int(self.ignore_freq_index_below))

        green_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
        green_fft_index = -1
        green_fft_realabs = [0] * (int(self.ignore_freq_index_above) - int(self.ignore_freq_index_below))

        blue_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
        blue_fft_index = -1
        blue_fft_realabs = [0] * (int(self.ignore_freq_index_above) - int(self.ignore_freq_index_below))

        realabs_i = 0

        for x in range((int(self.ignore_freq_index_below + 1)), (int(self.ignore_freq_index_above + 1))):
            # "apply" the ramp to generate the shaped frequency values for IR
            # find the max value and the index of the max value.
            if (not self.ignoreGray):
                # "apply" the ramp to generate the shaped frequency values for Grey
                current_greyNum = self.ramp_design[realabs_i] * np.abs(Gy_fft_Copy[x].real)
                grey_fft_realabs[realabs_i] = current_greyNum
                if ((grey_fft_maxVal is None) or (current_greyNum > grey_fft_maxVal)):
                    grey_fft_maxVal = current_greyNum
                    grey_fft_index = x

            # "apply" the ramp to generate the shaped frequency values for Red
            current_redNum = self.ramp_design[realabs_i] * np.abs(R_fft_Copy[x].real)
            red_fft_realabs[realabs_i] = current_redNum
            if ((red_fft_maxVal is None) or (current_redNum > red_fft_maxVal)):
                red_fft_maxVal = current_redNum
                red_fft_index = x

            # "apply" the ramp to generate the shaped frequency values for Green
            current_greenNum = self.ramp_design[realabs_i] * np.abs(G_fft_Copy[x].real)
            green_fft_realabs[realabs_i] = current_greenNum
            if ((green_fft_maxVal is None) or (current_greenNum > green_fft_maxVal)):
                green_fft_maxVal = current_greenNum
                green_fft_index = x

            # "apply" the ramp to generate the shaped frequency values for blue
            current_blueNum = self.ramp_design[realabs_i] * np.abs(B_fft_Copy[x].real)
            blue_fft_realabs[realabs_i] = current_blueNum
            if ((blue_fft_maxVal is None) or (current_blueNum > blue_fft_maxVal)):
                blue_fft_maxVal = current_blueNum
                blue_fft_index = x

            realabs_i = realabs_i + 1

        # endregion

        # region IR
        #### FOR IR
        # define index for the low and high pass frequency filter in frequency space that has just been created
        # (depends on number of samples).
        self.ignore_freq_index_below = np.rint(
            ((self.ignore_freq_below * self.IRNumSamples) / self.IREstimatedFPS))  # high pass
        self.ignore_freq_index_above = np.rint(
            ((self.ignore_freq_above * self.IRNumSamples) / self.IREstimatedFPS))  # low pass

        # compute the ramp filter start and end indices double
        self.ramp_start = self.ignore_freq_index_below
        self.ramp_end = np.rint(((self.ramp_end_hz * self.IRNumSamples) / self.IREstimatedFPS))
        self.rampDesignLength = self.ignore_freq_index_above - self.ignore_freq_index_below
        self.ramp_design = [None] * int(self.rampDesignLength)
        self.ramplooprange = int(self.ramp_end - self.ramp_start)

        # setup linear ramp
        for x in range(0, self.ramplooprange):
            self.ramp_design[x] = ((((self.ramp_end_percentage - self.ramp_start_percentage) / (
                        self.ramp_end - self.ramp_start)) * (x)) + self.ramp_start_percentage)

        # setup plateu of linear ramp
        for x in range(int(self.ramp_end - self.ramp_start),
                       int(self.ignore_freq_index_above - self.ignore_freq_index_below)):
            # ramp_design.append(1)
            self.ramp_design[x] = 1

        # apply ramp filter and find index of maximum frequency (after filter is applied).
        # nullable so this works even if you have all super-low negatives
        # Calculate and process signal data
        ir_fft_maxVal = None  # nullable so this works even if you have all super-low negatives
        ir_fft_index = -1
        ir_fft_realabs = [0] * (int(self.ignore_freq_index_above) - int(self.ignore_freq_index_below))

        realabs_i = 0

        for x in range((int(self.ignore_freq_index_below + 1)), (int(self.ignore_freq_index_above + 1))):
            # "apply" the ramp to generate the shaped frequency values for IR
            # find the max value and the index of the max value.
            current_irNum = self.ramp_design[realabs_i] * np.abs(IR_fft_Copy[x].real)
            ir_fft_realabs[realabs_i] = current_irNum
            if ((ir_fft_maxVal is None) or (current_irNum > ir_fft_maxVal)):
                ir_fft_maxVal = current_irNum
                ir_fft_index = x

            realabs_i = realabs_i + 1

        # endregion

        #####GET BPM#####
        self.getBPM(ir_fft_index, grey_fft_index, red_fft_index, green_fft_index, blue_fft_index)

        #### GET SAMPLING ERROR ####
        self.getSamplingError(ir_fft_index, grey_fft_index, red_fft_index, green_fft_index, blue_fft_index)

        ### GET SNR ###
        SNR= self.getSNR(ir_fft_maxVal, ir_fft_realabs, grey_fft_maxVal, grey_fft_realabs, red_fft_maxVal,
                    red_fft_realabs, green_fft_maxVal, green_fft_realabs, blue_fft_maxVal, blue_fft_realabs)
        # BpmSummary = 'IrBpm: ' + str(self.IrBpm) + ', GreyBpm: ' + str(self.GreyBpm) + ', RedBpm: ' + str(self.RedBpm) \
        #       + ', GreenBpm: ' + str(self.GreenBpm) + ', BlueBpm: ' + str(self.BlueBpm) + '\n'

        # self.WriteDetails(SNR)
        return  SNR
    '''
    getBPM:
    '''
    def getBPM(self,ir_fft_index,grey_fft_index,red_fft_index,green_fft_index,blue_fft_index):

        self.IrBpm = self.freq_bpmIr[ir_fft_index]
        if (not self.ignoreGray):
            self.GreyBpm = self.freq_bpmColor[grey_fft_index]
        else:
            self.GreyBpm = 0
        self.RedBpm = self.freq_bpmColor[red_fft_index]
        self.GreenBpm = self.freq_bpmColor[green_fft_index]
        self.BlueBpm = self.freq_bpmColor[blue_fft_index]

    '''
    getSNR:
    '''
    def getSNR(self, ir_fft_maxVal, ir_fft_realabs, grey_fft_maxVal, grey_fft_realabs, red_fft_maxVal,
               red_fft_realabs, green_fft_maxVal, green_fft_realabs, blue_fft_maxVal, blue_fft_realabs):

        objSNR = SNR()

        # if(self.snrType == 1):
        # if (self.snrType == 2):
        #Test and write old
        self.IrSnr = objSNR.signaltonoiseDB(ir_fft_realabs) *1
        if (not self.ignoreGray):
            self.GreySnr = objSNR.signaltonoiseDB(grey_fft_realabs)
        else:
            self.GreySnr = 0.0
        self.RedSnr= objSNR.signaltonoiseDB(red_fft_realabs)
        self.GreenSnr= objSNR.signaltonoiseDB(green_fft_realabs)
        self.BlueSnr= objSNR.signaltonoiseDB(blue_fft_realabs)

        SNR1 = 'IrSnr2: ' + str(self.IrSnr) + ', GreySnr2: ' + str(self.GreySnr) + ', RedSnr2: ' + str(self.RedSnr) \
               + ', GreenSnr2: ' + str(self.GreenSnr) + ', BlueSnr2: ' + str(self.BlueSnr) + '\n'

        #Override previous and take following as final
        self.IrSnr = float(ir_fft_maxVal) / np.average(
            ir_fft_realabs) * 1  # * 1 # could artificially increase SNR for IR as provdes higher accuracy readings, enabling higher weighting for readings
        if (not self.ignoreGray):
            self.GreySnr = float(grey_fft_maxVal) / np.average(grey_fft_realabs)
        else:
            self.GreySnr = 0.0
        self.RedSnr = float(red_fft_maxVal) / np.average(red_fft_realabs)
        self.GreenSnr = float(green_fft_maxVal) / np.average(green_fft_realabs)
        self.BlueSnr = float(blue_fft_maxVal) / np.average(blue_fft_realabs)

        SNR2 = 'IrSnr: ' + str(self.IrSnr) + ', GreySnr: ' + str(self.GreySnr) + ', RedSnr: ' + str(self.RedSnr) \
               + ', GreenSnr: ' + str(self.GreenSnr) + ', BlueSnr: ' + str(self.BlueSnr)

        return SNR1 + SNR2
    '''
    getSamplingError:
    '''
    def getSamplingError(self, ir_fft_index, grey_fft_index, red_fft_index, green_fft_index, blue_fft_index):
        #TODO: Porbably due to different fps of color and IR
        if(((ir_fft_index +1) >= len(self.freq_bpmIr))):
            ir_fft_index = ir_fft_index -1
            print('ir_fft_index reduced by 1 : ' + str(ir_fft_index))

        if(((grey_fft_index +1) >= len(self.freq_bpmColor))):
            grey_fft_index = grey_fft_index -1
            print('grey_fft_index reduced by 1 : ' + str(grey_fft_index))

        if(((red_fft_index +1) >= len(self.freq_bpmColor))):
            red_fft_index = red_fft_index -1
            print('red_fft_index reduced by 1 : ' + str(red_fft_index))

        if(((green_fft_index +1) >= len(self.freq_bpmColor))):
            green_fft_index = green_fft_index -1
            print('green_fft_index reduced by 1 : ' + str(green_fft_index))

        if((blue_fft_index +1) >= len(self.freq_bpmColor)):
            blue_fft_index = blue_fft_index -1
            print('blue_fft_index reduced by 1 : ' + str(blue_fft_index))

        self.IrFreqencySamplingError = self.freq_bpmIr[ir_fft_index + 1] - self.freq_bpmIr[ir_fft_index - 1]
        if (not self.ignoreGray):
            self.GreyFreqencySamplingError = self.freq_bpmColor[grey_fft_index + 1] - self.freq_bpmColor[grey_fft_index - 1]
        else:
            self.GreyFreqencySamplingError = 0.0
        self.RedFreqencySamplingError = self.freq_bpmColor[red_fft_index + 1] - self.freq_bpmColor[red_fft_index - 1]
        self.GreenFreqencySamplingError = self.freq_bpmColor[green_fft_index + 1] - self.freq_bpmColor[green_fft_index - 1]
        self.BlueFreqencySamplingError = self.freq_bpmColor[blue_fft_index + 1] - self.freq_bpmColor[blue_fft_index - 1]