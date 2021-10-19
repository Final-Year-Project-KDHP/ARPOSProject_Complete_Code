import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import FastICA, PCA
from scipy import signal
from sklearn.preprocessing import StandardScaler
from scipy.fftpack import fft, ifft
from scipy.fft import fft, fftfreq, fftshift
import scipy.fftpack as fftpack

from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
import jade

from SaveGraphs import Plots

from numpy import abs, append, arange, arctan2, argsort, array, concatenate, \
    cos, diag, dot, eye, float32, float64, loadtxt, matrix, multiply, ndarray, \
    newaxis, savetxt, sign, sin, sqrt, zeros
from numpy.linalg import eig, pinv
from scipy.sparse import *

class AlgorithmCollection:
    #constatns
    S_ =[]
    X = []

    objPlots = Plots()
    #FfT
    fftblue = [] # blue
    fftGreen = [] # green
    fftRed = [] # red
    fftGrey = [] # gr
    fftIR = [] # ir

    SavefigPath = ''

    fft_MaxIndex=0
    fft_MaxVal=0
    fft_frq=0

    freq = []
    EstimatedFPS = 30
    NumberofSample = 0

    powerfftblue = []
    powerfftGreen = []
    powerfftRed = []
    powerfftGrey = []
    powerfftIR = []

    def Detrend(self,P):
        P -= P.mean(axis=0)  # SUBTRACT mean
        P /= P.std(axis=0)  # Standardize data
        P = signal.detrend(P)
        return P

    #########------FFT------#############

    def apply_rfft_All(self, S_):
        y = S_
        yf = np.fft.rfft(y)
        return yf

    def apply_rfft(self, S_,ignoregray):
        self.fftblue = np.fft.rfft(S_[:, 0])  # blue
        self.fftGreen = np.fft.rfft(S_[:, 1])  # green
        self.fftRed = np.fft.rfft(S_[:, 2])  # red
        index =3
        if(not ignoregray):
            self.fftGrey = np.fft.rfft(S_[:, 3])  # gr
            index =4  # ir
        self.fftIR = np.fft.rfft(S_[:, index])  # ir

    def apply_fft(self, S_,ignoregray):
        self.fftblue = np.fft.fft(S_[:, 0])  # blue
        self.fftGreen = np.fft.fft(S_[:, 1])  # green
        self.fftRed = np.fft.fft(S_[:, 2])  # red
        index = 3
        if(not ignoregray):
            self.fftGrey = np.fft.fft(S_[:, 3])  # gr
            index =4  # ir
        self.fftIR = np.fft.fft(S_[:, index])  # ir

    def apply_sqrt_AfterFFt(self,ignoregray):
        self.fftblue = self.fftblue / np.sqrt(len(self.fftblue))
        self.fftGreen = self.fftGreen / np.sqrt(len(self.fftGreen))
        self.fftRed = self.fftRed / np.sqrt(len(self.fftRed))
        if(not ignoregray):
            self.fftGrey = self.fftGrey / np.sqrt(len(self.fftGrey))
        self.fftIR = self.fftIR / np.sqrt(len(self.fftIR))

    def apply_fft_All(self, S_):
        y = S_
        yf = np.fft.fft(y)
        return yf

    def apply_Abs_All(self,S):
        S = np.abs(S)
        return S

    def apply_Abs(self,ignoregray):
        self.fftblue = np.abs(self.fftblue)  # blue
        self.fftGreen = np.abs(self.fftGreen)  # green
        self.fftRed = np.abs(self.fftRed )  # red
        if(not ignoregray):
            self.fftGrey = np.abs(self.fftGrey)  # gr
        self.fftIR = np.abs(self.fftIR)  # ir

    def getPower_All(self,S):
        S = np.abs(S) ** 2
        return S

    def getPower(self,ignoregray):
        self.powerfftblue = np.abs(self.fftblue) ** 2  # blue
        self.powerfftGreen = np.abs(self.fftGreen) ** 2  # green
        self.powerfftRed = np.abs(self.fftRed) ** 2  # red
        if(not ignoregray):
            self.powerfftGrey = np.abs(self.fftGrey) ** 2  # gr
        self.powerfftIR = np.abs(self.fftIR) ** 2  # ir

    def get_Freq_bySamplesize(self,N,T):
        xf = np.fft.fftfreq(N, T)[:N // 2]
        return xf

    def get_Freq(self,N):
        time_step = 1 / self.EstimatedFPS
        self.freq = np.fft.fftfreq(N, d=time_step)
        return self.freq

    def FilterSignals(self,cutoff,upperbound, blue,green,red,grey,ir):

        blue_Copy= blue
        green_Copy= green
        red_Copy= red
        grey_Copy= grey
        Ir_Copy= ir

        blue_Copy[np.abs(self.freq) <= cutoff ] = 0
        blue_Copy[np.abs(self.freq) >= upperbound ] = 0

        green_Copy[np.abs(self.freq) <= cutoff ] = 0
        green_Copy[np.abs(self.freq) >= upperbound ] = 0

        red_Copy[np.abs(self.freq) <= cutoff ] = 0
        red_Copy[np.abs(self.freq) >= upperbound ] = 0

        grey[np.abs(self.freq) <= cutoff ] = 0
        grey[np.abs(self.freq) >= upperbound ] = 0

        Ir_Copy[np.abs(self.freq) <= cutoff ] = 0
        Ir_Copy[np.abs(self.freq) >= upperbound ] = 0

        blue_max_peak = blue_Copy[np.abs(self.freq)  <= upperbound].argmax()
        blue_max_peak_value = np.abs(self.freq)[blue_Copy[np.abs(self.freq)  <= upperbound].argmax()]
        blue_bpm = np.abs(self.freq)[blue_Copy[np.abs(self.freq)  <= upperbound].argmax()] * 60
        #frqY = np.abs(self.freq)[blue_max_peak]

        green_max_peak = green_Copy[np.abs(self.freq)  <= upperbound].argmax()
        green_max_peak_value= np.abs(self.freq)[green_Copy[np.abs(self.freq)  <= upperbound].argmax()]
        green_bpm = np.abs(self.freq)[green_Copy[np.abs(self.freq)  <= upperbound].argmax()] * 60

        red_max_peak = red_Copy[np.abs(self.freq)  <= upperbound].argmax()
        red_max_peak_value = np.abs(self.freq)[red_Copy[np.abs(self.freq)  <= upperbound].argmax()]
        red_bpm = np.abs(self.freq)[red_Copy[np.abs(self.freq)  <= upperbound].argmax()] * 60

        grey_max_peak = grey_Copy[np.abs(self.freq)  <= upperbound].argmax()
        grey_max_peak_value = np.abs(self.freq)[grey_Copy[np.abs(self.freq)  <= upperbound].argmax()]
        grey_bpm = np.abs(self.freq)[grey_Copy[np.abs(self.freq)  <= upperbound].argmax()] * 60

        ir_max_peak = Ir_Copy[np.abs(self.freq)  <= upperbound].argmax()
        ir_max_peak_value = np.abs(self.freq)[Ir_Copy[np.abs(self.freq)  <= upperbound].argmax()]
        ir_bpm = np.abs(self.freq)[Ir_Copy[np.abs(self.freq)  <= upperbound].argmax()] * 60

        return blue_Copy, green_Copy, red_Copy, grey_Copy, Ir_Copy, \
               blue_max_peak, green_max_peak,red_max_peak,grey_max_peak,ir_max_peak, \
               blue_max_peak_value, green_max_peak_value, red_max_peak_value, grey_max_peak_value, ir_max_peak_value, \
               blue_bpm,green_bpm,red_bpm,grey_bpm,ir_bpm

    def Apply_FFT_M1_byeachsignal(self,yf,ignoregray): #rfft, abs, get power and its freq

        self.apply_rfft(yf,ignoregray)
        self.apply_Abs(ignoregray)
        self.getPower(ignoregray)
        self.freq = np.fft.rfftfreq(len(yf),1/self.EstimatedFPS)

        # self.powerfftblue,self.powerfftGreen,\
        # self.powerfftRed,self.powerfftGrey,self.powerfftIR, blue_max_peak, \
        # green_max_peak,red_max_peak,grey_max_peak,ir_max_peak,blue_bpm,green_bpm,red_bpm,\
        # grey_bpm,ir_bpm = self.FilterSignals(cutoff, upperbound, self.powerfftblue,
        #                                      self.powerfftGreen,self.powerfftRed,self.powerfftGrey,
        #                                      self.powerfftIR)

        # objPlot.PlotFFT(self.powerfftblue,self.powerfftGreen,self.powerfftRed,self.powerfftGrey,self.powerfftIR, self.freq, "freq", "Savefilepath", "filename")

        # return self.powerfftblue,self.powerfftGreen,self.powerfftRed,self.powerfftGrey,self.powerfftIR,self.freq,\
        #        blue_max_peak, green_max_peak,red_max_peak,grey_max_peak,ir_max_peak,\
        #        blue_bpm,green_bpm,red_bpm,grey_bpm,ir_bpm
        return self.powerfftblue, self.powerfftGreen, self.powerfftRed, self.powerfftGrey, self.powerfftIR, self.freq

        #fft_plot, xf[max_peak] * 60, xf[max_peak], max_peak

    def Apply_FFT_forpreprocessed(self,yf,ignoregray): #rfft, abs, get power and its freq
        '''Apply Fast Fourier Transform'''
        L = len(yf)
        self.freq = float(self.EstimatedFPS) / L * np.arange(L / 2 + 1)

        # self.fftblue =S_[:, 0]
        raw_fft = np.fft.rfft(yf[:, 0] * 30)
        self.fftblue = np.abs(raw_fft) ** 2

        raw_fft = np.fft.rfft(yf[:, 1] * 30)
        self.fftGreen = np.abs(raw_fft) ** 2

        raw_fft = np.fft.rfft(yf[:, 2] * 30)
        self.fftRed = np.abs(raw_fft) ** 2

        index=3
        if(not ignoregray):
            raw_fft = np.fft.rfft(yf[:, 3] * 30)
            self.fftGrey = np.abs(raw_fft) ** 2
            index=4
        else:
            self.fftGrey = []

        raw_fft = np.fft.rfft(yf[:, index] * 30)
        self.fftIR = np.abs(raw_fft) ** 2

        return self.fftblue, self.fftGreen, self.fftRed, self.fftGrey, self.fftIR, self.freq

        #fft_plot, xf[max_peak] * 60, xf[max_peak], max_peak

    def Apply_FFT_M2_eachsignal(self,S_, ignoregray): #rfft, abs, get power and its freq


        N = len(S_)  # Number of data points
        T = 1. / self.EstimatedFPS  # delta between frames (s)
        '''perform fourier transform'''
        self.fftblue = fftpack.fft(S_[:, 0])  # blue
        self.fftblue = 2.0 / N * np.abs(self.fftblue[0:N // 2])

        self.fftGreen = fftpack.fft(S_[:, 1])  # green
        self.fftGreen = 2.0 / N * np.abs(self.fftGreen[0:N // 2])

        self.fftRed = fftpack.fft(S_[:, 2])  # red
        self.fftRed = 2.0 / N * np.abs(self.fftRed[0:N // 2])

        index = 3
        if (not ignoregray):
            self.fftGrey = fftpack.fft(S_[:, 3])  # gr
            self.fftGrey = 2.0 / N * np.abs(self.fftGrey[0:N // 2])
            index = 4  # ir

        self.fftIR = fftpack.fft(S_[:, index])  # ir
        self.fftIR = 2.0 / N * np.abs(self.fftIR[0:N // 2])

        self.freq = np.linspace(0.0, 1 / (T * 2), N // 2)  # replot complex data over freq domain

        return self.fftblue, self.fftGreen, self.fftRed, self.fftGrey, self.fftIR, self.freq

        #fft_plot, xf[max_peak] * 60, xf[max_peak], max_peak

    def Apply_FFT_WithoutPower_M4_eachsignal(self,yf,ignoregray): #rfft, abs, get power and its freq

        self.apply_rfft(yf,ignoregray)
        self.apply_Abs(ignoregray)
        self.freq = np.fft.rfftfreq(len(yf),1/self.EstimatedFPS)

        # freqfft = [float(np.argmax(self.fftblue)), float(np.argmax(self.fftGreen)), float(np.argmax(self.fftRed)), float(np.argmax(self.fftIR))] ## for gray add
        return self.fftblue, self.fftGreen, self.fftRed, self.fftGrey, self.fftIR, self.freq

        #fft_plot, xf[max_peak] * 60, xf[max_peak], max_peak


    def Apply_FFT_M2(self,yf): #rfft, abs, get power and its freq

        N = len(yf)
        T = 1 / self.EstimatedFPS  # timestep

        yf = self.fft_All(yf)
        xf = self.get_Freq_bySamplesize(N,T)

        yf = 2.0 / N * np.abs(yf[0:N // 2])

        mY = np.abs(yf)  # Find magnitude
        peakY = np.max(mY)  # Find max peak
        locY = np.argmax(mY)  # Find its location
        frqY = xf[locY]  # Get the actual frequency value

        return yf, frqY * 60, frqY, locY

    def Apply_FFT_M3(self,yf): #rfft, abs, get power and its freq

        N = len(yf)
        T = 1 / self.EstimatedFPS  # timestep

        yf = self.apply_rfft_All(yf)
        yf = self.apply_Abs_All(yf)
        xf = self.get_Freq_bySamplesize(N,T)

        #yf = 2.0 / N * np.abs(yf[0:N // 2])
        yf = yf[0:len(xf)]

        mY = np.abs(yf)  # Find magnitude
        peakY = np.max(mY)  # Find max peak
        locY = np.argmax(mY)  # Find its location
        frqY = xf[locY]  # Get the actual frequency value

        return yf, frqY * 60, frqY, locY

    def Apply_FFT_M4(self,yf,objPlot): #rfft, abs, get power and its freq

        N = len(yf)

        yf = self.apply_fft_All(yf)
        yf = self.getPower_All(yf)
        xf = self.get_Freq(N)
        self.freq =xf


        yf[np.abs(self.freq) < 0.66 ] = 0
        yf[np.abs(self.freq) > 3.33 ] = 0

        max_peakgreen = yf[np.abs(self.freq)  <= 0.66].argmax()
        bpmir = np.abs(self.freq)[yf[np.abs(self.freq)  <= 3.33].argmax()] * 60
        # objPlot.PlotFFT(yf[:, 0], yf[:, 1], yf[:, 2], yf[:, 3],yf[:, 4], self.freq, "freq", "Savefilepath", "filename")
        # mY = np.abs(yf)  # Find magnitude
        # peakY = np.max(mY)  # Find max peak
        # locY = np.argmax(mY)  # Find its location
        # frqY = xf[locY]  # Get the actual frequency value
        #
        # hr =frqY * 60

        return yf[:, 0], yf[:, 1], yf[:, 2], yf[:, 3], yf[:, 4],self.freq #, frqY * 60, frqY, locY

    def Apply_FFT_M5_Individual(self, yf,ignoregray): #original 4
        N =len(yf)

        #aply fft
        self.fftblue = np.fft.fft(yf[:, 0])  # blue
        self.fftGreen = np.fft.fft(yf[:, 1])  # green
        self.fftRed = np.fft.fft(yf[:, 2])  # red


        index = 3
        if(not ignoregray):
            self.fftGrey = np.fft.fft(yf[:, 3])  # gr  # gr
            index = 4

        self.fftIR = np.fft.fft(yf[:, index])  # ir

        self.fftblue = self.fftblue / np.sqrt(len(self.fftblue))
        self.fftGreen = self.fftGreen / np.sqrt(len(self.fftGreen))
        self.fftRed = self.fftRed / np.sqrt(len(self.fftRed))

        if(not ignoregray):
            self.fftGrey = self.fftGrey / np.sqrt(len(self.fftGrey))

        self.fftIR = self.fftIR / np.sqrt(len(self.fftIR))

        xf = self.get_Freq(N)
        xf = np.fft.fftshift(xf)

        #IR
        yplotIR = np.fft.fftshift(abs(self.fftIR))
        # fft_plotIR = yplotIR
        # fft_plotIR[xf <= 0.66] = 0
        # max_peakIR =fft_plotIR[xf <= 3.33].argmax()
        # irbpm =xf[fft_plotIR[xf <= 3.33].argmax()] * 60

        # Red
        yplotRed = np.fft.fftshift(abs(self.fftRed))
        # fft_plotRed = yplotRed
        # fft_plotRed[xf <= 0.66] = 0
        # max_peakRed = fft_plotRed[xf <= 3.33].argmax()
        # Redbpm = xf[fft_plotRed[xf <= 3.33].argmax()] * 60

        # Green
        yplotGreen = np.fft.fftshift(abs(self.fftGreen))
        # fft_plotGreen = yplotGreen
        # fft_plotGreen[xf <= 0.66] = 0
        # max_peakGreen = fft_plotGreen[xf <= 3.33].argmax()
        # Greenbpm = xf[fft_plotGreen[xf <= 3.33].argmax()] * 60


        # Blue
        yplotBlue = np.fft.fftshift(abs(self.fftblue))
        # fft_plotBlue = yplotBlue
        # fft_plotBlue[xf <= 0.66] = 0
        # max_peakBlue = fft_plotBlue[xf <= 3.33].argmax()
        # Bluebpm = xf[fft_plotBlue[xf <= 3.33].argmax()] * 60


        if(not ignoregray):
            # Grey
            yplotGrey = np.fft.fftshift(abs(self.fftGrey))
        else:

            yplotGrey = []
        # fft_plotGrey = yplotGrey
        # fft_plotGrey[xf <= 0.66] = 0
        # max_peakGrey = fft_plotGrey[xf <= 3.33].argmax()
        # Greybpm = xf[fft_plotGrey[xf <= 3.33].argmax()] * 60

        #self.apply_fft(yf)
        #self.apply_sqrt_AfterFFt()
        # xf = self.get_Freq(N)
        # # xf = np.fft.fftshift(xf)
        # yplot = np.fft.fftshift(abs(yf))
        # fft_plot = yplot
        # fft_plot[xf <= 0.66] = 0
        # print("Method4")
        # max_peak =fft_plot[xf <= 3.33].argmax()
        # print(str(xf[fft_plot[xf <= 3.33].argmax()] * 60) + ' bpm for ')

        #return fft_plot, xf[max_peak] * 60, xf[max_peak], max_peak #freqs[max_peak] * 60, freqs[max_peak], max_peak

        return yplotBlue,yplotGreen,yplotRed,yplotGrey,yplotIR,xf

    def Apply_FFT_M5(self, yf): #original 4
        N =len(yf)
        yf = self.apply_fft_All(yf)
        yf = yf / np.sqrt(N)
        xf = self.get_Freq(N)
        xf = np.fft.fftshift(xf)
        yplot = np.fft.fftshift(abs(yf))
        fft_plot = yplot

        ##uncomment below
        # fft_plot[xf <= 0.66] = 0
        # print("Method4")
        # max_peak =fft_plot[xf <= 3.33].argmax()
        # print(str(xf[fft_plot[xf <= 3.33].argmax()] * 60) + ' bpm for ')
        # return fft_plot, xf[max_peak] * 60, xf[max_peak], max_peak #freqs[max_peak] * 60, freqs[max_peak], max_peak

        return fft_plot[:, 0], fft_plot[:, 1], fft_plot[:, 2], fft_plot[:, 3], fft_plot[:, 4]

    def Apply_FFT_M6_Individual(self, yf,ignoregray): #original 5
        N = len(yf)

        # aply fft
        self.fftblue = np.fft.fft(yf[:, 0])  # blue
        self.fftGreen = np.fft.fft(yf[:, 1])  # green
        self.fftRed = np.fft.fft(yf[:, 2])  # red
        index = 3
        if(not ignoregray):
            self.fftGrey = np.fft.fft(yf[:, 3])  # gr
            index = 4
        self.fftIR = np.fft.fft(yf[:, index])  # ir

        self.fftblue = self.fftblue / np.sqrt(len(self.fftblue))
        self.fftGreen = self.fftGreen / np.sqrt(len(self.fftGreen))
        self.fftRed = self.fftRed / np.sqrt(len(self.fftRed))
        if(not ignoregray):
            self.fftGrey = self.fftGrey / np.sqrt(len(self.fftGrey))
        self.fftIR = self.fftIR / np.sqrt(len(self.fftIR))

        xf = self.get_Freq(N)

        # IR
        yplotIR = abs(self.fftIR) #with and without abs
        # fft_plotIR = yplotIR.copy()
        # fft_plotIR[xf <= 0.66] = 0
        # max_peakIR = fft_plotIR[xf <= 3.33].argmax()
        # irbpm = xf[fft_plotIR[xf <= 3.33].argmax()] * 60

        # Red
        yplotRed = abs(self.fftRed)
        # fft_plotRed = yplotRed.copy()
        # fft_plotRed[xf <= 0.66] = 0
        # max_peakRed = fft_plotRed[xf <= 3.33].argmax()
        # Redbpm = xf[fft_plotRed[xf <= 3.33].argmax()] * 60

        # Green
        yplotGreen = abs(self.fftGreen)
        # fft_plotGreen = yplotGreen.copy()
        # fft_plotGreen[xf <= 0.66] = 0
        # max_peakGreen = fft_plotGreen[xf <= 3.33].argmax()
        # Greenbpm = xf[fft_plotGreen[xf <= 3.33].argmax()] * 60

        # Blue
        yplotBlue = abs(self.fftblue)
        # fft_plotBlue = yplotBlue.copy()
        # fft_plotBlue[xf <= 0.66] = 0
        # max_peakBlue = fft_plotBlue[xf <= 3.33].argmax()
        # Bluebpm = xf[fft_plotBlue[xf <= 3.33].argmax()] * 60


        if(not ignoregray):
            # Grey
            yplotGrey = abs(self.fftGrey)
            # fft_plotGrey = yplotGrey
            # fft_plotGrey[xf <= 0.66] = 0
            # max_peakGrey = fft_plotGrey[xf <= 3.33].argmax()
            # Greybpm = xf[fft_plotGrey[xf <= 3.33].argmax()] * 60
        else:
            # Greybpm=0
            # max_peakGrey=0
            yplotGrey= []

        # N = len(yf)
        # yf = self.apply_fft_All(yf)
        # yf = yf / np.sqrt(N)
        # xf = self.get_Freq()
        # fft_plot = yf
        # fft_plot[xf <= 0.66] = 0
        # print("Method5")
        # max_peak =fft_plot[xf <= 3.33].argmax()
        # print(str(xf[fft_plot[xf <= 3.33].argmax()] * 60) + ' bpm')
        # objPlot.PlotFFT(yplotBlue, yplotGreen, yplotRed, yplotGrey,
        #                 yplotIR, self.freq, "freq", "Savefilepath", "filename")

        # return fft_plot, xf[max_peak] * 60, xf[max_peak], max_peak #freqs[max_peak] * 60, freqs[max_peak], max_peak
        return yplotBlue,yplotGreen,yplotRed,yplotGrey,yplotIR,xf
               # max_peakBlue,max_peakGreen,max_peakRed,max_peakGrey,max_peakIR,\
               # Bluebpm,Greenbpm,Redbpm,Greybpm,irbpm

    def Apply_FFT_M6(self, yf): #original 5
        N = len(yf)
        yf = self.apply_fft_All(yf)
        yf = yf / np.sqrt(N)
        xf = self.get_Freq(N)
        fft_plot = yf

        ##uncomment below
        # fft_plot[xf <= 0.66] = 0
        # print("Method5")
        # max_peak =fft_plot[xf <= 3.33].argmax()
        # print(str(xf[fft_plot[xf <= 3.33].argmax()] * 60) + ' bpm')
        # return fft_plot, xf[max_peak] * 60, xf[max_peak], max_peak #freqs[max_peak] * 60, freqs[max_peak], max_peak

        return fft_plot[:, 0], fft_plot[:, 1], fft_plot[:, 2], fft_plot[:, 3], fft_plot[:, 4]

    def ApplyFFTMethod6(self,yf):
        N = len(yf)
        yf = self.apply_fft_All(yf)
        yf = yf / np.sqrt(N)
        xf = self.get_Freq(N)
        fft_plot = yf
        peakY2 = np.max(fft_plot)  # Find max peak
        locY2 = np.argmax(fft_plot)  # Find its location
        frqY2 = xf[locY2]  # Get the actual frequency value

        print("Method6")
        print(str(frqY2 * 60) + ' bpm')

    def ApplyFFTMethod7(self,yf):
        yf = self.apply_rfft_All(yf)
        xf = self.get_Freq()
        mY=abs(yf)
        peakY = np.max(mY)  # Find max peak
        locY = np.argmax(mY)  # Find its location
        frqY = xf[locY]  # Get the actual frequency value
        print("Method7")
        print(frqY*60)

    def ApplyFFTMethod8(self,savepath,objPlots,EstimatedFPS,yf,type,color,fig):
        yf = np.absolute(self.apply_rfft_All(yf))
        fsample= EstimatedFPS #fps
        N = len(yf)
        freqs = np.arange(0, fsample / 2, fsample / N)

        # Truncate to fs/2
        uf = yf[0:len(freqs)]


        # Get heartrate from FFT
        max_val = 0
        max_index = 0
        for index, fft_val in enumerate(yf):
            if fft_val > max_val:
                max_val = fft_val
                max_index = index

        heartrate = round(freqs[max_index] * 60, 1)
        print("Method8 : " + type + " , BPM : " + str(heartrate))

    def ApplyFFTMethod8Part2(self,savepath,objPlots,EstimatedFPS,S,type,color,fig):
        S_fft = np.absolute(self.apply_rfft_All(S))
        fsample= EstimatedFPS #fps
        N = len(S_fft)
        freqs = np.arange(0, fsample / 2, fsample / N)

        # Truncate to fs/2
        S_fft = S_fft[0:len(freqs)]


        objPlots.plotSingle(savepath,freqs,S_fft,type + 'fft8', color,fig)


        return S_fft

    def ApplyFFTTest(self,EstimatedFPS,objPlots,SavefigPath, S):


        #FFT1

        raw4 = np.fft.rfft(S)  # ir
        fftIR = np.abs(raw4)  # ir
        powerfftIR = np.abs(fftIR) ** 2   # ir

        # FFT2 wihtout **2
        S_fft = np.absolute(np.fft.rfft(S))

        n = len(raw4)
        N =len(S_fft)
        time_step = 1 / EstimatedFPS

        sample_freq = np.fft.fftfreq(n, d=time_step)
        freqs = np.arange(0, EstimatedFPS / 2, EstimatedFPS / N)
        S_fft = S_fft[0:len(freqs)]


        mY = np.abs(powerfftIR)  # Find magnitude
        peakY = np.max(mY)  # Find max peak
        locY = np.argmax(mY)  # Find its location
        frqY = sample_freq[locY]  # Get the actual frequency value
        value= frqY*60
        print(value)

        # Get heartrate from FFT
        max_val = 0
        max_index = 0
        for index, fft_val in enumerate(S_fft):
            if fft_val > max_val:
                max_val = fft_val
                max_index = index

        heartrate = round(freqs[max_index] * 60, 1)
        print("Method8 : " + type + " , BPM : " + str(heartrate))

    def ApplyFFTTest2(self, Algopath, objPlots,yf):
        yf = self.apply_fft(yf)
        yf = self.apply_Abs()
        xf = self.get_Freq()


    def ApplyFFTTest3(self,  yf):

        self.rfft(yf)
        self.apply_Abs()
        xf = self.get_Freq_bySamplesize()

        # mY = np.abs(self.powerfftIR)  # Find magnitude
        # peakY = np.max(mY)  # Find max peak
        # locY = np.argmax(mY)  # Find its location
        # frqY = self.freq[locY]  # Get the actual frequency value

        return self.B_fft,self.Gr_fft,self.R_fft,self.Gy_fft,self.IR_fft


    def ApplyFFT9(self,S,ignoregray):
        # obtain the frequencies using scipy function
        self.get_Freq(len(S))

        #### blue
        blue_x=S[:, 0]
        # FFT the signal
        blue_sig_fft = fft(blue_x)

        # # copy the FFT results
        # blue_sig_fft_filtered = abs(blue_sig_fft.copy())
        #
        # # high-pass filter by assign zeros to the
        # # FFT amplitudes where the absolute
        # # frequencies smaller than the cut-off
        # blue_sig_fft_filtered[np.abs(self.freq) < cut_off ] = 0
        # blue_sig_fft_filtered[np.abs(self.freq) > upperbound ] = 0
        #
        # max_peakBlue = blue_sig_fft_filtered[np.abs(self.freq)  <= upperbound].argmax()
        # Bluebpm = np.abs(self.freq)[blue_sig_fft_filtered[np.abs(self.freq)  <= upperbound].argmax()] * 60

        #### GREEN
        green_x=S[:, 1]
        # FFT the signal
        green_sig_fft = fft(green_x)
        # copy the FFT results
        # green_sig_fft_filtered = abs(green_sig_fft.copy())
        # # high-pass filter by assign zeros to the
        # # FFT amplitudes where the absolute
        # # frequencies smaller than the cut-off
        # green_sig_fft_filtered[np.abs(self.freq) < cut_off ] = 0
        # green_sig_fft_filtered[np.abs(self.freq) > upperbound ] = 0
        #
        # max_peakgreen = green_sig_fft_filtered[np.abs(self.freq)  <= upperbound].argmax()
        # green_bpm = np.abs(self.freq)[green_sig_fft_filtered[np.abs(self.freq)  <= upperbound].argmax()] * 60


        #### RED
        red_x=S[:, 2]
        # FFT the signal
        red_sig_fft = fft(red_x)
        # # copy the FFT results
        # red_sig_fft_filtered = abs(red_sig_fft.copy())
        # # high-pass filter by assign zeros to the
        # # FFT amplitudes where the absolute
        # # frequencies smaller than the cut-off
        # red_sig_fft_filtered[np.abs(self.freq) < cut_off ] = 0
        # red_sig_fft_filtered[np.abs(self.freq) > upperbound ] = 0
        #
        # max_peakred = red_sig_fft_filtered[np.abs(self.freq)  <= upperbound].argmax()
        # red_bpm = np.abs(self.freq)[red_sig_fft_filtered[np.abs(self.freq)  <= upperbound].argmax()] * 60

        if(not ignoregray):
            #### grey
            grey_x=S[:, 3]
            # FFT the signal
            grey_sig_fft = fft(grey_x)
            # # copy the FFT results
            # grey_sig_fft_filtered = abs(grey_sig_fft.copy())
            # # high-pass filter by assign zeros to the
            # # FFT amplitudes where the absolute
            # # frequencies smaller than the cut-off
            # grey_sig_fft_filtered[np.abs(self.freq) < cut_off ] = 0
            # grey_sig_fft_filtered[np.abs(self.freq) >upperbound ] = 0
            #
            # max_peakgrey = grey_sig_fft_filtered[np.abs(self.freq)  <=upperbound].argmax()
            # grey_bpm = np.abs(self.freq)[grey_sig_fft_filtered[np.abs(self.freq)  <= upperbound].argmax()] * 60


            #### Ir
            ir_x=S[:, 4]
            # FFT the signal
            ir_sig_fft = fft(ir_x)
            # # copy the FFT results
            # ir_sig_fft_filtered = abs(ir_sig_fft.copy())
            # # high-pass filter by assign zeros to the
            # # FFT amplitudes where the absolute
            # # frequencies smaller than the cut-off
            # ir_sig_fft_filtered[np.abs(self.freq) < cut_off ] = 0
            # ir_sig_fft_filtered[np.abs(self.freq) > upperbound ] = 0
            #
            # max_peakir = ir_sig_fft_filtered[np.abs(self.freq)  <= upperbound].argmax()
            # ir_bpm = np.abs(self.freq)[ir_sig_fft_filtered[np.abs(self.freq)  <= upperbound].argmax()] * 60

        else:
            #### Ir
            ir_x=S[:, 3]
            # FFT the signal
            ir_sig_fft = fft(ir_x)
            # # copy the FFT results
            # ir_sig_fft_filtered = abs(ir_sig_fft.copy())
            # # high-pass filter by assign zeros to the
            # # FFT amplitudes where the absolute
            # # frequencies smaller than the cut-off
            # ir_sig_fft_filtered[np.abs(self.freq) < cut_off ] = 0
            # ir_sig_fft_filtered[np.abs(self.freq) > upperbound ] = 0
            #
            # max_peakir = ir_sig_fft_filtered[np.abs(self.freq)  <= upperbound].argmax()
            # ir_bpm = np.abs(self.freq)[ir_sig_fft_filtered[np.abs(self.freq)  <= upperbound].argmax()] * 60
            #
            # grey_sig_fft = None
            # max_peakgrey = 0
            # grey_bpm = 0
            grey_x = []
            grey_sig_fft = []

        # objPlot.PlotFFT(blue_sig_fft_filtered,green_sig_fft_filtered,red_sig_fft_filtered,grey_sig_fft_filtered,ir_sig_fft_filtered,self.freq,"freq","Savefilepath","filename")
        #
        # plt.figure(figsize=(12, 6))
        # # plt.subplot(121)
        #
        # plt.stem(self.freq, X, 'b', \
        #          markerfmt=" ", basefmt="-b")
        # plt.xlabel('Freq (Hz)')
        # plt.ylabel('FFT Amplitude |X(freq)|')
        # plt.xlim(0, 10)
        # plt.ylim(0, 20000)
        # #
        # # plt.subplot(122)
        # # plt.plot( ifft(X), 'r')
        # # plt.xlabel('Time (s)')
        # # plt.ylabel('Amplitude')
        # plt.tight_layout()
        # plt.show()
        # scipy.signal.find_peaks_cwt can also be used for more advanced
        # peak detection
        # a=0

        #-----
        # plt.figure(figsize=(12, 6))
        # plt.subplot(121)
        #
        # plt.stem(self.freq, np.abs(X), 'b', \
        #          markerfmt=" ", basefmt="-b")
        # plt.xlabel('Freq (Hz)')
        # plt.ylabel('FFT Amplitude |X(freq)|')
        # #plt.xlim(0, 10)
        #
        # plt.subplot(122)
        # plt.plot( ifft(X), 'r')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.tight_layout()
        # plt.show()

        return blue_sig_fft,green_sig_fft,red_sig_fft,grey_sig_fft,ir_sig_fft,self.freq
               # max_peakBlue,max_peakgreen,max_peakred,max_peakgrey,max_peakir,\
               # Bluebpm,green_bpm,red_bpm,grey_bpm,ir_bpm

    #########------Algorithms------#############

    def ApplyICA(self,S,compts):
        ica = FastICA(n_components=compts, max_iter=1000)#n_components=3,max_iter=10000 whiten=True, max_iter=1000
        np.random.seed(0)
        S /= S.std(axis=0)  # Standardize data
        self.X = S
        # X = np.dot(self.S, A.T)
        self.S_ = ica.fit(self.X).transform(self.X)  # Get the estimated sources
        A_ = ica.mixing_  # Get estimated mixing matrix , setting seed , numpy

        # assert np.allclose(self.X, np.dot(self.S_, A_.T) + ica.mean_)
        return self.S_

    def ApplySVD(self,S):

        S /= S.std(axis=0)  # Standardize data
        result = np.linalg.svd(S, full_matrices=True)
        # svd = TruncatedSVD(n_components=5)
        g=0
        #print()
    #>> > print()
#    >> > print()

    def ApplyICARGB(self, S):
        ica = FastICA(whiten=True)  # n_components=3
        np.random.seed(0)
        S /= S.std(axis=0)  # Standardize data
        self.X = S
        # X = np.dot(self.S, A.T)
        self.S_ = ica.fit(self.X).transform(self.X)  # Get the estimated sources
        A_ = ica.mixing_  # Get estimated mixing matrix , setting seed , numpy
        return self.S_

    def ApplyICASingle(self,S): #Grey and IR
        ica = FastICA(whiten=True)  # n_components=3
        np.random.seed(0)
        S /= S.std(axis=0)  # Standardize data
        self.X = S
        self.S_ = ica.fit(self.X).transform(self.X)  # Get the estimated sources
        A_ = ica.mixing_  # Get estimated mixing matrix , setting seed , numpy
        return self.S_

    def ApplyPCA(self,X,compts):
        X /= X.std(axis=0)  # Standardize data
        pca = PCA(n_components=compts)
        pca.fit(X)
        X_new = pca.fit_transform(X)
        #v=pca.explained_variance_ratio_
        return X_new

    def jadeR2(self, X):
        origtype = X.dtype  # float64
        X = matrix(X.astype(float64))  # create a matrix from a copy of X created as a float 64 array
        [n, T] = X.shape
        m = n
        X -= X.mean(1)

        # whitening & projection onto signal subspace
        # -------------------------------------------
        # An eigen basis for the sample covariance matrix
        [D, U] = eig((X * X.T) / float(T))
        # Sort by increasing variances
        k = D.argsort()
        Ds = D[k]

        # The m most significant princip. comp. by decreasing variance
        PCs = arange(n - 1, n - m - 1, -1)

        # PCA
        # At this stage, B does the PCA on m components
        B = U[:, k[PCs]].T

        # --- Scaling ---------------------------------
        # The scales of the principal components
        scales = sqrt(Ds[PCs])
        B = diag(1. / scales) * B
        # Sphering
        X = B * X

        # We have done the easy part: B is a whitening matrix and X is white.

        del U, D, Ds, k, PCs, scales

        # Estimation of Cumulant Matrices
        # -------------------------------

        # Reshaping of the data, hoping to speed up things a little bit...
        X = X.T  # transpose data to (256, 3)
        # Dim. of the space of real symm matrices
        dimsymm = (m * (m + 1)) / 2  # 6
        # number of cumulant matrices
        nbcm = dimsymm  # 6
        # Storage for cumulant matrices

        z =zeros((int(m), int(m * nbcm)), dtype=float64)

        CM = matrix(z)

        R = matrix(eye(m, dtype=float64))  # [[ 1.  0.  0.] [ 0.  1.  0.] [ 0.  0.  1.]]
        # Temp for a cum. matrix
        Qij = matrix(zeros([m, m], dtype=float64))
        # Temp
        Xim = zeros(m, dtype=float64)
        # Temp
        Xijm = zeros(m, dtype=float64)

        # will index the columns of CM where to store the cum. mats.
        Range = arange(m)  # [0 1 2]

        txt = [1890, 1786995]
        CM = matrix(zeros(txt, dtype=float64))
        for im in range(m):
            Xim = X[:, im]
            Xijm = multiply(Xim, Xim)
            Qij = multiply(Xijm, X).T * X / float(T) - R - 2 * dot(R[:, im], R[:, im].T)
            CM[:, Range] = Qij
            Range = Range + m
            for jm in range(im):
                Xijm = multiply(Xim, X[:, jm])
                Qij = sqrt(2) * multiply(Xijm, X).T * X / float(T) - R[:, im] * R[:, jm].T - R[:, jm] * R[:, im].T
                CM[:, Range] = Qij
                Range = Range + m

        # Now we have nbcm = m(m+1)/2 cumulants matrices stored in a big
        # m x m*nbcm array.

        # Joint diagonalization of the cumulant matrices
        # ==============================================

        V = matrix(eye(m, dtype=float64))  # [[ 1.  0.  0.] [ 0.  1.  0.] [ 0.  0.  1.]]

        Diag = zeros(m, dtype=float64)  # [0. 0. 0.]
        On = 0.0
        Range = arange(m)  # [0 1 2]

        nbcm = int(nbcm)

        for im in range(nbcm):  # nbcm == 6
            Diag = diag(CM[:, Range])  ##CM[:,Range]
            On = On + (Diag * Diag).sum(axis=0)
            Range = Range + m
        Off = (multiply(CM, CM).sum(axis=0)).sum(axis=0) - On
        # A statistically scaled threshold on `small" angles
        seuil = 1.0e-6 / sqrt(T)  # 6.25e-08
        # sweep number
        encore = True
        sweep = 0
        # Total number of rotations
        updates = 0
        # Number of rotations in a given seep
        upds = 0
        g = zeros([2, nbcm], dtype=float64)  # [[ 0.  0.  0.  0.  0.  0.] [ 0.  0.  0.  0.  0.  0.]]
        gg = zeros([2, 2], dtype=float64)  # [[ 0.  0.]  [ 0.  0.]]
        G = zeros([2, 2], dtype=float64)
        c = 0
        s = 0
        ton = 0
        toff = 0
        theta = 0
        Gain = 0

        # Joint diagonalization proper

        while encore:
            encore = False
            sweep = sweep + 1
            upds = 0
            Vkeep = V

            for p in range(m - 1):  # m == 3
                for q in range(p + 1, m):  # p == 1 | range(p+1, m) == [2]

                    Ip = arange(p, m * nbcm, m)  # [ 0  3  6  9 12 15] [ 0  3  6  9 12 15] [ 1  4  7 10 13 16]
                    Iq = arange(q, m * nbcm, m)  # [ 1  4  7 10 13 16] [ 2  5  8 11 14 17] [ 2  5  8 11 14 17]

                    # computation of Givens angle
                    g = concatenate([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
                    gg = dot(g, g.T)
                    ton = gg[0, 0] - gg[1, 1]  # -6.54012319852 4.44880758012 -1.96674621935
                    toff = gg[0, 1] + gg[1, 0]  # -15.629032394 -4.3847687273 6.72969915184
                    theta = 0.5 * arctan2(toff, ton + sqrt(
                        ton * ton + toff * toff))  # -0.491778606993 -0.194537202087 0.463781701868
                    Gain = (sqrt(ton * ton + toff * toff) - ton) / 4.0  # 5.87059352069 0.449409565866 2.24448683877

                    if abs(theta) > seuil:
                        encore = True
                        upds = upds + 1
                        c = cos(theta)
                        s = sin(theta)
                        G = matrix([[c, -s], [s, c]])  # DON"T PRINT THIS! IT"LL BREAK THINGS! HELLA LONG
                        pair = array([p, q])  # don't print this either
                        V[:, pair] = V[:, pair] * G
                        CM[pair, :] = G.T * CM[pair, :]
                        CM[:, concatenate([Ip, Iq])] = append(c * CM[:, Ip] + s * CM[:, Iq],
                                                              -s * CM[:, Ip] + c * CM[:, Iq], axis=1)
                        On = On + Gain
                        Off = Off - Gain
            updates = updates + upds  # 3 6 9 9

        # A separating matrix
        # -------------------

        B = V.T * B  # [[ 0.17242566  0.10485568 -0.7373937 ] [-0.41923305 -0.84589716  1.41050008]  [ 1.12505903 -2.42824508  0.92226197]]

        # Permute the rows of the separating matrix B to get the most energetic
        # components first. Here the **signals** are normalized to unit variance.
        # Therefore, the sort is according to the norm of the columns of
        # A = pinv(B)

        A = pinv(
            B)  # [[-3.35031851 -2.14563715  0.60277625] [-2.49989794 -1.25230985 -0.0835184 ] [-2.49501641 -0.67979249  0.12907178]]
        keys = array(argsort(multiply(A, A).sum(axis=0)[0]))[0]  # [2 1 0]
        B = B[keys,
            :]  # [[ 1.12505903 -2.42824508  0.92226197] [-0.41923305 -0.84589716  1.41050008] [ 0.17242566  0.10485568 -0.7373937 ]]
        B = B[::-1,
            :]  # [[ 0.17242566  0.10485568 -0.7373937 ] [-0.41923305 -0.84589716  1.41050008] [ 1.12505903 -2.42824508  0.92226197]]
        # just a trick to deal with sign == 0
        b = B[:, 0]  # [[ 0.17242566] [-0.41923305] [ 1.12505903]]
        signs = array(sign(sign(b) + 0.1).T)[0]  # [1. -1. 1.]
        B = diag(
            signs) * B  # [[ 0.17242566  0.10485568 -0.7373937 ] [ 0.41923305  0.84589716 -1.41050008] [ 1.12505903 -2.42824508  0.92226197]]
        return B

    def jadeR(self,X):
        origtype = X.dtype  # float64

        X = matrix(X.astype(float64))  # create a matrix from a copy of X created as a float 64 array

        [T,n] = X.shape

        m = 5

        X -= X.mean(1)

        # whitening & projection onto signal subspace
        # -------------------------------------------

        # An eigen basis for the sample covariance matrix
        [D, U] = eig((X * X.T) / float(T))
        # Sort by increasing variances
        k = D.argsort()
        Ds = D[k]

        # The m most significant princip. comp. by decreasing variance
        PCs = arange(n - 1, n - m - 1, -1)

        # PCA
        # At this stage, B does the PCA on m components
        B = U[:, k[PCs]] #.T

        # --- Scaling ---------------------------------
        # The scales of the principal components
        scales = sqrt(Ds[PCs])
        B = diag(1. / scales) * B
        # Sphering
        X = B * X

        # We have done the easy part: B is a whitening matrix and X is white.

        del U, D, Ds, k, PCs, scales

        # Reshaping of the data, hoping to speed up things a little bit...
        X = X.T  # transpose data to (256, 3)
        # Dim. of the space of real symm matrices
        dimsymm = (m * (m + 1)) / 2  # 6
        # number of cumulant matrices
        nbcm = dimsymm  # 6
        # Storage for cumulant matrices
        CM = matrix(zeros([int(m),int(m * nbcm) ], dtype=float64))
        R = matrix(np.eye(m, dtype=float64))  # [[ 1.  0.  0.] [ 0.  1.  0.] [ 0.  0.  1.]]

        # Temp for a cum. matrix
        Qij = matrix(zeros([m, m], dtype=float64))
        # Temp
        Xim = zeros(m, dtype=float64)
        # Temp
        Xijm = zeros(m, dtype=float64)

        # I am using a symmetry trick to save storage. I should write a short note
        # one of these days explaining what is going on here.
        # will index the columns of CM where to store the cum. mats.
        Range = arange(m)  # [0 1 2]

        for im in range(m):
            Xim = X[:, im]
            Xijm = multiply(Xim, Xim)
            Qij = multiply(Xijm, X).T * X / float(T) - R - 2 * dot(R[:, im], R[:, im].T)
            CM[:, Range] = Qij
            Range = Range + m
            for jm in range(im):
                Xijm = multiply(Xim, X[:, jm])
                Qij = sqrt(2) * multiply(Xijm, X).T * X / float(T) - R[:, im] * R[:, jm].T - R[:, jm] * R[:, im].T
                CM[:, Range] = Qij
                Range = Range + m

        # Now we have nbcm = m(m+1)/2 cumulants matrices stored in a big
        # m x m*nbcm array.

        # Joint diagonalization of the cumulant matrices
        # ==============================================

        V = matrix(np.eye(m, dtype=float64))  # [[ 1.  0.  0.] [ 0.  1.  0.] [ 0.  0.  1.]]

        Diag = zeros(m, dtype=float64)  # [0. 0. 0.]
        On = 0.0
        Range = arange(m)  # [0 1 2]
        for im in range(int(nbcm)):  # nbcm == 6
            Diag = diag(CM[:, Range])
            On = On + (Diag * Diag).sum(axis=0)
            Range = Range + m
        Off = (multiply(CM, CM).sum(axis=0)).sum(axis=0) - On
        # A statistically scaled threshold on `small" angles
        seuil = 1.0e-6 / sqrt(T)  # 6.25e-08
        # sweep number
        encore = True
        sweep = 0
        # Total number of rotations
        updates = 0
        # Number of rotations in a given seep
        upds = 0
        g = zeros([2, int(nbcm)], dtype=float64)  # [[ 0.  0.  0.  0.  0.  0.] [ 0.  0.  0.  0.  0.  0.]]
        gg = zeros([2, 2], dtype=float64)  # [[ 0.  0.]  [ 0.  0.]]
        G = zeros([2, 2], dtype=float64)
        c = 0
        s = 0
        ton = 0
        toff = 0
        theta = 0
        Gain = 0

        # Joint diagonalization proper

        while encore:
            encore = False
            sweep = sweep + 1
            upds = 0
            Vkeep = V

            for p in range(m - 1):  # m == 3
                for q in range(p + 1, m):  # p == 1 | range(p+1, m) == [2]

                    Ip = arange(p, int(m * nbcm), m)  # [ 0  3  6  9 12 15] [ 0  3  6  9 12 15] [ 1  4  7 10 13 16]
                    Iq = arange(q, int(m * nbcm), m)  # [ 1  4  7 10 13 16] [ 2  5  8 11 14 17] [ 2  5  8 11 14 17]

                    # computation of Givens angle
                    g = concatenate([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
                    gg = dot(g, g.T)
                    ton = gg[0, 0] - gg[1, 1]  # -6.54012319852 4.44880758012 -1.96674621935
                    toff = gg[0, 1] + gg[1, 0]  # -15.629032394 -4.3847687273 6.72969915184
                    theta = 0.5 * arctan2(toff, ton + sqrt(
                        ton * ton + toff * toff))  # -0.491778606993 -0.194537202087 0.463781701868
                    Gain = (sqrt(ton * ton + toff * toff) - ton) / 4.0  # 5.87059352069 0.449409565866 2.24448683877

                    if abs(theta) > seuil:
                        encore = True
                        upds = upds + 1
                        c = cos(theta)
                        s = sin(theta)
                        G = matrix([[c, -s], [s, c]])  # DON"T PRINT THIS! IT"LL BREAK THINGS! HELLA LONG
                        pair = array([p, q])  # don't print this either
                        abc=V[:, pair] * G
                        V[:, pair] = V[:, pair] * G
                        CM[pair, :] = G.T * CM[pair, :]
                        CM[:, concatenate([Ip, Iq])] = append(c * CM[:, Ip] + s * CM[:, Iq], -s * CM[:, Ip] + c * CM[:, Iq],
                                                              axis=1)
                        On = On + Gain
                        Off = Off - Gain
            updates = updates + upds  # 3 6 9 9

        # A separating matrix
        # -------------------

        B = V.T * B  # [[ 0.17242566  0.10485568 -0.7373937 ] [-0.41923305 -0.84589716  1.41050008]  [ 1.12505903 -2.42824508  0.92226197]]

        # Permute the rows of the separating matrix B to get the most energetic
        # components first. Here the **signals** are normalized to unit variance.
        # Therefore, the sort is according to the norm of the columns of
        # A = pinv(B)

        A = pinv(
            B)  # [[-3.35031851 -2.14563715  0.60277625] [-2.49989794 -1.25230985 -0.0835184 ] [-2.49501641 -0.67979249  0.12907178]]
        keys = array(argsort(multiply(A, A).sum(axis=0)[0]))[0]  # [2 1 0]
        B = B[keys,
            :]  # [[ 1.12505903 -2.42824508  0.92226197] [-0.41923305 -0.84589716  1.41050008] [ 0.17242566  0.10485568 -0.7373937 ]]
        B = B[::-1,
            :]  # [[ 0.17242566  0.10485568 -0.7373937 ] [-0.41923305 -0.84589716  1.41050008] [ 1.12505903 -2.42824508  0.92226197]]
        # just a trick to deal with sign == 0
        b = B[:, 0]  # [[ 0.17242566] [-0.41923305] [ 1.12505903]]
        signs = array(sign(sign(b) + 0.1).T)[0]  # [1. -1. 1.]
        B = diag(
            signs) * B  # [[ 0.17242566  0.10485568 -0.7373937 ] [ 0.41923305  0.84589716 -1.41050008] [ 1.12505903 -2.42824508  0.92226197]]
        return B

    def jadeR4(self,X,m):

        ICA = jade.main(X,m)
        return ICA

    def jadeR3(self,X, m=None, verbose=False):
        a=0
        assert isinstance(X, ndarray), \
            "X (input data matrix) is of the wrong type (%s)" % type(X)
        origtype = X.dtype  # remember to return matrix B of the same type
        X = matrix(X.astype(float64))
        assert X.ndim == 2, "X has %d dimensions, should be 2" % X.ndim
        assert (verbose == True) or (verbose == False), \
            "verbose parameter should be either True or False"

        # GB: n is number of input signals, T is number of samples
        [T,n] = X.shape
        assert n < T, "number of sensors must be smaller than number of samples"

        # Number of sources defaults to number of sensors
        if m == None:
            m = n
        assert m <= n, \
            "number of sources (%d) is larger than number of sensors (%d )" % (m, n)

        if verbose:
            print("jade -> Looking for %d sources" % m)
            print("jade -> Removing the mean value")
        X -= X.mean(1)

        # whitening & projection onto signal subspace
        # ===========================================

        if verbose: print("jade -> Whitening the data")
        # An eigen basis for the sample covariance matrix
        [D, U] = eig((X * X.T) / float(T))
        # Sort by increasing variances
        k = D.argsort()
        Ds = D[k]
        # The m most significant princip. comp. by decreasing variance
        PCs = arange(n - 1, n - m - 1, -1)

        # --- PCA  ----------------------------------------------------------
        # At this stage, B does the PCA on m components
        B = U[:, k[PCs]].T

        # --- Scaling  ------------------------------------------------------
        # The scales of the principal components
        scales = sqrt(Ds[PCs])
        # Now, B does PCA followed by a rescaling = sphering
        B = diag(1. / scales) * B
        # --- Sphering ------------------------------------------------------
        X = B * X

        # We have done the easy part: B is a whitening matrix and X is white.

        del U, D, Ds, k, PCs, scales

        # NOTE: At this stage, X is a PCA analysis in m components of the real
        # data, except that all its entries now have unit variance. Any further
        # rotation of X will preserve the property that X is a vector of
        # uncorrelated components. It remains to find the rotation matrix such
        # that the entries of X are not only uncorrelated but also `as independent
        # as possible". This independence is measured by correlations of order
        # higher than 2. We have defined such a measure of independence which 1)
        # is a reasonable approximation of the mutual information 2) can be
        # optimized by a `fast algorithm" This measure of independence also
        # corresponds to the `diagonality" of a set of cumulant matrices. The code
        # below finds the `missing rotation " as the matrix which best
        # diagonalizes a particular set of cumulant matrices.

        # Estimation of the cumulant matrices
        # ===================================

        if verbose: print("jade -> Estimating cumulant matrices")

        # Reshaping of the data, hoping to speed up things a little bit...
        X = X.T
        # Dim. of the space of real symm matrices
        dimsymm = (m * (m + 1)) / 2
        # number of cumulant matrices
        nbcm = int(dimsymm)
        # Storage for cumulant matrices
        #CM = matrix(zeros([m, m * nbcm], dtype=float64))
        CM = matrix(zeros([int(m),int(m * nbcm) ], dtype=float64))
        R = matrix(np.eye(m, dtype=float64))
        # Temp for a cum. matrix
        Qij = matrix(np.zeros([m, m], dtype=float64))
        # Temp
        Xim = np.zeros(m, dtype=float64)
        # Temp
        Xijm = np.zeros(m, dtype=float64)

        # I am using a symmetry trick to save storage. I should write a short note
        # one of these days explaining what is going on here.
        # will index the columns of CM where to store the cum. mats.
        Range = arange(m)

        for im in range(m):
            Xim = X[:, im]
            Xijm = multiply(Xim, Xim)
            # Note to myself: the -R on next line can be removed: it does not affect
            # the joint diagonalization criterion
            Qij = multiply(Xijm, X).T * X / float(T) - R - 2 * (R[:, im] * R[:, im].T)
            CM[:, Range] = Qij
            Range = Range + m
            for jm in range(im):
                Xijm = multiply(Xim, X[:, jm])
                Qij = sqrt(2) * (multiply(Xijm, X).T * X / float(T)
                                 - R[:, im] * R[:, jm].T - R[:, jm] * R[:, im].T)
                CM[:, Range] = Qij
                Range = Range + m

        # Now we have nbcm = m(m+1)/2 cumulants matrices stored in a big
        # m x m*nbcm array.

        # Joint diagonalization of the cumulant matrices
        # ==============================================

        V = matrix(np.eye(m, dtype=float64))

        Diag = zeros(m, dtype=float64)
        On = 0.0
        Range = arange(m)
        for im in range(int(nbcm)):
            Diag = diag(CM[:, Range])
            On = On + (Diag * Diag).sum(axis=0)
            Range = Range + m
        Off = (multiply(CM, CM).sum(axis=0)).sum(axis=1) - On
        # A statistically scaled threshold on `small" angles
        seuil = 1.0e-6 / sqrt(T)
        # sweep number
        encore = True
        sweep = 0
        # Total number of rotations
        updates = 0
        # Number of rotations in a given seep
        upds = 0
        g = zeros([2, nbcm], dtype=float64)
        gg = zeros([2, 2], dtype=float64)
        G = zeros([2, 2], dtype=float64)
        c = 0
        s = 0
        ton = 0
        toff = 0
        theta = 0
        Gain = 0

        # Joint diagonalization proper
        # ============================
        if verbose: print("jade -> Contrast optimization by joint diagonalization")

        while encore:
            encore = False
            if verbose: print("jade -> Sweep #%3d" % sweep)
            sweep = sweep + 1
            upds = 0
            Vkeep = V

            for p in range(m - 1):
                for q in range(p + 1, m):

                    Ip = arange(p, m * nbcm, m)
                    Iq = arange(q, m * nbcm, m)

                    # computation of Givens angle
                    g = concatenate([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
                    gg = dot(g, g.T)
                    ton = gg[0, 0] - gg[1, 1]
                    toff = gg[0, 1] + gg[1, 0]
                    theta = 0.5 * arctan2(toff, ton + sqrt(ton * ton + toff * toff))
                    Gain = (sqrt(ton * ton + toff * toff) - ton) / 4.0

                    # Givens update
                    if abs(theta) > seuil:
                        encore = True
                        upds = upds + 1
                        c = cos(theta)
                        s = sin(theta)
                        G = matrix([[c, -s], [s, c]])
                        pair = array([p, q])
                        V[:, pair] = V[:, pair] * G
                        CM[pair, :] = G.T * CM[pair, :]
                        CM[:, concatenate([Ip, Iq])] = \
                            append(c * CM[:, Ip] + s * CM[:, Iq], -s * CM[:, Ip] + c * CM[:, Iq], \
                                   axis=1)
                        On = On + Gain
                        Off = Off - Gain

            if verbose: print("completed in %d rotations" % upds)
            updates = updates + upds

        if verbose: print("jade -> Total of %d Givens rotations" % updates)

        # A separating matrix
        # ===================
        B = V.T * B

        # Permute the rows of the separating matrix B to get the most energetic
        # components first. Here the **signals** are normalized to unit variance.
        # Therefore, the sort is according to the norm of the columns of
        # A = pinv(B)

        if verbose: print("jade -> Sorting the components")

        A = pinv(B)

        keys = array(argsort(multiply(A, A).sum(axis=0)[0]))[0]
        B = B[keys, :]
        # % Is this smart ?
        B = B[::-1, :]

        if verbose: print("jade -> Fixing the signs")
        b = B[:, 0]
        # just a trick to deal with sign == 0
        signs = array(sign(sign(b) + 0.1).T)[0]
        B = diag(signs) * B

        return B.astype(origtype)


    def jadeR5(self,X, m=None, verbose=False):

        a=0
        assert isinstance(X, ndarray), \
            "X (input data matrix) is of the wrong type (%s)" % type(X)
        origtype = X.dtype  # remember to return matrix B of the same type
        X = matrix(X.astype(float64))
        assert X.ndim == 2, "X has %d dimensions, should be 2" % X.ndim
        assert (verbose == True) or (verbose == False), \
            "verbose parameter should be either True or False"

        # GB: n is number of input signals, T is number of samples
        [T,n] = X.shape
        assert n < T, "number of sensors must be smaller than number of samples"

        # Number of sources defaults to number of sensors
        if m == None:
            m = n
        assert m <= n, \
            "number of sources (%d) is larger than number of sensors (%d )" % (m, n)

        if verbose:
            print("jade -> Looking for %d sources" % m)
            print("jade -> Removing the mean value")
        X -= X.mean(1)

        # whitening & projection onto signal subspace
        # ===========================================

        if verbose: print("jade -> Whitening the data")
        # An eigen basis for the sample covariance matrix
        [D, U] = eig((X * X.T) / float(T))
        # Sort by increasing variances
        k = D.argsort()
        Ds = D[k]
        # The m most significant princip. comp. by decreasing variance
        PCs = arange(n - 1, n - m - 1, -1)

        # --- PCA  ----------------------------------------------------------
        # At this stage, B does the PCA on m components
        B = U[:, k[PCs]].T

        # --- Scaling  ------------------------------------------------------
        # The scales of the principal components
        scales = sqrt(Ds[PCs])
        # Now, B does PCA followed by a rescaling = sphering
        B = diag(1. / scales) * B
        # --- Sphering ------------------------------------------------------
        X = B * X

        # We have done the easy part: B is a whitening matrix and X is white.

        del U, D, Ds, k, PCs, scales

        # NOTE: At this stage, X is a PCA analysis in m components of the real
        # data, except that all its entries now have unit variance. Any further
        # rotation of X will preserve the property that X is a vector of
        # uncorrelated components. It remains to find the rotation matrix such
        # that the entries of X are not only uncorrelated but also `as independent
        # as possible". This independence is measured by correlations of order
        # higher than 2. We have defined such a measure of independence which 1)
        # is a reasonable approximation of the mutual information 2) can be
        # optimized by a `fast algorithm" This measure of independence also
        # corresponds to the `diagonality" of a set of cumulant matrices. The code
        # below finds the `missing rotation " as the matrix which best
        # diagonalizes a particular set of cumulant matrices.

        # Estimation of the cumulant matrices
        # ===================================

        if verbose: print("jade -> Estimating cumulant matrices")

        # Reshaping of the data, hoping to speed up things a little bit...
        X = X.T
        # Dim. of the space of real symm matrices
        dimsymm = (m * (m + 1)) / 2
        # number of cumulant matrices
        nbcm = int(dimsymm)
        # Storage for cumulant matrices
        #CM = matrix(zeros([m, m * nbcm], dtype=float64))
        CM = matrix(zeros([int(m),int(m * nbcm) ], dtype=float64))
        R = matrix(np.eye(m, dtype=float64))
        # Temp for a cum. matrix
        Qij = matrix(np.zeros([m, m], dtype=float64))
        # Temp
        Xim = np.zeros(m, dtype=float64)
        # Temp
        Xijm = np.zeros(m, dtype=float64)
        Uns		= np.ones(m, dtype=float64)   # for convenience

        # I am using a symmetry trick to save storage. I should write a short note
        # one of these days explaining what is going on here.
        # will index the columns of CM where to store the cum. mats.
        Range = arange(m)

        for im in range(m):
            Xim = X[:, im]
            Xijm = multiply(Xim, Xim)
            # Note to myself: the -R on next line can be removed: it does not affect
            # the joint diagonalization criterion
            Qij = multiply(Xijm, X).T * X / float(T) - R - 2 * (R[:, im] * R[:, im].T)
            CM[:, Range] = Qij
            Range = Range + m
            for jm in range(im):
                Xijm = multiply(Xim, X[:, jm])
                Qij = sqrt(2) * (multiply(Xijm, X).T * X / float(T)
                                 - R[:, im] * R[:, jm].T - R[:, jm] * R[:, im].T)
                CM[:, Range] = Qij
                Range = Range + m

        # Now we have nbcm = m(m+1)/2 cumulants matrices stored in a big
        # m x m*nbcm array.

        # Joint diagonalization of the cumulant matrices
        # ==============================================

        V = matrix(np.eye(m, dtype=float64))
        if 0:
            if verbose: print("jade -> Total of %d Givens rotations")
            [V, D] = eig(CM[:, 1: m])

            for u in range(1, m, m * nbcm):
                CM[:, u: u + m - 1] = CM[:, u: u + m - 1] * V
            CM = V.T * CM
        else:
            V = np.eye(m)
        #
        Diag = zeros(m, dtype=float64)
        On = 0.0
        Range = arange(m)
        for im in range(int(nbcm)):
            Diag = diag(CM[:, Range])
            On = On + (Diag * Diag).sum(axis=0)
            Range = Range + m
        Off = (multiply(CM, CM).sum(axis=0)).sum(axis=1) - On
        # A statistically scaled threshold on `small" angles
        seuil = 1.0e-6 / sqrt(T)
        # sweep number
        encore = True
        sweep = 0
        # Total number of rotations
        updates = 0
        # Number of rotations in a given seep
        upds = 0
        g = zeros([2, nbcm], dtype=float64)
        gg = zeros([2, 2], dtype=float64)
        G = zeros([2, 2], dtype=float64)
        c = 0
        s = 0
        ton = 0
        toff = 0
        theta = 0
        Gain = 0

        # Joint diagonalization proper
        # ============================
        if verbose: print("jade -> Contrast optimization by joint diagonalization")
        updates = updates + upds
        # while encore:
        #     encore = False
        #     if verbose: print("jade -> Sweep #%3d" % sweep)
        #     sweep = sweep + 1
        #     upds = 0
        #     Vkeep = V
        #
        #     for p in range(m - 1):
        #         for q in range(p + 1, m):
        #
        #             Ip = arange(p, m * nbcm, m)
        #             Iq = arange(q, m * nbcm, m)
        #
        #             # computation of Givens angle
        #             g = concatenate([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
        #             gg = dot(g, g.T)
        #             ton = gg[0, 0] - gg[1, 1]
        #             toff = gg[0, 1] + gg[1, 0]
        #             theta = 0.5 * arctan2(toff, ton + sqrt(ton * ton + toff * toff))
        #             Gain = (sqrt(ton * ton + toff * toff) - ton) / 4.0
        #
        #             # Givens update
        #             if abs(theta) > seuil:
        #                 encore = True
        #                 upds = upds + 1
        #                 c = cos(theta)
        #                 s = sin(theta)
        #                 G = matrix([[c, -s], [s, c]])
        #                 pair = array([p, q])
        #                 V[:, pair] = V[:, pair] * G
        #                 CM[pair, :] = G.T * CM[pair, :]
        #                 CM[:, concatenate([Ip, Iq])] = \
        #                     append(c * CM[:, Ip] + s * CM[:, Iq], -s * CM[:, Ip] + c * CM[:, Iq], \
        #                            axis=1)
        #                 On = On + Gain
        #                 Off = Off - Gain
        #
        #     if verbose: print("completed in %d rotations" % upds)
        #     updates = updates + upds



        # A separating matrix
        # ===================
        B = V.T * B

        # Permute the rows of the separating matrix B to get the most energetic
        # components first. Here the **signals** are normalized to unit variance.
        # Therefore, the sort is according to the norm of the columns of
        # A = pinv(B)

        if verbose: print("jade -> Sorting the components")

        A = pinv(B)

        keys = array(argsort(multiply(A, A).sum(axis=0)[0]))[0]
        B = B[keys, :]
        # % Is this smart ?
        B = B[::-1, :]

        if verbose: print("jade -> Fixing the signs")
        b = B[:, 0]
        # just a trick to deal with sign == 0
        signs = array(sign(sign(b) + 0.1).T)[0]
        B = diag(signs) * B

        return B.astype(origtype)














