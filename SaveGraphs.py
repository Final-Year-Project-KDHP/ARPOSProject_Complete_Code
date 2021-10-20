import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

class Plots:
    figNumber =0

    '''
    '''
    def plotGraphAllWithoutTimewithParam(self, Savefilepath,filename, blue,green,red, grey,ir,xlabel,ylabel):
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()

        fig, (ax1, ax2,  ax4, ax5) = plt.subplots(4, sharex=True,figsize=(20,11))


        # plt.xlabel(xlabel, fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.rcParams.update({'font.size': 18})
        fig.suptitle(filename)
        ax1.plot( red, 'red',label = 'Red channel')
        ax1.legend(loc="upper right")
        ax2.plot( green, 'green',label = 'Green channel')
        ax2.legend(loc="upper right")
        # ax3.plot( blue, 'blue',label = 'Blue channel')
        # ax3.legend(loc="upper right")
        ax4.plot( grey, 'gray',label = 'Grey channel')
        ax4.legend(loc="upper right")
        ax5.plot( ir, 'black',label = 'IR channel')
        ax5.legend(loc="upper right")

        # Set the tick labels font
        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(16)

        # Set the tick labels font
        for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(16)

        # Set the tick labels font
        for label in (ax4.get_xticklabels() + ax4.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(16)

        # Set the tick labels font
        for label in (ax5.get_xticklabels() + ax5.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(16)

        ax4.set_ylabel(ylabel, fontsize=18)
        plt.xlabel(xlabel, fontsize = 18)

        plt.savefig(Savefilepath  + filename + '.png')#, dpi=300, bbox_inches='tight'

    '''
        '''
    def plotAllinOneWithoutTime(self,Savefilepath,filename, blue,green,red, grey,ir,xlabel,ylabel):
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        self.figNumber = self.figNumber + 1
        plt.figure(self.figNumber, figsize=(20, 11))
        # plt.tick_params(axis='both', which='major', labelsize=16)
        plt.rcParams.update({'font.size': 18})

        plt.tick_params(axis='both', which='major', labelsize=16)

        plt.rc('font', size=16)  # controls default text sizes
        plt.rc('axes', titlesize=16)  # fontsize of the axes title
        plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=16)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=16)  # fontsize of the tick labels

        plt.plot(blue,  'blue', label="blue")
        plt.plot(green,  'green', label="green")
        plt.plot(red,  'red', label="red")
        plt.plot(grey,  'gray', label="gray")
        plt.plot(ir,  'black', label="black")
        # plt.legend(["blue", "green", "red","grey","Ir"])
        plt.legend(loc="upper right")

        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)


        plt.savefig(Savefilepath + filename + '.png', dpi=300, bbox_inches='tight')

    '''
    '''
    def plotAllinOne(self,Savefilepath, blue,green,red, grey,ir,filename,fps,windowsize,xlabel,ylabel):

        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        T = 1 / fps  # sampling frequency
        tx = np.arange(0, windowsize, T)

        self.figNumber = self.figNumber + 1
        plt.figure(self.figNumber, figsize=(20, 16))

        # plt.tick_params(axis='both', which='major', labelsize=16)
        plt.rcParams.update({'font.size': 18})

        plt.tick_params(axis='both', which='major', labelsize=16)

        plt.rc('font', size=16)  # controls default text sizes
        plt.rc('axes', titlesize=16)  # fontsize of the axes title
        plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=16)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=16)  # fontsize of the tick labels
        plt.plot(tx, blue, 'blue')
        plt.plot(tx, green, 'green')
        plt.plot(tx, red, 'red')
        plt.plot(tx, grey, 'gray')
        plt.plot(tx, ir, 'black')

        plt.legend(["blue", "green", "red","grey","Ir"])
        plt.xlabel(xlabel, fontsize=18)
        plt.ylabel(ylabel, fontsize=18)

        plt.savefig(Savefilepath + filename + '.png')#, dpi=300, bbox_inches='tight'

    '''
    '''
    def plotGraphAllwithParam(self, Savefilepath, filename, timearrycolor, timeir, blue, green, red, grey, ir, xlabel,
                              ylabel, fps, windowsize):
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()

        T = 1 / fps  # sampling frequency
        tx = np.arange(0, windowsize, T)

        fig, (ax1, ax2, ax4, ax5) = plt.subplots(4, sharex=True, figsize=(20, 11))

        # plt.xlabel(xlabel, fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.rcParams.update({'font.size': 18})

        fig.suptitle(filename)
        ax1.plot(tx, red, 'red', label='Red channel')
        ax1.legend(loc="upper right")
        ax2.plot(tx, green, 'green', label='Green channel')
        ax2.legend(loc="upper right")
        # ax3.plot( blue, 'blue',label = 'Blue channel')
        # ax3.legend(loc="upper right")
        ax4.plot(tx, grey, 'gray', label='Grey channel')
        ax4.legend(loc="upper right")
        ax5.plot(tx, ir, 'black', label='IR channel')
        ax5.legend(loc="upper right")

        # Set the tick labels font
        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(16)

        # Set the tick labels font
        for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(16)

        # Set the tick labels font
        for label in (ax4.get_xticklabels() + ax4.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(16)

        # Set the tick labels font
        for label in (ax5.get_xticklabels() + ax5.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(16)

        ax4.set_ylabel(ylabel, fontsize=18)
        plt.xlabel(xlabel, fontsize=18)
        # plt.title(title)
        # plt.legend()
        plt.savefig(Savefilepath + filename + '.png')

    '''
    '''
    def PlotFFT(self,blue, green,red,grey,ir,freq,title,Savefilepath,filename):
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()

        # plt.xlabel(xlabel, fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.rcParams.update({'font.size': 18})
        fig, ( ax2, ax3, ax4,ax5) = plt.subplots(4, sharex=True,figsize=(20,11))
        fig.suptitle(title)
        plt.tight_layout()
        # ax1.title.set_text('Blue')
        ax2.title.set_text('Green')
        ax3.title.set_text('Red')
        ax4.title.set_text('Gray')
        ax5.title.set_text('IR')

        ax2.stem(freq, np.abs(green), 'green',label="green")
        ax2.legend(loc="upper right")
        ax3.stem(freq, np.abs(red), 'red',label="red")
        ax3.legend(loc="upper right")

        ax4.stem(freq, np.abs(grey), 'gray',label="gray"  )#markerfmt=" ", basefmt="-b"
        ax4.legend(loc="upper right")
        ax5.stem(freq, np.abs(ir), 'black',label="IR" )#markerfmt=" ", basefmt="-b"
        ax5.legend(loc="upper right")

        plt.xlabel('Freq (Hz)', fontsize=16)

        ax4.set_ylabel('FFT Amplitude |X(freq)|', fontsize=16)
        plt.xlim(0, 5)

        plt.savefig(Savefilepath + filename + '.png')#, bbox_inches='tight', dpi=300

    '''
    '''
    def plotSingle(self,Savefilepath,x,y,filename, color,fig):
        plt.ioff()
        self.figNumber = self.figNumber + 1
        plt.figure(fig)
        plt.plot(x, y, color)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('power')

        plt.savefig(Savefilepath + filename + "-" + color + '.png')