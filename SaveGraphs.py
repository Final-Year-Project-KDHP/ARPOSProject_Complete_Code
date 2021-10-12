import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

class Plots:
    figNumber =0

    def plotSingle(self,Savefilepath,x,y,filename, color,fig):
        plt.ioff()
        self.figNumber = self.figNumber + 1
        # Plot the FFT power3
        plt.figure(fig)
        plt.plot(x, y, color)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('power')

        plt.savefig(Savefilepath + filename + "-" + color + '.png')

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


        # plt.switch_backend('agg')
        # plt.ioff()
        # plt.rcParams.update({'figure.max_open_warning': 0})
        # plt.clf()
        # self.figNumber = self.figNumber + 1
        # plt.figure(self.figNumber, figsize=(20, 11))
        # # plt.tick_params(axis='both', which='major', labelsize=16)
        # plt.rcParams.update({'font.size': 18})
        #
        # plt.tick_params(axis='both', which='major', labelsize=16)
        #
        # plt.rc('font', size=16)  # controls default text sizes
        # plt.rc('axes', titlesize=16)  # fontsize of the axes title
        # plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
        # plt.rc('xtick', labelsize=16)  # fontsize of the tick labels
        # plt.rc('ytick', labelsize=16)  # fontsize of the tick labels
        #
        # plt.plot(blue, 'blue')
        # plt.plot(green, 'green')
        # plt.plot(red, 'red')
        # plt.plot(grey, 'gray')
        # plt.plot(ir, 'black')
        # plt.legend(["blue", "green", "red", "grey", "Ir"])
        # plt.legend(loc="upper right")
        #
        # plt.xlabel("xlabel", fontsize=18)
        # plt.ylabel("ylabel", fontsize=18)
        #
        # plt.savefig(Savefilepath + filename + '.png', dpi=300, bbox_inches='tight')


    def plotGraphRGBwithParam(self, Savefilepath,filename,timearrycolor, blue,green,red, xlabel, ylabel):

        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        fig.suptitle(filename)
        ax1.plot(timearrycolor, red, 'red')
        ax2.plot(timearrycolor, green, 'green')
        ax3.plot(timearrycolor, blue, 'blue')
        plt.xlabel(xlabel)
        plt.xlabel(ylabel)
        plt.ioff()
        plt.savefig(Savefilepath  + filename + '.png', dpi=300, bbox_inches='tight')

    def plotGraphGIrwithParam(self, Savefilepath,filename,timearrycolor,timeir, grey,ir, xlabel, ylabel):

        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig.suptitle(filename)
        ax1.plot(timearrycolor, grey, 'gray')
        ax2.plot(timeir, ir, 'black')
        plt.xlabel(xlabel)
        plt.xlabel(ylabel)
        plt.ioff()
        plt.savefig(Savefilepath  + filename + '.png', dpi=300, bbox_inches='tight')



        # plt.figure(4)
        #
        # plt.subplot(5, 1, 1)
        # plt.plot(self.X[:, 4], color='black')
        # plt.title('IR')
        #
        # plt.subplot(5, 1, 2)
        # plt.plot(self.X[:, 3], color='gray')
        # plt.title('Grey')
        #
        # plt.subplot(5, 1, 3)
        # plt.plot(self.X[:, 2], color='red')
        # plt.title('Red')
        #
        # plt.subplot(5, 1, 4)
        # plt.plot(self.X[:, 1], color='green')
        # plt.title('Green')
        #
        # plt.subplot(5, 1, 5)
        # plt.plot(self.X[:, 0], color='blue')
        # plt.title('Blue')
        #
        # plt.ylabel("Amplitude")
        # plt.xlabel( "No of Frames" )
        #
        # plt.figure(5)
        # plt.subplot(5, 1, 1)
        # plt.plot(self.S_[:, 4], color='black')
        # plt.title('IR ICA')
        #
        # plt.subplot(5, 1, 2)
        # plt.plot(self.S_[:, 3], color='gray')
        # plt.title('Grey ICA')
        #
        # plt.subplot(5, 1, 3)
        # plt.plot(self.S_[:, 2], color='red')
        # plt.title('Red ICA')
        #
        # plt.subplot(5, 1, 4)
        # plt.plot(self.S_[:, 1], color='green')
        # plt.title('Green ICA')
        #
        # plt.subplot(5, 1, 5)
        # plt.plot(self.S_[:, 0], color='blue')
        # plt.title('Blue ICA')
        # plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)
        #
        # plt.ylabel("Amplitude")
        # plt.xlabel( "No of Frames" )
        # plt.show()
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
        # ax1.stem(freq, np.abs(blue), 'blue',label="blue")
        # ax1.legend(loc="upper right")
        ax2.stem(freq, np.abs(green), 'green',label="green")
        ax2.legend(loc="upper right")
        ax3.stem(freq, np.abs(red), 'red',label="red")
        ax3.legend(loc="upper right")
        # ax3.stem(freq, np.abs(blue), 'blue', \
        #          markerfmt=" ", basefmt="-b")
        ax4.stem(freq, np.abs(grey), 'gray',label="gray"  )#markerfmt=" ", basefmt="-b"
        ax4.legend(loc="upper right")
        ax5.stem(freq, np.abs(ir), 'black',label="IR" )#markerfmt=" ", basefmt="-b"
        ax5.legend(loc="upper right")

        plt.xlabel('Freq (Hz)', fontsize=16)
        # ax3.ylabel()
        ax4.set_ylabel('FFT Amplitude |X(freq)|', fontsize=16)
        plt.xlim(0, 5)
        #plt.clf()
        # plt.show()
        plt.savefig(Savefilepath + filename + '.png')#, bbox_inches='tight', dpi=300
        # plt.close(fig)


        # plt.figure(figsize=(12, 6))
        # plt.subplot(121)
        #
        # plt.stem(self.freq, singal, 'b', \
        #          markerfmt=" ", basefmt="-b")
        # plt.xlabel('Freq (Hz)')
        # plt.ylabel('FFT Amplitude |X(freq)|')
        # plt.xlim(0, 10)
        # plt.ylim(0, 20000)
        #
        # plt.subplot(122)
        # plt.plot( ifft(singal), 'r')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.tight_layout()
        # plt.show()

    def PlotFFT3(self,blue, green,red,grey,ir,freq,title,Savefilepath,filename):
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        fig, ( ax2, ax3, ax4,ax5) = plt.subplots(4, sharex=True,figsize=(20,11))
        fig.suptitle(title)
        plt.tight_layout()
        # ax1.title.set_text('Blue')
        ax2.title.set_text('Green')
        ax3.title.set_text('Red')
        ax4.title.set_text('Gray')
        ax5.title.set_text('IR')
        ax2.plot(freq, np.abs(green), 'green',label="green")
        ax2.legend(loc="upper right")
        ax3.plot(freq, np.abs(red), 'red',label="red")
        ax3.legend(loc="upper right")
        #          markerfmt=" ", basefmt="-b")
        ax4.plot(freq, np.abs(grey), 'gray',label="gray"  )#markerfmt=" ", basefmt="-b"
        ax4.legend(loc="upper right")
        ax5.plot(freq, np.abs(ir), 'black',label="IR" )#markerfmt=" ", basefmt="-b"
        ax5.legend(loc="upper right")

        plt.xlabel('Freq (Hz)', fontsize=16)
        # ax3.ylabel()
        ax4.set_ylabel('FFT Amplitude |X(freq)|', fontsize=16)
        plt.xlim(0, 5)
        #plt.clf()
        # plt.show()
        plt.savefig(Savefilepath + filename + '.png')#, bbox_inches='tight', dpi=300
        # plt.close(fig)


        # plt.figure(figsize=(12, 6))
        # plt.subplot(121)
        #
        # plt.stem(self.freq, singal, 'b', \
        #          markerfmt=" ", basefmt="-b")
        # plt.xlabel('Freq (Hz)')
        # plt.ylabel('FFT Amplitude |X(freq)|')
        # plt.xlim(0, 10)
        # plt.ylim(0, 20000)
        #
        # plt.subplot(122)
        # plt.plot( ifft(singal), 'r')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Amplitude')
        # plt.tight_layout()
        # plt.show()
    def plotGraph4withParam(self, Savefilepath,filename,timearrycolor,timeir, blue,green,red, ir,xlabel, ylabel):

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)
        fig.suptitle(filename)
        ax1.plot(timearrycolor, red, 'red')
        ax2.plot(timearrycolor, green, 'green')
        ax3.plot(timearrycolor, blue, 'blue')
        ax4.plot(timeir, ir, 'black')
        plt.xlabel(xlabel)
        plt.xlabel(ylabel)
        plt.ioff()
        plt.savefig(Savefilepath  + filename + '.png', dpi=300, bbox_inches='tight')


    def plotGraphRGBIrParam(self, Savefilepath,filename,timearrycolor,timeir, blue,green,red, ir,xlabel, ylabel,title):
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        fig, (ax1, ax2, ax3, ax5) = plt.subplots(4, sharex=True)
        fig.suptitle(title)
        #plt.title(title)
        plt.tight_layout()
        ax1.plot(timearrycolor, red, 'red')
        ax2.plot(timearrycolor, green, 'green')
        ax3.plot(timearrycolor, blue, 'blue')
        ax5.plot(timeir, ir, 'black')
        plt.xlabel(xlabel)
        plt.xlabel(ylabel)
        plt.savefig(Savefilepath  + filename + '.png', dpi=300, bbox_inches='tight')
        plt.clf()
        plt.close(fig)

    def plotAll(self, Savefilepath,filename, blue,green,red, grey,Irchannel,xlabel, ylabel,title,figno,components,ignoregray):
        plt.switch_backend('agg')
        plt.ioff()
        plt.clf()

        plt.figure(figno)
        plt.suptitle(title)
        plt.rcParams.update({'figure.max_open_warning': 0})
        # plt.tight_layout()
        plt.subplot(components, 1, 1)
        plt.plot(blue, color = 'blue')
        plt.title('Blue original')
        plt.subplot(components, 1, 2)
        plt.plot(green, color = 'green')
        plt.title('Green original')
        plt.subplot(components, 1, 3)
        plt.plot(red, color = 'red')
        plt.title('Red original')
        if(not ignoregray):
            plt.subplot(components, 1, components-1)
            plt.plot(grey, color = 'gray')
            plt.title('gray original')
        plt.subplot(components, 1, components,sharex=True)
        plt.plot(Irchannel, color = 'black')
        plt.title('Ir original')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(Savefilepath  + filename + '.png', dpi=300, bbox_inches='tight')
        plt.close(figno)

    def plotGraphAllwithParam(self, Savefilepath,filename,timearrycolor,timeir, blue,green,red, grey,ir,xlabel, ylabel,fps,windowsize):
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
        # plt.tick_params(axis='both', which='major', labelsize=16)
        #
        # plt.rc('font', size=16)  # controls default text sizes
        # plt.rc('axes', titlesize=16)  # fontsize of the axes title
        # plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
        # plt.rc('xtick', labelsize=16)  # fontsize of the tick labels
        # plt.rc('ytick', labelsize=16)  # fontsize of the tick labels

        fig.suptitle(filename)
        ax1.plot(tx, red, 'red', label='Red channel')
        ax1.legend(loc="upper right")
        ax2.plot(tx,green, 'green', label='Green channel')
        ax2.legend(loc="upper right")
        # ax3.plot( blue, 'blue',label = 'Blue channel')
        # ax3.legend(loc="upper right")
        ax4.plot(tx,grey, 'gray', label='Grey channel')
        ax4.legend(loc="upper right")
        ax5.plot(tx,ir, 'black', label='IR channel')
        ax5.legend(loc="upper right")

        # Set the tick labels font
        for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(16)

        # Set the tick labels font
        for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(16)

        # # Set the tick labels font
        # for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        #     label.set_fontname('Arial')
        #     label.set_fontsize(16)

        # Set the tick labels font
        for label in (ax4.get_xticklabels() + ax4.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(16)

        # Set the tick labels font
        for label in (ax5.get_xticklabels() + ax5.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(16)
        # ax1.set_xlabel("Heart Rate Difference", fontsize=16)
        # Removing top axes and right axes
        # ticks
        # ax1.get_xaxis().tick_bottom()
        # ax1.get_yaxis().tick_bottom()
        ax4.set_ylabel(ylabel, fontsize=18)
        plt.xlabel(xlabel, fontsize=18)
        # plt.title(title)
        # plt.legend()
        plt.savefig(Savefilepath + filename + '.png')  # , dpi=300, bbox_inches='tight'
        #
        # ax1.plot(timearrycolor, red, 'red')
        # ax2.plot(timearrycolor, green, 'green')
        # ax3.plot(timearrycolor, blue, 'blue')
        # ax4.plot(timearrycolor, grey, 'gray')
        # ax5.plot(timeir, ir, 'black')
        # plt.xlabel(xlabel)
        # plt.xlabel(ylabel)
        # plt.savefig(Savefilepath  + filename + '.png', dpi=300, bbox_inches='tight')
        # plt.clf()
        # plt.close(fig)

        # fig = plt.figure(fig,figsize=(9, 6))
        # sub1 = fig.add_subplot(5, 1, 1)
        # plt.plot(timearrycolor, blue, color='blue' )
        # sub2 = fig.add_subplot(5, 1, 2)
        # plt.plot(timearrycolor, green, color='green')
        # sub3 = fig.add_subplot(5, 1, 3)
        # plt.plot(timearrycolor, red, color='red')
        # sub4 = fig.add_subplot(5, 1, 4)
        # plt.plot(timearrycolor, grey, color='grey')
        # sub5 = fig.add_subplot(5, 1, 5)
        # plt.plot(timeir, ir, color='black')
        # plt.savefig(Savefilepath  + filename + '.png', dpi=300, bbox_inches='tight')

        # fig = plt.figure()
        # gs = fig.add_gridspec(5, hspace=0)
        # axs = gs.subplots(sharex=True, sharey=True)
        # fig.suptitle('Sharing both axes')
        # axs[0].plot(self.timecolorCount, self.red,'red' )
        # axs[1].plot(self.timecolorCount, self.green, 'green')
        # axs[2].plot(self.timecolorCount, self.blue,'blue')
        # axs[3].plot(self.timecolorCount, self.grey,'gray')
        # axs[4].plot(self.timeirCount, self.Irchannel,'black')
        # plt.savefig(self.SavefigPath + 'Inititalred.png')

        # fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True)
        # fig.suptitle('Aligning x-axis using sharex')
        # ax1.plot(self.timecolorCount, self.red, 'red')
        # ax2.plot(self.timecolorCount, self.green, 'green')
        # ax3.plot(self.timecolorCount, self.blue, 'blue')
        # ax4.plot(self.timecolorCount, self.grey, 'gray')
        # ax5.plot(self.timeirCount, self.Irchannel, 'black')
        # plt.savefig(self.SavefigPath + 'Inititalred.png')

    # def plotAllinOne(self,Savefilepath,filename, blue,green,red, grey,ir,xlabel,ylabel,fps,windowsize):
    #     plt.switch_backend('agg')
    #     plt.ioff()
    #     plt.rcParams.update({'figure.max_open_warning': 0})
    #     plt.clf()
    #
    #     T = 1 / fps  # sampling frequency
    #     tx = np.arange(0, windowsize, T)
    #
    #     self.figNumber = self.figNumber + 1
    #     plt.figure(self.figNumber)
    #     plt.plot(tx,blue,  'blue')
    #     plt.plot(tx,green,  'green')
    #     plt.plot(tx,red,  'red')
    #     plt.plot(tx,grey,  'gray')
    #     plt.plot(tx,ir,  'black')
    #     plt.legend(["blue", "green", "red","grey","Ir"])
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #
    #     plt.savefig(Savefilepath + filename + '.png')


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

    def PlotFFT2(self, blue, green, red, grey, ir, freq, title, Savefilepath, filename):
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.rcParams.update({'font.size': 18})

        plt.figure(6, figsize=(20, 11))
        plt.subplot(4, 1, 1)
        plt.plot(freq,red, color='red')
        plt.title('Red FFT')

        plt.subplot(4, 1, 2)
        plt.plot(freq,green, color='green')
        plt.title('Green FFT')

        plt.subplot(4, 1, 3)
        plt.plot(freq,grey, color='gray')
        plt.title('Gray FFT')

        plt.subplot(4, 1, 4)
        plt.plot(freq,ir, color='black')
        plt.title('IR FFT')
        # plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.46)

        plt.xlim(0, 5)
        plt.xlabel('Freq (Hz)', fontsize=16)
        plt.ylabel('FFT Amplitude |X(freq)|')
        plt.savefig(Savefilepath + filename + '.png')

        # fig, (ax2, ax3, ax4, ax5) = plt.subplots(4, sharex=True, figsize=(20, 11))
        # fig.suptitle(title)
        # plt.tight_layout()
        # # ax1.title.set_text('Blue')
        # ax2.title.set_text('Green')
        # ax3.title.set_text('Red')
        # ax4.title.set_text('Gray')
        # ax5.title.set_text('IR')
        # # ax1.stem(freq, np.abs(blue), 'blue',label="blue")
        # # ax1.legend(loc="upper right")
        # ax2.plot(green, 'green', label="green")
        # ax2.legend(loc="upper right")
        # ax3.plot( red, 'red', label="red")
        # ax3.legend(loc="upper right")
        # # ax3.stem(freq, np.abs(blue), 'blue', \
        # #          markerfmt=" ", basefmt="-b")
        # ax4.plot(grey, 'gray', label="gray")  # markerfmt=" ", basefmt="-b"
        # ax4.legend(loc="upper right")
        # ax5.plot( ir, 'black', label="IR")  # markerfmt=" ", basefmt="-b"
        # ax5.legend(loc="upper right")
        #
        # plt.xlabel('Freq (Hz)')
        # # ax3.ylabel()
        # ax4.set_ylabel('FFT Amplitude |X(freq)|', fontsize=16)
        # plt.xlim(0, 6)
        # # plt.clf()
        # # plt.show()
        # plt.savefig(Savefilepath + filename + '.png')  # , bbox_inches='tight', dpi=300
        # plt.close(fig)

    def plotGraphAllWithoutTimewithParam(self, Savefilepath,filename, blue,green,red, grey,ir,xlabel,ylabel):
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()

        fig, (ax1, ax2,  ax4, ax5) = plt.subplots(4, sharex=True,figsize=(20,11))


        # plt.xlabel(xlabel, fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.rcParams.update({'font.size': 18})
        # plt.tick_params(axis='both', which='major', labelsize=16)
        #
        # plt.rc('font', size=16)  # controls default text sizes
        # plt.rc('axes', titlesize=16)  # fontsize of the axes title
        # plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
        # plt.rc('xtick', labelsize=16)  # fontsize of the tick labels
        # plt.rc('ytick', labelsize=16)  # fontsize of the tick labels

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

        # # Set the tick labels font
        # for label in (ax3.get_xticklabels() + ax3.get_yticklabels()):
        #     label.set_fontname('Arial')
        #     label.set_fontsize(16)

        # Set the tick labels font
        for label in (ax4.get_xticklabels() + ax4.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(16)

        # Set the tick labels font
        for label in (ax5.get_xticklabels() + ax5.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(16)
        # ax1.set_xlabel("Heart Rate Difference", fontsize=16)
        # Removing top axes and right axes
        # ticks
        # ax1.get_xaxis().tick_bottom()
        # ax1.get_yaxis().tick_bottom()
        ax4.set_ylabel(ylabel, fontsize=18)
        plt.xlabel(xlabel, fontsize = 18)
        # plt.title(title)
        # plt.legend()
        plt.savefig(Savefilepath  + filename + '.png')#, dpi=300, bbox_inches='tight'
        # plt.show()
        # fig = self.fig + 1
        # fig = plt.figure(fig,figsize=(9, 6))
        #
        # sub1 = fig.add_subplot(5, 1, 1)
        # plt.plot(blue, color='blue')
        #
        # sub2 = fig.add_subplot(5, 1, 2)
        # plt.plot(green, color='green')
        #
        # sub3 = fig.add_subplot(5, 1, 3)
        # plt.plot(red, color='red')
        #
        # sub4 = fig.add_subplot(5, 1, 4)
        # plt.plot( grey, color='grey')
        #
        # sub5 = fig.add_subplot(5, 1, 5)
        # plt.plot( ir, color='black')
        #
        # plt.savefig(Savefilepath + filename + '.png', dpi=300, bbox_inches='tight')


    def plotGraphAllWithoutTimewithParamRGB(self, Savefilepath,filename, blue,green,red):
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
        fig.suptitle(filename)
        ax1.plot( red, 'red')
        ax2.plot( green, 'green')
        ax3.plot( blue, 'blue')
        plt.ioff()
        plt.savefig(Savefilepath  + filename + '.png', dpi=300, bbox_inches='tight')

    def plotGraphAllWithoutTimewithParamGIR(self, Savefilepath,filename, grey,ir):
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        fig.suptitle(filename)
        ax1.plot( grey, 'red')
        ax2.plot( ir, 'green')
        plt.ioff()
        plt.savefig(Savefilepath  + filename + '.png', dpi=300, bbox_inches='tight')