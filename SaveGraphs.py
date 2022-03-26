import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

class Plots:
    figNumber =0
    ColorEstimatedFPS = 0
    IREstimatedFPS = 0
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
    def plotAllinOne(self,Savefilepath, blue,green,red, grey,ir,filename,windowsize,xlabel,ylabel):

        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()

        Colortx, IRtx = self.getTimeStep(red, ir)

        self.figNumber = self.figNumber + 1
        plt.figure(self.figNumber, figsize=(20, 11))

        # plt.tick_params(axis='both', which='major', labelsize=16)
        plt.rcParams.update({'font.size': 16})

        plt.tick_params(axis='both', which='major', labelsize=16)

        plt.rc('font', size=16)  # controls default text sizes
        plt.rc('axes', titlesize=16)  # fontsize of the axes title
        plt.rc('axes', labelsize=16)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=16)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=16)  # fontsize of the tick labels
        plt.plot(Colortx, blue, 'blue')
        plt.plot(Colortx, green, 'green')
        plt.plot(Colortx, red, 'red')
        plt.plot(Colortx, grey, 'gray')
        plt.plot(IRtx, ir, 'black')

        plt.legend(["blue", "green", "red","grey","Ir"])
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)

        plt.savefig(Savefilepath + filename + '.png', dpi=300, bbox_inches='tight')#

    def getTimeStep(self, signal, signalIR):
        # tx = np.arange(0, windowsize, T)

        ##For Color
        signalL = len(signal)
        T = 1 / self.ColorEstimatedFPS  # sampling frequency
        max_time = signalL / self.ColorEstimatedFPS
        Colortx = np.linspace(0, max_time, signalL) #time_steps

        ##For IR
        signalL = len(signalIR)
        T = 1 / self.IREstimatedFPS  # sampling frequency
        max_time = signalL / self.IREstimatedFPS
        IRtx = np.linspace(0, max_time, signalL) #time_steps

        return  Colortx, IRtx
    '''
    '''
    def plotGraphAllwithParam(self, Savefilepath, filename, timearrycolor, timeir, blue, green, red, grey, ir, xlabel,
                              ylabel):
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()

        Colortx, IRtx = self.getTimeStep(red, ir)

        fig, (ax1, ax2, ax3,ax4, ax5) = plt.subplots(5, sharex=True, figsize=(20, 11))

        # plt.xlabel(xlabel, fontsize = 16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.rcParams.update({'font.size': 18})

        fig.suptitle(filename)
        ax1.plot(Colortx, red, 'red', label='Red channel')
        ax1.legend(loc="upper right")
        ax2.plot(Colortx, green, 'green', label='Green channel')
        ax2.legend(loc="upper right")
        ax3.plot( Colortx,blue, 'blue',label = 'Blue channel')
        ax3.legend(loc="upper right")
        ax4.plot(Colortx, grey, 'gray', label='Grey channel')
        ax4.legend(loc="upper right")
        ax5.plot(IRtx, ir, 'black', label='IR channel')
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
    def PlotFFT(self,blue, green,red,grey,ir,colorfreq,irfreq,title,Savefilepath,filename):
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

        ax2.stem(colorfreq, np.abs(green), 'green',label="green")
        ax2.legend(loc="upper right")
        ax3.stem(colorfreq, np.abs(red), 'red',label="red")
        ax3.legend(loc="upper right")

        ax4.stem(colorfreq, np.abs(grey), 'gray',label="gray"  )#markerfmt=" ", basefmt="-b"
        ax4.legend(loc="upper right")
        ax5.stem(irfreq, np.abs(ir), 'black',label="IR" )#markerfmt=" ", basefmt="-b"
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

    def Genrateboxplot(self,data, savepath, position_type, datatype, xlabeltype,skintype):
        fig = plt.figure(figsize=(12, 8))
        plt.rc('font', size=18)
        ax = fig.add_subplot(111)

        # Creating axes instance
        bp = ax.boxplot(data, patch_artist=True,
                        notch='True', vert=0, showfliers=False)

        colors = ['#DCDDDE', '#B4B5B5',
                  '#8B8D8D', '#5A5F5F', '#242525', '#B4B5B5', '#5A5F5F']

        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        # changing color and linewidth of
        # whiskers
        for whisker in bp['whiskers']:
            whisker.set(color='#8B008B',
                        linewidth=1.5,
                        linestyle=":")

        # changing color and linewidth of
        # caps
        for cap in bp['caps']:
            cap.set(color='#8B008B',
                    linewidth=2)

        # changing color and linewidth of
        # medians
        for median in bp['medians']:
            median.set(color='red',
                       linewidth=3)

        # changing style of fliers
        for flier in bp['fliers']:
            flier.set(marker='D',
                      color='#e7298a',
                      alpha=0.5)

        # x-axis labels
        ax.set_yticklabels(['FastICA', 'None',
                            'PCA', 'PCAICA', 'Spectralembedding','Jade'])

        # Adding title
        # plt.title("Boxplot showing " + xlabeltype + " differences among various algorithms")
        plt.xlabel(xlabeltype + 'Difference (Commercial - ARPOS)')
        plt.ylabel('Algorithms')
        # Removing top axes and right axes
        # ticks
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.tight_layout()

        # show plot
        path = savepath + position_type + "_boxplot_"+ skintype+"_" + datatype + ".png"
        plt.savefig(path)  # show()

    ###Results

    def GenerateObservedvsActual(self, Actual_HR_AllValues_Resting, Observed_HR_AllValues_Resting,
                                 path,fileName,rvalue):
        ###PLOT chart3
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()

        actual = []
        observed = []

        actual = Actual_HR_AllValues_Resting
        observed = Observed_HR_AllValues_Resting

        actualArry = []
        # iterate over the list
        for val in actual:
            actualArry.append(int(float(val)))

        observedArry = []
        # iterate over the list
        for val in observed:
            observedArry.append(int(float(val)))

        rng = np.random.RandomState(0)
        sizes = 1000 * rng.rand(len(Actual_HR_AllValues_Resting))
        true_value = actualArry
        observed_value = observedArry
        plt.figure(figsize=(10, 10))
        plt.rc('font', size=20)
        plt.scatter(true_value, observed_value, c='crimson', s=sizes, alpha=0.3)


        # plt.yscale('log')
        # plt.xscale('log')

        p1 = max(max(observed_value), max(true_value))
        p2 = min(min(observed_value), min(true_value))

        plt.plot([p1, p2], [p1, p2], 'b-')

        vitaltype ="Heart Rate (bpm)"
        if(fileName.__contains__("SPO")):
            vitaltype = "SPO %"
        plt.xlabel('Commercial '+vitaltype, fontsize=20)
        plt.ylabel('ARPOS '+vitaltype, fontsize=20)
        # plt.title("plot with r vale " + str(rvalue))

        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.tick_params(axis='both', which='minor', labelsize=18)
        plt.axis('equal')
        # plt.legend(['data', 'line-regression r={}'.format(rvalue)], 'best')
        # plt.text(0, 1, 'r = %0.2f' % rvalue)
        plt.savefig(path + fileName + "_ActualvsObserved"  + ".png")
        plt.close()

    def Historgram(self,x1,x2,x3,x4,x1name,x2name,x3name,x4name,SavePATH,filename):

        ##General over all
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)


        # kwargs = dict(alpha=0.5, bins=100)

        fig = plt.figure(figsize=(16, 12))
        # plt.rc('axes', labelsize=16)
        plt.rc('font', size=18)
        # plt.rc('xtick', labelsize=16)
        # plt.rc('ytick', labelsize=16)
        plt.tight_layout()
        plt.hist(x1, **kwargs, color='blue', label=x1name)
        plt.hist(x3, **kwargs, color='red', label=x3name)
        plt.hist(x2, **kwargs, color='green', label=x2name)
        plt.hist(x4, **kwargs, color='purple', label=x4name)
        plt.gca().set(title='HR difference among different commercial pulse oximeter devices', ylabel='Frequency',
                      xlabel='HR (BPM) Difference (Commericial - ARPOS)')

        plt.legend()
        plt.savefig(SavePATH + filename+".png")  # Save here
        plt.close()
        plt.clf()
