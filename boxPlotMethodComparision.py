
import os
import sys
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing._private.parameterized import param

from Configurations import Configurations
from FileIO import FileIO


class BoxPlot:
    objConfig =None
    #Constructor
    def __init__(self, skinGroup='None'):
        self.objConfig = Configurations(True, skinGroup)

    ticks = ['None', 'FastICA', 'ICAPCA', 'PCA','Jade', 'Gr']
    hrfileName = 'HeartRate'
    # fftTypeListNames = ["rfft + abs + rfftFreq", "fftpack fft", "rfft + abs + power + rfftFreq", "fft",
    #                     "fft + sqrt + abs", "fft + sqrt + abs + fftshift",
    #                     "rfft"]  # , "M4","M5", "M6" with butter filter try
    # filtertypeListNames = ["0 below and above", "only below", "cutoff[bound_high:-bound_high]",
    #                        "butter_bandpass_filter", "None", "above and below fft and frequency filter", "7"]
    # preprocessesNames = ["None", "Detrend + Interpolate + Normalise", "Detrend + Interpolate +  Normalise",
    #                      "Detrend + SmoothFilter +  Normalise", "Normalise only", "6",
    #                      "7"]  # ?? without detrend and normalise?
    # resulttypeListNames = ["ManualIndexIteration for Peak",
    #                        "ManualIndexIteration for Peak  with channelArrays limited data", "Frequency for Peak",
    #                        "FrequencyBPM for Peak", "FrequencyBPM for Peak with channelArrays limited data"
    #     , "Frequency for Peak with channelArrays limited data", "find_heart_rate loop and limits",
    #                        "find_heart_rate loop and limits with channelArrays limit data",
    #                        "Limited Arrays with ARGmax"]
    # SmoothenNames = ["False", "True"]

    def Getdata(self,filepath, differnece,isEntireSignal):
        Filedata = open(filepath, "r")
        data = Filedata.read().split("\n")
        generatedresult = []
        # for row in data:
        dataRow = data[0].split(",\t")
        #str(round(HrAvegrage)) + " ,\t" + str(round(heartRateValue)) + " ,\t" + str(difference)+ " ,\t" + str(diffNow)+ " ,\t" + str(regiontype)
        indexvalue = 2  # 0 index for windowCount(skip if not window file meaning entire signal), 1 index for GroundTruth, 2 index for generatedresult by arpos, 3 index for diference
        if(isEntireSignal):
            indexvalue=1
            if (differnece == True):
                indexvalue = 2
        else:
            if (differnece == True):
                indexvalue = 3

        generatedValue = dataRow[indexvalue]  # change index here
        # generatedresult.append(generatedValue)

        generatedValue = np.abs(int(generatedValue))
        return generatedValue

    def ExtractGrdata(self,loadpath, filename):  # "None", loadpath,methodtype
        filepath = loadpath + filename + ".txt"  # HRdata-M1_1_1

        Filedata = open(filepath, "r")
        data = Filedata.read().split("\n")
        generatedresult = []
        for row in data:
            dataRow = row.split(",\t")

            indexvalue = 1  # 0 index for windowCount, 1 index for GroundTruth, 2 index for generatedresult by arpos, 3 index for diference
            # if(differnece== True):
            #     #indexvalue = 3
            #     generatedValue = 0
            # else:
            generatedValue = dataRow[indexvalue]
            generatedresult.append(generatedValue)

        generatedresult = list(map(int, generatedresult))
        return generatedresult

    def GetgroundTruth(self,loadpath, filename):  # "None", loadpath,methodtype
        filepath = loadpath + filename + ".txt"  # HRdata-M1_1_1

        Filedata = open(filepath, "r")
        data = Filedata.read().split("\n")
        data = list(map(int, data))
        return data

    #
    # data =Getdata()
    # df = pd.DataFrame(data)
    # df.plot.box(grid='True')
    #
    # plt.show()
    # plt.figure()
    # plt.boxplot(data)
    # plt.tight_layout()
    # # show plot
    # plt.show()


    def GenerateBlandAltman_plot(self,savepath,filename,data,data_gr,methodtype,filtertype,resulttype,processtype,isSmooth, boxTitle,filename2):
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        df = pd.DataFrame({'GR': data_gr,
                           'dNone': data})
        plt.ioff()
        plt.clf()
        plt.title(filename2+ filename)
        # create Bland-Altman plot
        f, ax = plt.subplots(1, figsize=(8, 5))
        sm.graphics.mean_diff_plot(df.GR, df.dNone, ax=ax) #,df.ICA,df.ICAPCA,df.PCA

        # display Bland-Altman plot
        # plt.savefig(savepath + filename)
        path = savepath +filename2+ filename+ ".png"
            #    + "boxplot-" + methodtype + "_FilterType_" + str(filtertype) + "_ResultType_" + str(
            # resulttype) + ", pre-processing " + str(processtype) + "_Sm_" + str(
            # isSmooth) + "_Algo_" +filename + ".png"  # + "_" + str(resulttype)
        plt.savefig(path)  # show()
        plt.close(f)


    def Genrateboxplot(self,data,savepath,boxTitle,filename):
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        plt.ioff()
        plt.clf()
        fig = plt.figure(figsize=(20, 11))
        ax = fig.add_subplot(111)
        plt.rcParams.update({'font.size': 18})
        # plt.rc('xtick', labelsize=16)
        # plt.rc('ytick', labelsize=16)
        plt.tick_params(axis='both', which='major', labelsize=16)
        plt.rc('axes', labelsize=16)
        # plt.rc('ytick', labelsize=20)

        # Creating axes instance
        bp = ax.boxplot(data, patch_artist=True,
                        notch='True', vert=0, showmeans=True)

        colors = ['#DCDDDE', '#B4B5B5',
                  '#8B8D8D', '#5A5F5F', '#242525']

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
        ax.set_yticklabels(['None', 'FastICA',
                            'PCAICA', 'PCA', 'Jade', 'GroundTruth'])

        # Adding title
        # fftfilterIndex = [x for x in range(len(fftTypeList)) if fftTypeList[x] == methodtype][0]  # np.where(fftTypeList == methodtype)
        # filterIndex =[x for x in range(len(filtertypeList)) if filtertypeList[x] == int(filtertype)][0]   #  np.where(filtertypeList == filtertype)
        # resultIndex = [x for x in range(len(resulttypeList)) if resulttypeList[x] == int(resulttype)][0]   # np.where(resulttypeList == resulttype)
        # processtypeIndex = [x for x in range(len(preprocesses)) if preprocesses[x] == int(processtype)][0]   # np.where(preprocesses == processtype)
        # isSmoothIndex = [x for x in range(len(Smoothen)) if Smoothen[x] == isSmooth][0]   # np.where(Smoothen == isSmooth)

        # titlename = "Boxplot for [method]: " + fftTypeListNames[fftfilterIndex] + ", [FilterType]: " +  str(filtertypeListNames[filterIndex])+',\n '+\
        #             "[ResultType]: " + str(resulttypeListNames[resultIndex])  + ", [PreProcess]: " + str(preprocessesNames[processtypeIndex])+', '+ \
        #             "[Smoothen]: " + str(Smoothen[isSmoothIndex])
        plt.title(boxTitle)
        ax.set_xlabel("Heart Rate Difference", fontsize = 18)
        ax.set_ylabel("Types of Algorithms", fontsize = 18)
        # Removing top axes and right axes
        # ticks
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()


        # show plot
        path =savepath + filename + ".png"
              # "boxplot-" + methodtype + "_FilterType_" +  str(filtertype)+ "_ResultType_" + str(resulttype)  + ", pre-processing " + str(processtype)+ "_Sm_" + str(isSmooth)
        #+ "_" + str(resulttype)
        plt.savefig(path)  # show()
        plt.close()

    def SaveGroundTruth(self,savepath,filename, data):
        fHr = open(savepath + filename + ".txt", "w+")
        i = 0
        for item in data:
            if (i == len(data) - 1):
                fHr.write(str(item))
            else:
                fHr.write(str(item) + '\n')
            i = i + 1
        fHr.close()

    def GenerateGroundTruthFile(self):
        for participant_number in self.objConfig.ParticipantNumbers:

            for position in self.objConfig.hearratestatus:
                loadpath = self.objConfig.DiskPath + participant_number + '\\' + position + '\\'  + "None\\"
                savepath= self.objConfig.DiskPath + participant_number + '\\' + position + '\\'
                Grdata = self.ExtractGrdata(loadpath, "HRdata-M1_Fl_1_Rs_1")
                filename = "groundTruth"
                # if(not difference):
                #     filename = "groundTruthDifference"
                self.SaveGroundTruth(savepath,filename,Grdata)


    def RunBoxplotgeneration(self,difference):

        for participant_number in self.objConfig.ParticipantNumbers:

            for position in self.objConfig.hearratestatus:
                loadpath = self.objConfig.DiskPath + participant_number + '\\' + position + '\\'

                for resulttype in self.objConfig.resulttypeList:
                    resultpath = "ResultMethod_"
                    if (difference):
                        resultpath = "ResultDifferenceMethod_"
                    savepath = loadpath + "BoxPlots\\" + resultpath + str(resulttype) + '\\'

                    if not os.path.exists(savepath):
                        os.makedirs(savepath)

                    for methodtype in self.objConfig.fftTypeList:

                        for filtertype in self.objConfig.filtertypeList:

                            for processtype in self.objConfig.preprocesses:

                                for isSmooth in self.objConfig.Smoothen:

                                    data_d =self.Getdata("None", loadpath,methodtype,"HRdata",filtertype,resulttype,difference,processtype,isSmooth ) # filepath = loadpath  + algotype + '\\' + filename + "-"+ methodtype+ "_"+ filtertype + "_" + resulttype+ ".txt" #HRdata-M1_1_1
                                    data_a =self.Getdata("FastICA", loadpath,methodtype,"HRdata",filtertype,resulttype,difference,processtype,isSmooth)
                                    data_b =self.Getdata("ICAPCA", loadpath,methodtype,"HRdata",filtertype,resulttype,difference,processtype,isSmooth)
                                    data_c =self.Getdata("PCA", loadpath,methodtype,"HRdata",filtertype,resulttype,difference,processtype,isSmooth)


                                    filenamegr = "groundTruth"
                                    data_gr =[]
                                    if(difference):
                                        # filenamegr = "groundTruthDifference"
                                        for x in data_d:
                                            data_gr.append(0)
                                    else:
                                        data_gr =self.GetgroundTruth(loadpath,filenamegr)

                                    data = [data_d, data_a, data_b,data_c,data_gr]

                                    self.Genrateboxplot(data,savepath,methodtype,filtertype,resulttype,processtype,isSmooth)



    def RunBoxplotforCommonResultsonly(self,position,difference,fileholdingfilenames,isEntireSignal):
        #load filenames
        loadpathCommonfiles = self.objConfig.DiskPath + 'Result\\' + fileholdingfilenames + "_"+position  +".txt"
        # read data from files
        HrFiledata = open(loadpathCommonfiles, "r")
        FileNames = HrFiledata.read().split("\n")
        HrFiledata.close()

        for participant_number in self.objConfig.ParticipantNumbers:
            self.objConfig.setSavePath(participant_number,position)

            savepathBoxplots = self.objConfig.SavePath + "\\BoxPlots\\"
            savepathBlandAltmanplots = self.objConfig.SavePath + "\\BlandAltmanPlots\\"

            if not os.path.exists(savepathBoxplots):
                os.makedirs(savepathBoxplots)

            if not os.path.exists(savepathBlandAltmanplots):
                os.makedirs(savepathBlandAltmanplots)

            algoCount_FastICA = 0
            algoCount_None = 0
            algoCount_PCA = 0
            algoCount_PCAICA = 0
            algoCount_Jade = 0

            data_fastica = []
            data_icapca = []
            data_pca = []
            data_jade = []
            data_none = []

            for file_name in FileNames:
                # methodindex=0
                # if (containsAlgoName):
                #     algoname =fileDetails[0]
                #     methodindex =1
                #
                # methodtype = fileDetails[methodindex]
                # filtertype = fileDetails[methodindex+2]
                # resulttype = fileDetails[methodindex+4]
                # processtype = fileDetails[methodindex+6]
                # isSmooth = bool(fileDetails[methodindex+8].replace(".txt",""))

                #file location
                loadpath = self.objConfig.SavePath+  file_name + '\\' + self.hrfileName +'_'+ file_name+'.txt'
                #get data
                data = self.Getdata(loadpath, difference, isEntireSignal)

                if(file_name.__contains__('FastICA')):
                    data_fastica.append(data)
                elif (file_name.__contains__('ICAPCA')):
                    data_icapca.append(data)
                elif (file_name.__contains__('PCA')):
                    data_pca.append(data)
                elif (file_name.__contains__('Jade')):
                    data_jade.append(data)
                elif (file_name.__contains__('None')):
                    data_none.append(data)
                # loadpathICA = loadpath + "FastICA\\" + file_name
                # loadpathICAPCA = loadpath + "ICAPCA\\" + file_name
                # loadpathPCA = loadpath + "PCA\\" + file_name
                # loadpathJade = loadpath + "Jade\\" + file_name
                 # filepath = loadpath  + algotype + '\\' + filename + "-"+ methodtype+ "_"+ filtertype + "_" + resulttype+ ".txt" #HRdata-M1_1_1
                # data_fastica =self.Getdata("FastICA", loadpath,methodtype,"HRdata",filtertype,resulttype,difference,processtype,isSmooth)
                # data_icapca =self.Getdata("ICAPCA", loadpath,methodtype,"HRdata",filtertype,resulttype,difference,processtype,isSmooth)
                # data_pca =self.Getdata("PCA", loadpath,methodtype,"HRdata",filtertype,resulttype,difference,processtype,isSmooth)
                # data_jade = self.Getdata("Jade", loadpath, methodtype, "HRdata", filtertype, resulttype, difference,
                #                        processtype, isSmooth)

            filenamegr = "HR"
            data_gr =[]
            # if(difference):
            #     for x in data_none:
            #         data_gr.append(0)
            # else:
            data_gr =self.GetgroundTruth(self.objConfig.DiskPath + 'GroundTruthData\\' + participant_number+'\\' +position+'\\' ,filenamegr)

            avgFastIca= np.mean(data_fastica)
            avgNone= np.mean(data_none)
            avgPca= np.mean(data_pca)
            avgIcapca= np.mean(data_icapca)
            avgjade= np.mean(data_jade)
            minDiff = sys.float_info.max
            algotype = ""
            if(avgFastIca<minDiff and avgFastIca>=0):
                minDiff=avgFastIca
                algotype="FastICA"
            if (avgNone < minDiff and avgNone>=0):
                minDiff = avgNone
                algotype="None"
            if (avgPca < minDiff and avgPca>=0):
                minDiff = avgPca
                algotype="PCA"
            if (avgIcapca < minDiff and avgIcapca>=0):
                minDiff = avgIcapca
                algotype="PCAICA"
            if (avgjade < minDiff and avgjade>=0):
                minDiff = avgjade
                algotype="Jade"

            if(not (minDiff == sys.float_info.max)):
                data = [data_none, data_fastica, data_icapca,data_pca,data_jade,data_gr]

                boxTitle= "Best_Mean: " + str(minDiff) + " using " + str(algotype)+ "\n" \
                          + " (ICA: " + str(avgFastIca) + " , PCA: " + str(avgPca) + " , PCAICA: " + str(avgIcapca) + " , None: " + str(avgNone)+ " , Jade: " + str(avgjade) + ")"\
                          + "\n" + file_name

                self.Genrateboxplot(data,savepathBoxplots +algotype+"\\",boxTitle,file_name)


                if(algotype=="FastICA"):
                    self.GenerateBlandAltman_plot(savepathBlandAltmanplots+"FastICA\\", "_FastICA", data_fastica, data_gr,boxTitle,file_name)
                    algoCount_FastICA =algoCount_FastICA+1

                if (algotype=="None"):
                    self.GenerateBlandAltman_plot(savepathBlandAltmanplots+"None\\", "_None", data_none, data_gr, boxTitle,file_name)
                    algoCount_None =algoCount_None+1
                if (algotype=="PCA"):
                    self.GenerateBlandAltman_plot(savepathBlandAltmanplots+"PCA\\", "_PCA", data_pca, data_gr, boxTitle,file_name)
                    algoCount_PCA =algoCount_PCA+1
                if (algotype=="PCAICA"):
                    self.GenerateBlandAltman_plot(savepathBlandAltmanplots+"PCAICA\\", "_PCAICA", data_icapca, data_gr, boxTitle,file_name)
                    algoCount_PCAICA =algoCount_PCAICA+1

                if(algotype=="Jade"):
                    self.GenerateBlandAltman_plot(savepathBlandAltmanplots+"Jade\\", "_Jade", data_jade, data_gr, boxTitle,file_name)
                    algoCount_Jade =algoCount_Jade+1

        print(participant_number + " , FastICA count: " +str(algoCount_FastICA )+
              " , PCA count: " +str(algoCount_PCA )+
              " , PCAICA count: " +str(algoCount_PCAICA )+
              " , None count: " +str(algoCount_None )+
              " , Jade count: " +str(algoCount_Jade ))

    def RunBoxplotforCommonResultsFromList(self,position,difference,filelist,isEntireSignal):
        #load filenames
        # loadpathCommonfiles = self.objConfig.DiskPath + 'Result\\' + fileholdingfilenames + "_"+position  +".txt"
        # read data from files
        # HrFiledata = open(loadpathCommonfiles, "r")
        # FileNames = HrFiledata.read().split("\n")
        # HrFiledata.close()

        for participant_number in self.objConfig.ParticipantNumbers:
            self.objConfig.setSavePath(participant_number,position)

            savepathBoxplots = self.objConfig.SavePath + "\\BoxPlots\\"
            savepathBlandAltmanplots = self.objConfig.SavePath + "\\BlandAltmanPlots\\"

            if not os.path.exists(savepathBoxplots):
                os.makedirs(savepathBoxplots)

            if not os.path.exists(savepathBlandAltmanplots):
                os.makedirs(savepathBlandAltmanplots)

            algoCount_FastICA = 0
            algoCount_None = 0
            algoCount_PCA = 0
            algoCount_PCAICA = 0
            algoCount_Jade = 0

            data_fastica = []
            data_icapca = []
            data_pca = []
            data_jade = []
            data_none = []

            for file_nameList in filelist:
                file_name = file_nameList[0]
                #file location
                loadpath = self.objConfig.SavePath+  file_name + '\\' + self.hrfileName +'_'+ file_name+'.txt'
                #get data
                data = self.Getdata(loadpath, difference, isEntireSignal)

                if(file_name.__contains__('FastICA')):
                    data_fastica.append(data)
                elif (file_name.__contains__('ICAPCA')):
                    data_icapca.append(data)
                elif (file_name.__contains__('PCA')):
                    data_pca.append(data)
                elif (file_name.__contains__('Jade')):
                    data_jade.append(data)
                elif (file_name.__contains__('None')):
                    data_none.append(data)

            filenamegr = "HR"
            data_gr =[]
            data_gr =self.GetgroundTruth(self.objConfig.DiskPath + 'GroundTruthData\\' + participant_number+'\\' +position+'\\' ,filenamegr)

            avgFastIca= np.mean(data_fastica)
            avgNone= np.mean(data_none)
            avgPca= np.mean(data_pca)
            avgIcapca= np.mean(data_icapca)
            avgjade= np.mean(data_jade)
            minDiff = sys.float_info.max
            algotype = ""
            if(avgFastIca<minDiff and avgFastIca>=0):
                minDiff=avgFastIca
                algotype="FastICA"
            if (avgNone < minDiff and avgNone>=0):
                minDiff = avgNone
                algotype="None"
            if (avgPca < minDiff and avgPca>=0):
                minDiff = avgPca
                algotype="PCA"
            if (avgIcapca < minDiff and avgIcapca>=0):
                minDiff = avgIcapca
                algotype="PCAICA"
            if (avgjade < minDiff and avgjade>=0):
                minDiff = avgjade
                algotype="Jade"

            if(not (minDiff == sys.float_info.max)):
                data = [data_none, data_fastica, data_icapca,data_pca,data_jade,data_gr]

                boxTitle= "Best_Mean: " + str(minDiff) + " using " + str(algotype)+ "\n" \
                          + " (ICA: " + str(avgFastIca) + " , PCA: " + str(avgPca) + " , PCAICA: " + str(avgIcapca) + " , None: " + str(avgNone)+ " , Jade: " + str(avgjade) + ")"\
                          + "\n" + file_name

                self.Genrateboxplot(data,savepathBoxplots +algotype+"\\",boxTitle,file_name)


                if(algotype=="FastICA"):
                    self.GenerateBlandAltman_plot(savepathBlandAltmanplots+"FastICA\\", "_FastICA", data_fastica, data_gr,boxTitle,file_name)
                    algoCount_FastICA =algoCount_FastICA+1

                if (algotype=="None"):
                    self.GenerateBlandAltman_plot(savepathBlandAltmanplots+"None\\", "_None", data_none, data_gr, boxTitle,file_name)
                    algoCount_None =algoCount_None+1
                if (algotype=="PCA"):
                    self.GenerateBlandAltman_plot(savepathBlandAltmanplots+"PCA\\", "_PCA", data_pca, data_gr, boxTitle,file_name)
                    algoCount_PCA =algoCount_PCA+1
                if (algotype=="PCAICA"):
                    self.GenerateBlandAltman_plot(savepathBlandAltmanplots+"PCAICA\\", "_PCAICA", data_icapca, data_gr, boxTitle,file_name)
                    algoCount_PCAICA =algoCount_PCAICA+1

                if(algotype=="Jade"):
                    self.GenerateBlandAltman_plot(savepathBlandAltmanplots+"Jade\\", "_Jade", data_jade, data_gr, boxTitle,file_name)
                    algoCount_Jade =algoCount_Jade+1

        print(participant_number + " , FastICA count: " +str(algoCount_FastICA )+
              " , PCA count: " +str(algoCount_PCA )+
              " , PCAICA count: " +str(algoCount_PCAICA )+
              " , None count: " +str(algoCount_None )+
              " , Jade count: " +str(algoCount_Jade ))


    """
       Gemerate cases:
       """

    def GenerateCasesNewMethod(self):
        for preprocesstype in self.objConfig.preprocesses:
            for fftype in self.objConfig.fftTypeList:
                for resulttype in self.objConfig.resulttypeList:
                    for filtertype in self.objConfig.filtertypeList:
                        for isSmooth in self.objConfig.Smoothen:
                                #AlgoList = ["FastICA", "PCA", "ICAPCA", "None","Jade"]
                                fileNameFastICA = "ResultType_RS-" + str(resulttype) + "_Filtered_FL-"+ str(filtertype)+ "FFTtype-" + str(fftype)  + "_algotype-" + str('FastICA') +\
                                           '_PreProcessType-'+str(preprocesstype)+ "_Smoothed-" + str(isSmooth)

                                fileNamePCA = "ResultType_RS-" + str(resulttype) + "_Filtered_FL-"+ str(filtertype)+ "FFTtype-" + str(fftype)  + "_algotype-" + str('PCA') +\
                                           '_PreProcessType-'+str(preprocesstype)+ "_Smoothed-" + str(isSmooth)

                                fileNameICAPCA = "ResultType_RS-" + str(resulttype) + "_Filtered_FL-"+ str(filtertype)+ "FFTtype-" + str(fftype)  + "_algotype-" + str('ICAPCA') +\
                                           '_PreProcessType-'+str(preprocesstype)+ "_Smoothed-" + str(isSmooth)

                                fileNameNone = "ResultType_RS-" + str(resulttype) + "_Filtered_FL-"+ str(filtertype)+ "FFTtype-" + str(fftype)  + "_algotype-" + str('None') +\
                                           '_PreProcessType-'+str(preprocesstype)+ "_Smoothed-" + str(isSmooth)

                                fileNameJade = "ResultType_RS-" + str(resulttype) + "_Filtered_FL-"+ str(filtertype)+ "FFTtype-" + str(fftype)  + "_algotype-" + str('Jade') +\
                                           '_PreProcessType-'+str(preprocesstype)+ "_Smoothed-" + str(isSmooth)

                                self.boxPlotCasePariticipantWise('Resting1', fileNameFastICA,fileNamePCA,fileNameICAPCA,fileNameNone,fileNameJade)

                            #Genereate


    def readcase(self,fileName,participant_number,position):
        SavePath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\FinalComputedResult\\'
        filepath = SavePath +  fileName + '.txt' #HeartRate_FastICA_FFT-M1_FL-6_RS-1_PR-1_SM-False
        objFile = FileIO()
        pathExsists = objFile.FileExits(filepath)
        data =None
        # already generated
        if (pathExsists):
            Filedata = open(filepath, "r")
            data = Filedata.read().split("\n")[0]
            Filedata.close()
        del objFile
        return data

    def splitDataRow(self,DataRow,participant_number,position): #"None", loadpath,methodtype
        dataRow = DataRow.split(",\t")
        # 0 index for GroundTruth, 1 index for generatedresult by arpos, 2 index for diference
        generatedResult = int(dataRow[1]) #change index here
        avgGroundTruth = int(dataRow[0])
        # if(avgGroundTruthStored != round(avgGroundTruth)):
        #     differenceValue=int(dataRow[2])
        # else:
        differenceValue = avgGroundTruth - generatedResult
        differenceValue = np.abs( differenceValue/avgGroundTruth)*100
        return differenceValue, avgGroundTruth

    def boxPlotCasePariticipantWise(self,position, fileNameFastICA,fileNamePCA,fileNameICAPCA,fileNameNone,fileNameJade):

        ResultAlogrithm={}
        ParticipantsCaseData = {}
        self.objConfig.ParticipantNumbers =["PIS-8073","PIS-2047","PIS-4014","PIS-1949"]
        self.objConfig.Participantnumbers_SkinGroupTypes["PIS-8073"]='Europe_WhiteSkin_Group'
        self.objConfig.Participantnumbers_SkinGroupTypes["PIS-2047"]='Europe_WhiteSkin_Group'
        self.objConfig.Participantnumbers_SkinGroupTypes["PIS-4014"]='Europe_WhiteSkin_Group'
        self.objConfig.Participantnumbers_SkinGroupTypes["PIS-1949"]='Europe_WhiteSkin_Group'
        for participant_number in self.objConfig.ParticipantNumbers:
            self.objConfig.setSavePath(participant_number,position)

            ##read data for all pariticpants

            # if (caseDetail != None):
            caseDetail = self.readcase(fileNameFastICA,participant_number,position)
            FastICAerrorrate,avgGroundTruth = self.splitDataRow(caseDetail, participant_number, position)  # ROW,COlum
            ResultAlogrithm['FastICA'] = FastICAerrorrate

            caseDetail = self.readcase(fileNamePCA,participant_number,position)
            PCAerrorrate,avgGroundTruth = self.splitDataRow(caseDetail, participant_number, position)  # ROW,COlum
            ResultAlogrithm['PCA'] = PCAerrorrate

            caseDetail = self.readcase(fileNameICAPCA,participant_number,position)
            ICAPCAerrorrate,avgGroundTruth = self.splitDataRow(caseDetail, participant_number, position)  # ROW,COlum
            ResultAlogrithm['ICAPCA'] =  ICAPCAerrorrate

            caseDetail = self.readcase(fileNameNone,participant_number,position)
            Noneerrorrate,avgGroundTruth = self.splitDataRow(caseDetail, participant_number, position)  # ROW,COlum
            ResultAlogrithm['None'] = Noneerrorrate

            caseDetail = self.readcase(fileNameJade,participant_number,position)
            Jadeerrorrate,avgGroundTruth = self.splitDataRow(caseDetail, participant_number, position)  # ROW,COlum
            ResultAlogrithm['Jade'] = Jadeerrorrate
            avgGroundTruth=0
            ResultAlogrithm['GR'] = avgGroundTruth
            ParticipantsCaseData[participant_number] = ResultAlogrithm

        data = []
        fastica = []
        icapca = []
        pca = []
        none = []
        jade = []
        gr = []
        for k, v in ParticipantsCaseData.items():
            fastica.append(v.get('FastICA'))
            icapca.append(v.get('PCA'))
            pca.append(v.get('ICAPCA'))
            none.append(v.get('None'))
            jade.append(v.get('Jade'))
            gr.append(v.get('GR'))

        data = [none, fastica,icapca, pca,jade, gr]
        self.Genrateboxplot(data,'E:\\ARPOS_Server_Data\\Server_Study_Data\\Europe_WhiteSkin_Group\\Result\\',fileNameFastICA.replace('FASTICA',''),fileNameFastICA.replace('FASTICA',''))


# objboxplot = BoxPlot('Europe_WhiteSkin_Group')
# objboxplot.GenerateCasesNewMethod()
# GenerateGroundTruthFile() #Run to generate speerate groudntruth file
##Uncomment for all box plot
# RunBoxplotgeneration(True) #Set True to generate differnce box plot
#for common results only
#position,difference,fileholdingfilenames,containsAlgoName
# objboxplot.RunBoxplotforCommonResultsonly("Resting1",True,"BestDataFiles",True)#Set True to generate differnce box plot