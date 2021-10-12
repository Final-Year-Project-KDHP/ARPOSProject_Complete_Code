# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
import os
import sys
#
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

def Getdata(algotype,loadpath,methodtype,filename,filtertype,resulttype,differnece,processtype,isSmooth): #"None", loadpath,methodtype
    filepath = loadpath  + algotype + '\\' + filename + "-"+ methodtype+ "_Fl_"+ str(filtertype) + "_Rs_" + str(resulttype)  + "_Pr_" + str(processtype)+ "_Sm_" + str(isSmooth)+ ".txt" #HRdata-M1_1_1

    Filedata = open(filepath, "r")
    data =Filedata.read().split("\n")
    generatedresult = []
    for row in data:
        dataRow = row.split(",\t")

        indexvalue=2 # 0 index for windowCount, 1 index for GroundTruth, 2 index for generatedresult by arpos, 3 index for diference
        if(differnece== True):
            indexvalue = 3

        generatedValue = dataRow[indexvalue] #change index here
        generatedresult.append(generatedValue)

    generatedresult = list(map(int, generatedresult))
    return generatedresult


def ExtractGrdata(loadpath,filename): #"None", loadpath,methodtype
    filepath = loadpath  + filename + ".txt" #HRdata-M1_1_1

    Filedata = open(filepath, "r")
    data =Filedata.read().split("\n")
    generatedresult = []
    for row in data:
        dataRow = row.split(",\t")

        indexvalue=1 # 0 index for windowCount, 1 index for GroundTruth, 2 index for generatedresult by arpos, 3 index for diference
        # if(differnece== True):
        #     #indexvalue = 3
        #     generatedValue = 0
        # else:
        generatedValue = dataRow[indexvalue]
        generatedresult.append(generatedValue)

    generatedresult = list(map(int, generatedresult))
    return generatedresult

def GetgroundTruth(loadpath,filename): #"None", loadpath,methodtype
    filepath = loadpath  + filename + ".txt" #HRdata-M1_1_1

    Filedata = open(filepath, "r")
    data =Filedata.read().split("\n")
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
import matplotlib.pyplot as plt
import numpy as np

DiskPath = "E:\\StudyData\\Result\\" ## change path here for uncompressed dataset

ticks = [ 'None','FastICA', 'ICAPCA', 'PCA', 'Gr']

fftTypeList = ["M1","M2", "M3", "M4","M5","M6","M7"] #, "M4","M5", "M6" with butter filter try
fftTypeListNames = ["rfft + abs + rfftFreq","fftpack fft", "rfft + abs + power + rfftFreq", "fft","fft + sqrt + abs","fft + sqrt + abs + fftshift","rfft"] #, "M4","M5", "M6" with butter filter try
filtertypeList = [1,2,3,4,5,6,7] #, "M4","M5", "M6" with butter filter try ##,3,4,5,6,7,8,9,10 rest are all same result
filtertypeListNames = ["0 below and above","only below","cutoff[bound_high:-bound_high]","butter_bandpass_filter","None","above and below fft and frequency filter","7"]
preprocesses =[1,2,3,4,5,6,7]
preprocessesNames =["None" , "Detrend + Interpolate + Normalise" , "Detrend + Interpolate +  Normalise" , "Detrend + SmoothFilter +  Normalise" , "Normalise only","6","7" ] # ?? without detrend and normalise?
resulttypeList = [1,2,3,4,5,6,7,8,9] #, "M4","M5", "M6" with butter filter try
resulttypeListNames = ["ManualIndexIteration for Peak" , "ManualIndexIteration for Peak  with channelArrays limited data" ,"Frequency for Peak" , "FrequencyBPM for Peak" , "FrequencyBPM for Peak with channelArrays limited data"
    , "Frequency for Peak with channelArrays limited data" , "find_heart_rate loop and limits" , "find_heart_rate loop and limits with channelArrays limit data" , "Limited Arrays with ARGmax"]
Smoothen = [False,True]
SmoothenNames = ["False","True"]

#Only 30 fps
ParticipantNumbers=  ["PIS-2212","PIS-4497","PIS-8308","PIS-8343"]#,"PIS-3186","PIS-1118"["PIS-2212","PIS-4497"]#,"PIS-8308","PIS-8343","PIS-3186","PIS-1118"],"PIS-4497"
positionLists =  ["AfterExcersize"]#"Resting1","Resting2",

# methodtype= "M1"
# resulttype=1
# filtertype =1

def GenerateBlandAltman_plot(savepath,filename,data,data_gr,methodtype,filtertype,resulttype,processtype,isSmooth, boxTitle,filename2):
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


def Genrateboxplot(data,savepath,methodtype,filtertype,resulttype,processtype,isSmooth,boxTitle,filename):
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
    ax.set_yticklabels(['None', 'ICA',
                        'PCAICA', 'PCA', 'GroundTruth'])

    # Adding title
    fftfilterIndex = [x for x in range(len(fftTypeList)) if fftTypeList[x] == methodtype][0]  # np.where(fftTypeList == methodtype)
    filterIndex =[x for x in range(len(filtertypeList)) if filtertypeList[x] == int(filtertype)][0]   #  np.where(filtertypeList == filtertype)
    resultIndex = [x for x in range(len(resulttypeList)) if resulttypeList[x] == int(resulttype)][0]   # np.where(resulttypeList == resulttype)
    processtypeIndex = [x for x in range(len(preprocesses)) if preprocesses[x] == int(processtype)][0]   # np.where(preprocesses == processtype)
    isSmoothIndex = [x for x in range(len(Smoothen)) if Smoothen[x] == isSmooth][0]   # np.where(Smoothen == isSmooth)

    titlename = "Boxplot for [method]: " + fftTypeListNames[fftfilterIndex] + ", [FilterType]: " +  str(filtertypeListNames[filterIndex])+',\n '+\
                "[ResultType]: " + str(resulttypeListNames[resultIndex])  + ", [PreProcess]: " + str(preprocessesNames[processtypeIndex])+', '+ \
                "[Smoothen]: " + str(Smoothen[isSmoothIndex])
    plt.title(boxTitle )
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

def SaveGroundTruth(savepath,filename, data):
    fHr = open(savepath + filename + ".txt", "w+")
    i = 0
    for item in data:
        if (i == len(data) - 1):
            fHr.write(str(item))
        else:
            fHr.write(str(item) + '\n')
        i = i + 1
    fHr.close()

def GenerateGroundTruthFile():
    for participant_number in ParticipantNumbers:

        for position in positionLists:
            loadpath = DiskPath + participant_number + '\\' + position + '\\'  + "None\\"
            savepath= DiskPath + participant_number + '\\' + position + '\\'
            Grdata = ExtractGrdata(loadpath, "HRdata-M1_Fl_1_Rs_1")
            filename = "groundTruth"
            # if(not difference):
            #     filename = "groundTruthDifference"
            SaveGroundTruth(savepath,filename,Grdata)


def RunBoxplotgeneration(difference):

    for participant_number in ParticipantNumbers:

        for position in positionLists:
            loadpath = DiskPath + participant_number + '\\' + position + '\\'

            for resulttype in resulttypeList:
                resultpath = "ResultMethod_"
                if (difference):
                    resultpath = "ResultDifferenceMethod_"
                savepath = loadpath + "BoxPlots\\" + resultpath + str(resulttype) + '\\'

                if not os.path.exists(savepath):
                    os.makedirs(savepath)

                for methodtype in fftTypeList:

                    for filtertype in filtertypeList:

                        for processtype in preprocesses:

                            for isSmooth in Smoothen:

                                data_d =Getdata("None", loadpath,methodtype,"HRdata",filtertype,resulttype,difference,processtype,isSmooth ) # filepath = loadpath  + algotype + '\\' + filename + "-"+ methodtype+ "_"+ filtertype + "_" + resulttype+ ".txt" #HRdata-M1_1_1
                                data_a =Getdata("FastICA", loadpath,methodtype,"HRdata",filtertype,resulttype,difference,processtype,isSmooth)
                                data_b =Getdata("ICAPCA", loadpath,methodtype,"HRdata",filtertype,resulttype,difference,processtype,isSmooth)
                                data_c =Getdata("PCA", loadpath,methodtype,"HRdata",filtertype,resulttype,difference,processtype,isSmooth)


                                filenamegr = "groundTruth"
                                data_gr =[]
                                if(difference):
                                    # filenamegr = "groundTruthDifference"
                                    for x in data_d:
                                        data_gr.append(0)
                                else:
                                    data_gr =GetgroundTruth(loadpath,filenamegr)

                                data = [data_d, data_a, data_b,data_c,data_gr]

                                Genrateboxplot(data,savepath,methodtype,filtertype,resulttype,processtype,isSmooth)



def RunBoxplotforCommonResultsonly(position,difference,fileholdingfilenames,containsAlgoName):

    #load filenames
    loadpathCommonfiles = DiskPath + fileholdingfilenames + "_"+position  +".txt"
    # read data from files
    HrFiledata = open(loadpathCommonfiles, "r")
    HrFileNames = HrFiledata.read().split("\n")
    HrFiledata.close()

    for participant_number in ParticipantNumbers:
        loadpath = DiskPath + participant_number + '\\' + position + '\\'
        savepathBoxplots = loadpath + "\\BoxPlots\\"
        savepathBlandAltmanplots = loadpath + "\\BlandAltmanPlots\\"

        if not os.path.exists(savepathBoxplots):
            os.makedirs(savepathBoxplots)

        if not os.path.exists(savepathBlandAltmanplots):
            os.makedirs(savepathBlandAltmanplots)

        algoCount_FastICA = 0
        algoCount_None = 0
        algoCount_PCA = 0
        algoCount_PCAICA = 0

        for filename in HrFileNames:
            # M1_Fl_1_Rs_1_Pr_2_Sm_True.txt
            fileDetails = filename.split("_")
            algoname = ""
            methodindex=0
            if (containsAlgoName):
                algoname =fileDetails[0]
                methodindex =1
                filename = fileDetails[methodindex] + "_Fl_"+fileDetails[methodindex+2]+"_Rs_"+fileDetails[methodindex+4]+"_Pr_"+fileDetails[methodindex+6]+"_Sm_"+fileDetails[methodindex+8]


            methodtype = fileDetails[methodindex]
            filtertype = fileDetails[methodindex+2]
            resulttype = fileDetails[methodindex+4]
            processtype = fileDetails[methodindex+6]
            isSmooth = bool(fileDetails[methodindex+8].replace(".txt",""))

            loadpathNone = loadpath+ "None\\" + filename
            loadpathICA = loadpath + "FastICA\\" + filename
            loadpathICAPCA = loadpath + "ICAPCA\\" + filename
            loadpathPCA = loadpath + "PCA\\" + filename

            data_none =Getdata("None", loadpath,methodtype,"HRdata",filtertype,resulttype,difference,processtype,isSmooth ) # filepath = loadpath  + algotype + '\\' + filename + "-"+ methodtype+ "_"+ filtertype + "_" + resulttype+ ".txt" #HRdata-M1_1_1
            data_fastica =Getdata("FastICA", loadpath,methodtype,"HRdata",filtertype,resulttype,difference,processtype,isSmooth)
            data_icapca =Getdata("ICAPCA", loadpath,methodtype,"HRdata",filtertype,resulttype,difference,processtype,isSmooth)
            data_pca =Getdata("PCA", loadpath,methodtype,"HRdata",filtertype,resulttype,difference,processtype,isSmooth)


            filenamegr = "groundTruth"
            data_gr =[]
            if(difference):
                for x in data_none:
                    data_gr.append(0)
            else:
                data_gr =GetgroundTruth(loadpath,filenamegr)

            avgFastIca= np.mean(data_fastica)
            avgNone= np.mean(data_none)
            avgPca= np.mean(data_pca)
            avgIcapca= np.mean(data_icapca)
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

            if(not (minDiff == sys.float_info.max)):
                data = [data_none, data_fastica, data_icapca,data_pca,data_gr]

                boxTitle= "Best_Mean: " + str(minDiff) + " using " + str(algotype)+ "\n" \
                          + " (ICA: " + str(avgFastIca) + " , PCA: " + str(avgPca) + " , PCAICA: " + str(avgIcapca) + " , None: " + str(avgNone) + ")"\
                          + "\n" + filename

                Genrateboxplot(data,savepathBoxplots +algotype+"\\",methodtype,filtertype,resulttype,processtype,isSmooth,boxTitle,filename)


                if(algotype=="FastICA"):
                    GenerateBlandAltman_plot(savepathBlandAltmanplots+"FastICA\\", "_FastICA", data_fastica, data_gr, methodtype,
                                             filtertype, resulttype, processtype, isSmooth, boxTitle,filename)
                    algoCount_FastICA =algoCount_FastICA+1

                if (algotype=="None"):
                    GenerateBlandAltman_plot(savepathBlandAltmanplots+"None\\", "_None", data_none, data_gr, methodtype,
                                             filtertype, resulttype, processtype, isSmooth, boxTitle,filename)
                    algoCount_None =algoCount_None+1
                if (algotype=="PCA"):
                    GenerateBlandAltman_plot(savepathBlandAltmanplots+"PCA\\", "_PCA", data_pca, data_gr, methodtype, filtertype,
                                             resulttype, processtype, isSmooth, boxTitle,filename)
                    algoCount_PCA =algoCount_PCA+1
                if (algotype=="PCAICA"):
                    GenerateBlandAltman_plot(savepathBlandAltmanplots+"PCAICA\\", "_PCAICA", data_icapca, data_gr, methodtype,
                                             filtertype, resulttype, processtype, isSmooth, boxTitle,filename)
                    algoCount_PCAICA =algoCount_PCAICA+1


        print(participant_number + " , FastICA count: " +str(algoCount_FastICA )+
              " , PCA count: " +str(algoCount_PCA )+
              " , PCAICA count: " +str(algoCount_PCAICA )+
              " , None count: " +str(algoCount_None ))


# GenerateGroundTruthFile() #Run to generate speerate groudntruth file
##Uncomment for all box plot
# RunBoxplotgeneration(True) #Set True to generate differnce box plot
#for common results only
RunBoxplotforCommonResultsonly("AfterExcersize",True,"BestDataFiles",True)#Set True to generate differnce box plot