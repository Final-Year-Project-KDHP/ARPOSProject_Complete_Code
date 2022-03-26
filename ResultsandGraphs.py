import os
import statistics

import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import scipy.stats
from SaveGraphs import Plots
import statsmodels.api as sm

class ResultsandGraphs:
    # Global Objects
    objConfig = None
    objFileIO = None

    # Constructor
    def __init__(self,objConfig,objFileIO):
        self.objConfig = objConfig
        self.objFileIO = objFileIO

    def AllPlotsforComputedResults(self):
        # self.GenerateGraphfromOneFileforALlSPO("Resting1", True, "FastICA", "TestSPO", "Darker",
        #                                        True)  # step3 for all in one
        # testing only
        # self.CreateRestultsfromParitcipantIdforTestingoNLY("Resting1", "_Only" + "FastICA", "PIS-396")
        # self.CreateRestultsfromParitcipantIdforTestingoNLY("Resting1", "_Only" + "FastICA", "PIS-396P2")

        algorithmDataframe = self.CreateRestultsForICAOnly("Resting1","FastICA", "White")
        sns.catplot(x = "AlgorithmType",       # x variable name
                    y = "RMSE",       # y variable name
                    hue = "PreProcess",  # group variable name
                    data = algorithmDataframe,     # dataframe to plot
                    kind = "bar")
        plt.show()
        a=0
        # algorithmDataframe = self.CreateRestultsForICAOnly("Resting1","FastICA", "Brown")
        # sns.catplot(x="AlgorithmType",  # x variable name
        #             y="RMSE",  # y variable name
        #             hue="PreProcess",  # group variable name
        #             data=algorithmDataframe,  # dataframe to plot
        #             kind="bar")
        # plt.show()
        # a=0
        ###Plot values


        # for algorithm in self.objConfig.AlgoList:
        #     for position in self.objConfig.hearratestatus:
                #start steps...
                #Get best cases by calculating and comparing means
                # self.CreateRestultsfromBestFileCases(position,"_Only" + algorithm)#step1
                #Generate File containing same algorithm data from participnats
                # step2
                # fileName = "BestMeansbyAlgoTypeDataW" +str(self.objConfig.windowSize) + "_"+ position + "_Only" + algorithm#step2
                # self.GenerateDataFileWithAllDatafrombestmeans(position,algorithm,"White",fileName,True)#step2
                # self.GenerateDataFileWithAllDatafrombestmeans(position,algorithm,"Darker",fileName,True)#step2
                # # #Generate Graphs and r values
                # self.GenerateGraphfromOneFileforALl( position, True, algorithm,"White_BestMeansbyAlgoTypeDataW15_" + position+
                #                                      "_Only"+ algorithm  +"_SkinoverAllcommon_" + position,"White",True)#step3
                # self.GenerateGraphfromOneFileforALl( position, True, algorithm,"Darker_BestMeansbyAlgoTypeDataW15_" +position +
                #                                      "_Only"+ algorithm  + "_SkinoverAllcommon_" + position,"Darker",True)#step3
                # self.GenerateGraphfromOneFileforALl( position, True, algorithm,"_BestMeansbyAlgoTypeDataW15_" +position +
                #                                      "_Only"+ algorithm  + "_SkinoverAllcommon_" + position,"All",True)#step3 for all in one
                # self.GenerateGraphfromOneFileforALlSPO( position, True, algorithm,"_BestMeansbyAlgoTypeDataW15_" +position +
                #                                      "_Only"+ algorithm  + "_SkinoverAllcommon_" + position,"All",True)#step3 for all in one
                # self.GenerateGraphfromOneFileforALlSPO( position, True, algorithm,"Darker_BestMeansbyAlgoTypeDataW15_" +position +
                #                                      "_Only"+ algorithm  + "_SkinoverAllcommon_" + position,"Darker",True)#step3 for all in one
                # self.GenerateGraphfromOneFileforALlSPO( position, True, algorithm,"White_BestMeansbyAlgoTypeDataW15_" +position +
                #                                      "_Only"+ algorithm  + "_SkinoverAllcommon_" + position,"White",True)#step3 for all in one

                # Completed steps..

                # self.BlandAltmanPlotMain(position, "White",algorithm)  # step4
                # self.BlandAltmanPlotMain(position, "Darker",algorithm)  # step4



        # for position in self.objConfig.hearratestatus:
            # self.genearteBoxPlots(position,"White")####Final step1
            # self.genearteBoxPlots(position,"Darker")####Final step1
            # self.genearteBoxPlots(position,"All") ####Final step1
            # self.GenratebarPlotStats(position,"All") ####step2
            # self.GenratebarPlotStats(position,"White")####step2
            # self.GenratebarPlotStats(position,"Darker")####step2
        # #     # self.RvalPlotMainIndividual(position)#step5


        # BOX PLOT
        # self.RMSEPlotMain()#step1
        # self.DiffPlotMainSPO("All", "Resting1",'Original Obtianed Average differenceSPO',"SPO")#step2
        # self.DiffPlotMain("All", "Resting1",'HRDifference from averaged',"All")#step2
        # self.DiffPlotMain("White", "Resting2",'HRDifference from averaged')#step2
        # self.DiffPlotMain("White", "AfterExcersize",'HRDifference from averaged')#step2
        # self.DiffPlotMain("Darker", "Resting1",'HRDifference from averaged')#step2
        # self.DiffPlotMain("Darker", "Resting2",'HRDifference from averaged')#step2
        # self.DiffPlotMain("Darker", "AfterExcersize",'HRDifference from averaged')#step2
        # self.RvalPlotMain()#step3


                # self.ComputeStatResultsForPositionand("Resting1") #for each paritciapnt
                # self.ComputeStatResultsForParticiapnts("Resting1",True) #for all paritciapnt and send generate chart true or flase
                # self.getMaxValuebyType("Resting1")
                #"PIS-1949", "PIS-2047",
                # ParticipantNumbers=["PIS-2212", "PIS-3186", "PIS-3252", "PIS-3252P2",
                #                     "PIS-4014", "PIS-4497", "PIS-4709", "PIS-5868", "PIS-5868P2", "PIS-6327", "PIS-6729",
                #                     "PIS-6888", "PIS-7381", "PIS-7728", "PIS-8073", "PIS-8308", "PIS-8308P2",
                #                     "PIS-8343", "PIS-9219", "PIS-3807"]

                # self.SaveParticipantResultstoOneFile("Resting1",ParticipantNumbers,
                #                                      "ComputerHRandSPO_1_FFT_M4_Algorithm_FastICACombined_PreProcess_6",
                #                                      "WhitePigmentation_FastICACombinedP6")
                # self.SaveParticipantResultstoOneFile("Resting1", ParticipantNumbers,
                #                                      "ComputerHRandSPO_1_FFT_M4_Algorithm_FastICACombined_PreProcess_2",
                #                                      "WhitePigmentation_FastICACombinedP2")
                # self.SaveParticipantResultstoOneFile("Resting1", ParticipantNumbers,
                #                                      "ComputerHRandSPO_1_FFT_M4_Algorithm_FastICA_PreProcess_2",
                #                                      "WhitePigmentation_FastICAP2")
                # self.SaveParticipantResultstoOneFile("Resting1", ParticipantNumbers,
                #                                      "ComputerHRandSPO_1_FFT_M4_Algorithm_FastICA_PreProcess_6",
                #                                      "WhitePigmentation_FastICAP6")
                # self.SaveParticipantResultstoOneFile("Resting1", ParticipantNumbers,
                #                                      "ComputerHRandSPO_1_FFT_M4_Algorithm_FastICA3TimesForEachComponent_PreProcess_6",
                #                                      "WhitePigmentation_FastICA3TimesForEachComponentP6")
                # self.SaveParticipantResultstoOneFile("Resting1", ParticipantNumbers,
                #                                      "ComputerHRandSPO_1_FFT_M4_Algorithm_FastICA3TimesForEachComponent_PreProcess_2",
                #                                      "WhitePigmentation_FastICA3TimesForEachComponentP2")
                # self.SaveParticipantResultstoOneFile("Resting1", ParticipantNumbers,
                #                                      "ComputerHRandSPO_1_FFT_M4_Algorithm_FastICA3TimesCombined_PreProcess_2",
                #                                      "WhitePigmentation_FastICA3TimesCombinedP2")
                # self.SaveParticipantResultstoOneFile("Resting1", ParticipantNumbers,
                #                                      "ComputerHRandSPO_1_FFT_M4_Algorithm_FastICA3TimesCombined_PreProcess_6",
                #                                      "WhitePigmentation_FastICA3TimesCombinedP6")
    def GenratebarPlotStats(self,position,skintype):
        LoadfullPath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(self.objConfig.windowSize) + "\\" + position+ "\\" + "Filtered\\"
        algorithmDataframe = pd.read_csv(LoadfullPath+ position+"_statData_"+ skintype + "_AveragedGRWindowDiff_HR" + ".csv")  # read file
        algorithmDataframe = algorithmDataframe.drop(algorithmDataframe[algorithmDataframe.groupType == 'Slope'].index)
        algorithmDataframe = algorithmDataframe.dropna()
        algorithmDataframe = algorithmDataframe.drop(algorithmDataframe[algorithmDataframe.groupType == 'Intercept'].index)
        algorithmDataframe = algorithmDataframe.dropna()
        algorithmDataframe2 = algorithmDataframe[algorithmDataframe['groupType'] == 'Std']
        algorithmDataframe = algorithmDataframe.drop(algorithmDataframe[algorithmDataframe.groupType == 'Std'].index)
        algorithmDataframe = algorithmDataframe.dropna()
        # algorithmDataframe = algorithmDataframe.drop(algorithmDataframe[algorithmDataframe.groupType == 'p'].index)
        # algorithmDataframe = algorithmDataframe.dropna()
        # algorithmDataframe = algorithmDataframe.drop(algorithmDataframe[algorithmDataframe.groupType == 'meanAbs'].index)
        # algorithmDataframe = algorithmDataframe.dropna()

        # rearrange dataframe and plot
        fig = plt.figure(figsize=(20, 20))
        # plt.rcParams.update({'font.size': 20}) # must set in top
        # plt.rc('font', size=18)
        # plt.rcParams['legend.title_fontsize'] = 'small'
        # algorithmDataframe.plot(figsize=(20,15))
        algorithmDataframe.pivot(index="Algorithm", columns="groupType", values="value").plot.bar(edgecolor="white")
        plt.xticks(rotation=45)
        plt.xlabel("Algorithms", size=15)
        plt.ylabel("Statistical Values", size=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.tight_layout()
        # plt.show()
        plt.savefig(LoadfullPath +  position + "_statbarplot_" + skintype + "_AveragedGRWindowDiff_HR.png")

        plt.clf()
        fig = plt.figure(figsize=(20, 20))
        # plt.rcParams.update({'font.size': 20}) # must set in top
        # plt.rc('font', size=18)
        # algorithmDataframe.plot(figsize=(20,15))
        algorithmDataframe2.pivot(index="Algorithm", columns="groupType", values="value").plot.bar(edgecolor="white")
        plt.xticks(rotation=45)
        plt.xlabel("Algorithms", size=15)
        plt.ylabel("Statistical Values", size=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.tight_layout()
        plt.savefig(LoadfullPath + position + "_stdplot_" + skintype + "_AveragedGRWindowDiff_HR.png")
        # plt.show()

    def getMaxValuebyType(self, position):
        previousProcessingStep = "StatisticalResults"
        print("get Max value --> " + str(self.objConfig.windowSize))

        for participant_number in self.objConfig.ParticipantNumbers:
            currentSavePath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(
                self.objConfig.windowSize) + '\\' + participant_number + "\\" + position + "\\"

            ActualAveraged_vs_ARPOSComputed_StatValuesDictionary = {}
            ActualAveraged_vs_ARPOSComputed_StatValues = self.objFileIO.ReaddatafromFile(currentSavePath,"ActualAveraged_vs_ARPOSComputed_StatValues")
            for item in ActualAveraged_vs_ARPOSComputed_StatValues:
                splitItem = item.split(",")
                rvalue = splitItem[len(splitItem)-2]
                rvalue = rvalue.replace("r","")
                processName = splitItem[1]
                ActualAveraged_vs_ARPOSComputed_StatValuesDictionary[processName] = float(rvalue)

            MaxKey = max(ActualAveraged_vs_ARPOSComputed_StatValuesDictionary, key=ActualAveraged_vs_ARPOSComputed_StatValuesDictionary.get)
            MaxValue = ActualAveraged_vs_ARPOSComputed_StatValuesDictionary.get(MaxKey)

            print(participant_number + ", " + MaxKey + ", " + str(MaxValue))


    def SaveParticipantResultstoOneFile(self,position,ParticipantNumbers, processedFile, fileName):#ComputerHRandSPO_1_FFT_M4_Algorithm_FastICACombined_PreProcess_6
        processingStep = "SaveResultstoDiskDataFiles_W" + str(self.objConfig.windowSize)
        print(processingStep + " --> " + str(self.objConfig.windowSize))

        algorithmDataframe_AllData = pd.DataFrame()

        for participant_number in ParticipantNumbers:
            LoadPath= self.objConfig.DiskPath + processingStep +'\\' + participant_number + "\\" + position + "\\"
            algorithmDataframe = pd.read_csv(LoadPath + processedFile + ".csv")  # read file
            algorithmDataframe = algorithmDataframe.sort_values(by=['WindowCount'], ascending=[True])# sort before other processing
            count_row = algorithmDataframe.shape[0]  # Gives number of rows

            piList = []
            for x in range(0, count_row):
                piList.append(participant_number)

            algorithmDataframe["ParticipantId"] = piList

            if(algorithmDataframe_AllData.empty):
                algorithmDataframe_AllData = algorithmDataframe
            else:
                algorithmDataframe_AllData= algorithmDataframe_AllData.append(algorithmDataframe, ignore_index=True)

        algorithmDataframe_AllData.to_csv(self.objConfig.DiskPath + processingStep +'\\' + fileName + ".csv")  # Write to file

        print("completed")

    def genValuesMain(self,algorithmDataframe,AlgoName):
        print(AlgoName)
        r_avgedWindow, p_avgedWindow, meanABS_avgedWindow,\
        meanWithoutABS_avgedWindow, slope_avgedWindow, intercept_avgedWindow, \
        std_err_avgedWindow, rsqrd_avgedWindow,rms_avgedWindow =  self.GenerateValuesOnlyStats('GroundTruth HeartRate Averaged',
                                                                               'Computed HeartRate','HRDifference from averaged',
                                                                               algorithmDataframe, "HRDifference from averaged")
        r_orginalavgedWindow, p_orginalavgedWindow, meanABS_orginalavgedWindow, \
        meanWithoutABS_orginalavgedWindow, slope_orginalavgedWindow, intercept_orginalavgedWindow, \
        std_err_orginalavgedWindow, rsqrd_orginalavgedWindow,rms_orginalavgedWindow = self.GenerateValuesOnlyStats('GroundTruth HeartRate Averaged','bestBpm Without ReliabilityCheck','OriginalObtianedAveragedifferenceHR',
                                     algorithmDataframe, "Original Obtianed (bestbpm) Average differenceHR")
        r_lastsecWindow, p_lastsecWindow, meanABS_lastsecWindow, \
        meanWithoutABS_lastsecWindow, slope_lastsecWindow, intercept_lastsecWindow, \
        std_err_lastsecWindow, rsqrd_lastsecWindow,rms_lastsecWindow = self.GenerateValuesOnlyStats(' Hr from windows last second','Computed HeartRate',' LastSecondWindow differenceHR',algorithmDataframe, "HRDifference from LastSecondWindow")
        r_originallastsecWindow, p_originallastsecWindow, meanABS_originallastsecWindow, \
        meanWithoutABS_originallastsecWindow, slope_originallastsecWindow, intercept_originallastsecWindow, \
        std_err_originallastsecWindow, rsqrd_originallastsecWindow,rms_originallastsecWindow =    self.GenerateValuesOnlyStats(' Hr from windows last second','bestBpm Without ReliabilityCheck',' OriginalObtianed LastSecondWindow differenceHR',
                                     algorithmDataframe, "Original Obtianed (bestbpm) LastSecondWindow differenceHR")

        return r_avgedWindow, p_avgedWindow, meanABS_avgedWindow,\
        meanWithoutABS_avgedWindow, slope_avgedWindow, intercept_avgedWindow, \
        std_err_avgedWindow, rsqrd_avgedWindow,rms_avgedWindow, r_orginalavgedWindow, p_orginalavgedWindow, meanABS_orginalavgedWindow, \
        meanWithoutABS_orginalavgedWindow, slope_orginalavgedWindow, intercept_orginalavgedWindow, \
        std_err_orginalavgedWindow, rsqrd_orginalavgedWindow,rms_orginalavgedWindow,r_lastsecWindow, p_lastsecWindow, meanABS_lastsecWindow, \
        meanWithoutABS_lastsecWindow, slope_lastsecWindow, intercept_lastsecWindow, \
        std_err_lastsecWindow, rsqrd_lastsecWindow,rms_lastsecWindow,r_originallastsecWindow, p_originallastsecWindow, meanABS_originallastsecWindow, \
        meanWithoutABS_originallastsecWindow, slope_originallastsecWindow, intercept_originallastsecWindow, \
        std_err_originallastsecWindow, rsqrd_originallastsecWindow,rms_originallastsecWindow

    def getDataforAllskinTypes(self,skintype,algorithm,position,LoadfullPath):

        # lOAD Data
        if(skintype == "All"):
            fileName =  "White_BestMeansbyAlgoTypeDataW" + str(
                self.objConfig.windowSize) + "_" + position + "_Only" + algorithm + "_SkinoverAllcommon_" + position
            algorithmDataframe1 = pd.read_csv(LoadfullPath + fileName + ".csv")  # read file

            fileName =  "Darker_BestMeansbyAlgoTypeDataW" + str(
                self.objConfig.windowSize) + "_" + position + "_Only" + algorithm + "_SkinoverAllcommon_" + position
            algorithmDataframe2 = pd.read_csv(LoadfullPath + fileName + ".csv")  # read file

            alldata = [algorithmDataframe1, algorithmDataframe2]
            algorithmDataframe = pd.concat(alldata)
        else:
            fileName = skintype + "_BestMeansbyAlgoTypeDataW" + str(
                self.objConfig.windowSize) + "_" + position + "_Only" + algorithm + "_SkinoverAllcommon_" + position
            algorithmDataframe = pd.read_csv(LoadfullPath + fileName + ".csv")  # read file

        algorithmDataframe = algorithmDataframe.dropna()

        return algorithmDataframe


    def genearteBoxPlots(self,position,skintype):
        objPlots = Plots()
        LoadfullPath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(self.objConfig.windowSize) + "\\" + position+ "\\" + "Filtered\\"

        algoLabels = []  # list of algos
        r_avgedWindow_AlgoWise = []  # list of values for that algo in same order
        p_avgedWindow_AlgoWise = []  # list of values for that algo in same order
        meanABS_avgedWindow_AlgoWise = []
        meanWithoutABS_avgedWindow_AlgoWise = []
        slope_avgedWindow_AlgoWise = []
        intercept_avgedWindow_AlgoWise = []
        std_err_avgedWindow_AlgoWise = []
        rsqrd_avgedWindow_AlgoWise = []
        rms_avgedWindow_AlgoWise = []

        # FastICA
        algorithm = "FastICA"
        algorithmDataframe = self.getDataforAllskinTypes(skintype,algorithm,position,LoadfullPath)
        # pd.read_csv(LoadfullPath + fileName + ".csv")  # read file
        HRDifferenceAvg_FastICA =  algorithmDataframe['HRDifference from averaged'].tolist()
        OriginalObtianedAveragedifferenceHR_FastICA =  algorithmDataframe['OriginalObtianedAveragedifferenceHR'].tolist()
        LastSecondWindowdifferenceHR_FastICA =  algorithmDataframe[' LastSecondWindow differenceHR'].tolist()
        OriginalObtianedLastSecondWindowDiff_FastICA =  algorithmDataframe[' OriginalObtianed LastSecondWindow differenceHR'].tolist()

        r_avgedWindow, p_avgedWindow, meanABS_avgedWindow, \
        meanWithoutABS_avgedWindow, slope_avgedWindow, intercept_avgedWindow, \
        std_err_avgedWindow, rsqrd_avgedWindow, rms_avgedWindow, r_orginalavgedWindow, p_orginalavgedWindow, meanABS_orginalavgedWindow, \
        meanWithoutABS_orginalavgedWindow, slope_orginalavgedWindow, intercept_orginalavgedWindow, \
        std_err_orginalavgedWindow, rsqrd_orginalavgedWindow, rms_orginalavgedWindow, r_lastsecWindow, p_lastsecWindow, meanABS_lastsecWindow, \
        meanWithoutABS_lastsecWindow, slope_lastsecWindow, intercept_lastsecWindow, \
        std_err_lastsecWindow, rsqrd_lastsecWindow, rms_lastsecWindow, r_originallastsecWindow, p_originallastsecWindow, meanABS_originallastsecWindow, \
        meanWithoutABS_originallastsecWindow, slope_originallastsecWindow, intercept_originallastsecWindow, \
        std_err_originallastsecWindow, rsqrd_originallastsecWindow, rms_originallastsecWindow = self.genValuesMain(algorithmDataframe,algorithm)

        algoLabels.append(algorithm)
        r_avgedWindow_AlgoWise.append(r_avgedWindow)
        p_avgedWindow_AlgoWise.append(p_avgedWindow)
        meanABS_avgedWindow_AlgoWise.append(meanABS_avgedWindow)
        meanWithoutABS_avgedWindow_AlgoWise.append(meanWithoutABS_avgedWindow)
        slope_avgedWindow_AlgoWise.append(slope_avgedWindow)
        intercept_avgedWindow_AlgoWise.append(intercept_avgedWindow)
        std_err_avgedWindow_AlgoWise.append(std_err_avgedWindow)
        rsqrd_avgedWindow_AlgoWise.append(rsqrd_avgedWindow)
        rms_avgedWindow_AlgoWise.append(rms_avgedWindow)

        GRrecord = np.zeros_like(HRDifferenceAvg_FastICA)

        # None
        algorithm = "None"
        fileName = skintype + "_BestMeansbyAlgoTypeDataW" + str(
            self.objConfig.windowSize) + "_" + position + "_Only" + algorithm + "_SkinoverAllcommon_" + position
        algorithmDataframe = self.getDataforAllskinTypes(skintype,algorithm,position,LoadfullPath)
        HRDifferenceAvg_None = algorithmDataframe['HRDifference from averaged'].tolist()
        OriginalObtianedAveragedifferenceHR_None = algorithmDataframe['OriginalObtianedAveragedifferenceHR'].tolist()
        LastSecondWindowdifferenceHR_None = algorithmDataframe[' LastSecondWindow differenceHR'].tolist()
        OriginalObtianedLastSecondWindowDiff_None = algorithmDataframe[
            ' OriginalObtianed LastSecondWindow differenceHR'].tolist()


        r_avgedWindow, p_avgedWindow, meanABS_avgedWindow, \
        meanWithoutABS_avgedWindow, slope_avgedWindow, intercept_avgedWindow, \
        std_err_avgedWindow, rsqrd_avgedWindow, rms_avgedWindow, r_orginalavgedWindow, p_orginalavgedWindow, meanABS_orginalavgedWindow, \
        meanWithoutABS_orginalavgedWindow, slope_orginalavgedWindow, intercept_orginalavgedWindow, \
        std_err_orginalavgedWindow, rsqrd_orginalavgedWindow, rms_orginalavgedWindow, r_lastsecWindow, p_lastsecWindow, meanABS_lastsecWindow, \
        meanWithoutABS_lastsecWindow, slope_lastsecWindow, intercept_lastsecWindow, \
        std_err_lastsecWindow, rsqrd_lastsecWindow, rms_lastsecWindow, r_originallastsecWindow, p_originallastsecWindow, meanABS_originallastsecWindow, \
        meanWithoutABS_originallastsecWindow, slope_originallastsecWindow, intercept_originallastsecWindow, \
        std_err_originallastsecWindow, rsqrd_originallastsecWindow, rms_originallastsecWindow = self.genValuesMain(algorithmDataframe,algorithm)

        algoLabels.append(algorithm)
        r_avgedWindow_AlgoWise.append(r_avgedWindow)
        p_avgedWindow_AlgoWise.append(p_avgedWindow)
        meanABS_avgedWindow_AlgoWise.append(meanABS_avgedWindow)
        meanWithoutABS_avgedWindow_AlgoWise.append(meanWithoutABS_avgedWindow)
        slope_avgedWindow_AlgoWise.append(slope_avgedWindow)
        intercept_avgedWindow_AlgoWise.append(intercept_avgedWindow)
        std_err_avgedWindow_AlgoWise.append(std_err_avgedWindow)
        rsqrd_avgedWindow_AlgoWise.append(rsqrd_avgedWindow)
        rms_avgedWindow_AlgoWise.append(rms_avgedWindow)

        # PCA
        algorithm = "PCA"
        fileName = skintype + "_BestMeansbyAlgoTypeDataW" + str(
            self.objConfig.windowSize) + "_" + position + "_Only" + algorithm + "_SkinoverAllcommon_" + position
        algorithmDataframe = self.getDataforAllskinTypes(skintype,algorithm,position,LoadfullPath)
        HRDifferenceAvg_PCA =  algorithmDataframe['HRDifference from averaged'].tolist()
        OriginalObtianedAveragedifferenceHR_PCA =  algorithmDataframe['OriginalObtianedAveragedifferenceHR'].tolist()
        LastSecondWindowdifferenceHR_PCA =  algorithmDataframe[' LastSecondWindow differenceHR'].tolist()
        OriginalObtianedLastSecondWindowDiff_PCA =  algorithmDataframe[' OriginalObtianed LastSecondWindow differenceHR'].tolist()


        r_avgedWindow, p_avgedWindow, meanABS_avgedWindow, \
        meanWithoutABS_avgedWindow, slope_avgedWindow, intercept_avgedWindow, \
        std_err_avgedWindow, rsqrd_avgedWindow, rms_avgedWindow, r_orginalavgedWindow, p_orginalavgedWindow, meanABS_orginalavgedWindow, \
        meanWithoutABS_orginalavgedWindow, slope_orginalavgedWindow, intercept_orginalavgedWindow, \
        std_err_orginalavgedWindow, rsqrd_orginalavgedWindow, rms_orginalavgedWindow, r_lastsecWindow, p_lastsecWindow, meanABS_lastsecWindow, \
        meanWithoutABS_lastsecWindow, slope_lastsecWindow, intercept_lastsecWindow, \
        std_err_lastsecWindow, rsqrd_lastsecWindow, rms_lastsecWindow, r_originallastsecWindow, p_originallastsecWindow, meanABS_originallastsecWindow, \
        meanWithoutABS_originallastsecWindow, slope_originallastsecWindow, intercept_originallastsecWindow, \
        std_err_originallastsecWindow, rsqrd_originallastsecWindow, rms_originallastsecWindow = self.genValuesMain(algorithmDataframe,algorithm)

        algoLabels.append(algorithm)
        r_avgedWindow_AlgoWise.append(r_avgedWindow)
        p_avgedWindow_AlgoWise.append(p_avgedWindow)
        meanABS_avgedWindow_AlgoWise.append(meanABS_avgedWindow)
        meanWithoutABS_avgedWindow_AlgoWise.append(meanWithoutABS_avgedWindow)
        slope_avgedWindow_AlgoWise.append(slope_avgedWindow)
        intercept_avgedWindow_AlgoWise.append(intercept_avgedWindow)
        std_err_avgedWindow_AlgoWise.append(std_err_avgedWindow)
        rsqrd_avgedWindow_AlgoWise.append(rsqrd_avgedWindow)
        rms_avgedWindow_AlgoWise.append(rms_avgedWindow)

        # PCAICA #Spectralembedding
        algorithm = "PCAICA"
        fileName = skintype + "_BestMeansbyAlgoTypeDataW" + str(
            self.objConfig.windowSize) + "_" + position + "_Only" + algorithm + "_SkinoverAllcommon_" + position
        algorithmDataframe = self.getDataforAllskinTypes(skintype,algorithm,position,LoadfullPath)
        HRDifferenceAvg_PCAICA =  algorithmDataframe['HRDifference from averaged'].tolist()
        OriginalObtianedAveragedifferenceHR_PCAICA =  algorithmDataframe['OriginalObtianedAveragedifferenceHR'].tolist()
        LastSecondWindowdifferenceHR_PCAICA =  algorithmDataframe[' LastSecondWindow differenceHR'].tolist()
        OriginalObtianedLastSecondWindowDiff_PCAICA =  algorithmDataframe[' OriginalObtianed LastSecondWindow differenceHR'].tolist()


        r_avgedWindow, p_avgedWindow, meanABS_avgedWindow, \
        meanWithoutABS_avgedWindow, slope_avgedWindow, intercept_avgedWindow, \
        std_err_avgedWindow, rsqrd_avgedWindow, rms_avgedWindow, r_orginalavgedWindow, p_orginalavgedWindow, meanABS_orginalavgedWindow, \
        meanWithoutABS_orginalavgedWindow, slope_orginalavgedWindow, intercept_orginalavgedWindow, \
        std_err_orginalavgedWindow, rsqrd_orginalavgedWindow, rms_orginalavgedWindow, r_lastsecWindow, p_lastsecWindow, meanABS_lastsecWindow, \
        meanWithoutABS_lastsecWindow, slope_lastsecWindow, intercept_lastsecWindow, \
        std_err_lastsecWindow, rsqrd_lastsecWindow, rms_lastsecWindow, r_originallastsecWindow, p_originallastsecWindow, meanABS_originallastsecWindow, \
        meanWithoutABS_originallastsecWindow, slope_originallastsecWindow, intercept_originallastsecWindow, \
        std_err_originallastsecWindow, rsqrd_originallastsecWindow, rms_originallastsecWindow = self.genValuesMain(algorithmDataframe,algorithm)

        algoLabels.append(algorithm)
        r_avgedWindow_AlgoWise.append(r_avgedWindow)
        p_avgedWindow_AlgoWise.append(p_avgedWindow)
        meanABS_avgedWindow_AlgoWise.append(meanABS_avgedWindow)
        meanWithoutABS_avgedWindow_AlgoWise.append(meanWithoutABS_avgedWindow)
        slope_avgedWindow_AlgoWise.append(slope_avgedWindow)
        intercept_avgedWindow_AlgoWise.append(intercept_avgedWindow)
        std_err_avgedWindow_AlgoWise.append(std_err_avgedWindow)
        rsqrd_avgedWindow_AlgoWise.append(rsqrd_avgedWindow)
        rms_avgedWindow_AlgoWise.append(rms_avgedWindow)

        # Spectralembedding
        algorithm = "Spectralembedding"
        fileName = skintype + "_BestMeansbyAlgoTypeDataW" + str(
            self.objConfig.windowSize) + "_" + position + "_Only" + algorithm + "_SkinoverAllcommon_" + position
        algorithmDataframe = self.getDataforAllskinTypes(skintype,algorithm,position,LoadfullPath)
        HRDifferenceAvg_Spectralembedding = algorithmDataframe['HRDifference from averaged'].tolist()
        OriginalObtianedAveragedifferenceHR_Spectralembedding = algorithmDataframe['OriginalObtianedAveragedifferenceHR'].tolist()
        LastSecondWindowdifferenceHR_Spectralembedding = algorithmDataframe[' LastSecondWindow differenceHR'].tolist()
        OriginalObtianedLastSecondWindowDiff_Spectralembedding = algorithmDataframe[
            ' OriginalObtianed LastSecondWindow differenceHR'].tolist()


        r_avgedWindow, p_avgedWindow, meanABS_avgedWindow, \
        meanWithoutABS_avgedWindow, slope_avgedWindow, intercept_avgedWindow, \
        std_err_avgedWindow, rsqrd_avgedWindow, rms_avgedWindow, r_orginalavgedWindow, p_orginalavgedWindow, meanABS_orginalavgedWindow, \
        meanWithoutABS_orginalavgedWindow, slope_orginalavgedWindow, intercept_orginalavgedWindow, \
        std_err_orginalavgedWindow, rsqrd_orginalavgedWindow, rms_orginalavgedWindow, r_lastsecWindow, p_lastsecWindow, meanABS_lastsecWindow, \
        meanWithoutABS_lastsecWindow, slope_lastsecWindow, intercept_lastsecWindow, \
        std_err_lastsecWindow, rsqrd_lastsecWindow, rms_lastsecWindow, r_originallastsecWindow, p_originallastsecWindow, meanABS_originallastsecWindow, \
        meanWithoutABS_originallastsecWindow, slope_originallastsecWindow, intercept_originallastsecWindow, \
        std_err_originallastsecWindow, rsqrd_originallastsecWindow, rms_originallastsecWindow = self.genValuesMain(algorithmDataframe,algorithm)

        algoLabels.append(algorithm)
        r_avgedWindow_AlgoWise.append(r_avgedWindow)
        p_avgedWindow_AlgoWise.append(p_avgedWindow)
        meanABS_avgedWindow_AlgoWise.append(meanABS_avgedWindow)
        meanWithoutABS_avgedWindow_AlgoWise.append(meanWithoutABS_avgedWindow)
        slope_avgedWindow_AlgoWise.append(slope_avgedWindow)
        intercept_avgedWindow_AlgoWise.append(intercept_avgedWindow)
        std_err_avgedWindow_AlgoWise.append(std_err_avgedWindow)
        rsqrd_avgedWindow_AlgoWise.append(rsqrd_avgedWindow)
        rms_avgedWindow_AlgoWise.append(rms_avgedWindow)

        # Jade
        algorithm = "Jade"
        fileName = skintype + "_BestMeansbyAlgoTypeDataW" + str(
            self.objConfig.windowSize) + "_" + position + "_Only" + algorithm + "_SkinoverAllcommon_" + position
        algorithmDataframe = self.getDataforAllskinTypes(skintype,algorithm,position,LoadfullPath)
        HRDifferenceAvg_Jade =  algorithmDataframe['HRDifference from averaged'].tolist()
        OriginalObtianedAveragedifferenceHR_Jade =  algorithmDataframe['OriginalObtianedAveragedifferenceHR'].tolist()
        LastSecondWindowdifferenceHR_Jade =  algorithmDataframe[' LastSecondWindow differenceHR'].tolist()
        OriginalObtianedLastSecondWindowDiff_Jade =  algorithmDataframe[' OriginalObtianed LastSecondWindow differenceHR'].tolist()


        r_avgedWindow, p_avgedWindow, meanABS_avgedWindow, \
        meanWithoutABS_avgedWindow, slope_avgedWindow, intercept_avgedWindow, \
        std_err_avgedWindow, rsqrd_avgedWindow, rms_avgedWindow, r_orginalavgedWindow, p_orginalavgedWindow, meanABS_orginalavgedWindow, \
        meanWithoutABS_orginalavgedWindow, slope_orginalavgedWindow, intercept_orginalavgedWindow, \
        std_err_orginalavgedWindow, rsqrd_orginalavgedWindow, rms_orginalavgedWindow, r_lastsecWindow, p_lastsecWindow, meanABS_lastsecWindow, \
        meanWithoutABS_lastsecWindow, slope_lastsecWindow, intercept_lastsecWindow, \
        std_err_lastsecWindow, rsqrd_lastsecWindow, rms_lastsecWindow, r_originallastsecWindow, p_originallastsecWindow, meanABS_originallastsecWindow, \
        meanWithoutABS_originallastsecWindow, slope_originallastsecWindow, intercept_originallastsecWindow, \
        std_err_originallastsecWindow, rsqrd_originallastsecWindow, rms_originallastsecWindow = self.genValuesMain(algorithmDataframe,algorithm)

        algoLabels.append(algorithm)
        r_avgedWindow_AlgoWise.append(r_avgedWindow)
        p_avgedWindow_AlgoWise.append(p_avgedWindow)
        meanABS_avgedWindow_AlgoWise.append(meanABS_avgedWindow)
        meanWithoutABS_avgedWindow_AlgoWise.append(meanWithoutABS_avgedWindow)
        slope_avgedWindow_AlgoWise.append(slope_avgedWindow)
        intercept_avgedWindow_AlgoWise.append(intercept_avgedWindow)
        std_err_avgedWindow_AlgoWise.append(std_err_avgedWindow)
        rsqrd_avgedWindow_AlgoWise.append(rsqrd_avgedWindow)
        rms_avgedWindow_AlgoWise.append(rms_avgedWindow)

        data = [HRDifferenceAvg_FastICA, HRDifferenceAvg_None,HRDifferenceAvg_PCA, HRDifferenceAvg_PCAICA, HRDifferenceAvg_Spectralembedding, HRDifferenceAvg_Jade]
        objPlots.Genrateboxplot(data, LoadfullPath, position, 'HR', 'HR (BPM) ',skintype + "_AveragedGRWindowDiff")

        data = [OriginalObtianedAveragedifferenceHR_FastICA, OriginalObtianedAveragedifferenceHR_None,OriginalObtianedAveragedifferenceHR_PCA,
                OriginalObtianedAveragedifferenceHR_PCAICA, OriginalObtianedAveragedifferenceHR_Spectralembedding, OriginalObtianedAveragedifferenceHR_Jade]
        objPlots.Genrateboxplot(data, LoadfullPath, position, 'HR', 'HR (BPM) ',skintype + "_AveragedGRWindowDiffWithoutRealibilityCheck")

        data = [LastSecondWindowdifferenceHR_FastICA, LastSecondWindowdifferenceHR_None,LastSecondWindowdifferenceHR_PCA,
                LastSecondWindowdifferenceHR_PCAICA, LastSecondWindowdifferenceHR_Spectralembedding, LastSecondWindowdifferenceHR_Jade]
        objPlots.Genrateboxplot(data, LoadfullPath, position, 'HR', 'HR (BPM) ',skintype + "_LastSecWindowDiff")

        data = [OriginalObtianedLastSecondWindowDiff_FastICA, OriginalObtianedLastSecondWindowDiff_None,OriginalObtianedLastSecondWindowDiff_PCA,
                OriginalObtianedLastSecondWindowDiff_PCAICA, OriginalObtianedLastSecondWindowDiff_Spectralembedding, OriginalObtianedLastSecondWindowDiff_Jade]
        objPlots.Genrateboxplot(data, LoadfullPath, position, 'HR', 'HR (BPM) ',skintype + "_LastSecWindowDiffWithoutRealibilityCheck")


        # x = np.arange(len(algoLabels))  # the label locations
        # width = 0.25  # the width of the bars
        #
        # fig, ax = plt.subplots()
        # rects1 = ax.bar( r_avgedWindow_AlgoWise, width, label='r')
        # rects2 = ax.bar( p_avgedWindow_AlgoWise, width, label='p')
        # rects3 = ax.bar( meanABS_avgedWindow_AlgoWise, width, label='meanAbs')
        # rects4 = ax.bar( meanWithoutABS_avgedWindow_AlgoWise, width, label='mean')
        # rects5 = ax.bar( slope_avgedWindow_AlgoWise, width, label='slope')
        # rects6 = ax.bar( intercept_avgedWindow_AlgoWise, width, label='intercept')
        # rects7 = ax.bar( std_err_avgedWindow_AlgoWise, width, label='stderr')
        # rects8 = ax.bar(rsqrd_avgedWindow_AlgoWise, width, label='rsqrd')#x + width / 2,
        #
        # # Add some text for labels, title and custom x-axis tick labels, etc.
        # ax.set_ylabel('Values')
        # ax.set_title('Values by Algorithms')
        # ax.set_xticks(x, algoLabels)
        # ax.legend()
        #
        # ax.bar_label(rects1, padding=1)
        # ax.bar_label(rects2, padding=1)
        # ax.bar_label(rects3, padding=1)
        # ax.bar_label(rects4, padding=1)
        # ax.bar_label(rects5, padding=1)
        # ax.bar_label(rects6, padding=1)
        # ax.bar_label(rects7, padding=1)
        # ax.bar_label(rects8, padding=1)
        #
        # fig.tight_layout()
        #
        # plt.savefig(LoadfullPath + "barPlotstaVals.png")

        # # dataframe
        df = pd.DataFrame()

        #for r values
        count = 0
        for item in algoLabels:
            temp = pd.DataFrame(
                {
                    'Algorithm': item,
                    'groupType': "r",
                    'value': r_avgedWindow_AlgoWise[count]
                }, index=[0]
            )
            df = pd.concat([df, temp])
            count = count + 1

        # for p values
        count = 0
        for item in algoLabels:
            temp = pd.DataFrame(
                {
                    'Algorithm': item,
                    'groupType': "p",
                    'value': p_avgedWindow_AlgoWise[count]
                }, index=[0]
            )
            df = pd.concat([df, temp])
            count = count + 1

        # for meanABS values
        count = 0
        for item in algoLabels:
            temp = pd.DataFrame(
                {
                    'Algorithm': item,
                    'groupType': "MeanAbs",
                    'value': meanABS_avgedWindow_AlgoWise[count]
                }, index=[0]
            )
            df = pd.concat([df, temp])
            count = count + 1

        # for mean values
        count = 0
        for item in algoLabels:
            temp = pd.DataFrame(
                {
                    'Algorithm': item,
                    'groupType': "Mean",
                    'value': meanWithoutABS_avgedWindow_AlgoWise[count]
                }, index=[0]
            )
            df = pd.concat([df, temp])
            count = count + 1

        # for slope values
        count = 0
        for item in algoLabels:
            temp = pd.DataFrame(
                {
                    'Algorithm': item,
                    'groupType': "Slope",
                    'value': slope_avgedWindow_AlgoWise[count]
                }, index=[0]
            )
            df = pd.concat([df, temp])
            count = count + 1

        # for intercept values
        count = 0
        for item in algoLabels:
            temp = pd.DataFrame(
                {
                    'Algorithm': item,
                    'groupType': "Intercept",
                    'value': intercept_avgedWindow_AlgoWise[count]
                }, index=[0]
            )
            df = pd.concat([df, temp])
            count = count + 1

        # for std values
        count = 0
        for item in algoLabels:
            temp = pd.DataFrame(
                {
                    'Algorithm': item,
                    'groupType': "Std",
                    'value': std_err_avgedWindow_AlgoWise[count]
                }, index=[0]
            )
            df = pd.concat([df, temp])
            count = count + 1

        # for rsqrd values
        count = 0
        for item in algoLabels:
            temp = pd.DataFrame(
                {
                    'Algorithm': item,
                    'groupType': "Rsquared",
                    'value': rsqrd_avgedWindow_AlgoWise[count]
                }, index=[0]
            )
            df = pd.concat([df, temp])
            count = count + 1

        # for rsqrd values
        count = 0
        for item in algoLabels:
            temp = pd.DataFrame(
                {
                    'Algorithm': item,
                    'groupType': "RMSE",
                    'value': rms_avgedWindow_AlgoWise[count]
                }, index=[0]
            )
            df = pd.concat([df, temp])
            count = count + 1
        df.to_csv(LoadfullPath+  position + "_statData_" + skintype +"_AveragedGRWindowDiff_HR" + ".csv")  # Write to file



    def ComputeStatResultsForPositionand(self,position):
        objPlots = Plots()
        generateChart = False
        processingStep = "StatisticalResults"
        print(processingStep + " --> " + str(self.objConfig.windowSize))
        previousPath = self.objConfig.ProcessingSteps[len(self.objConfig.ProcessingSteps) - 1]

        dataTypeList = []

        prevdatatypepath = self.objConfig.getPathforFolderName(previousPath + "DataFiles", "PIS-1032",position)#any folder just to get all processed item names
        for fileName in os.listdir(prevdatatypepath):  # Load previous step files, for each file
            fileName =fileName.replace(".csv","")
            if ((not fileName.__contains__("ProcessedCompleted"))):
                if ((not fileName.__contains__("Graphs"))):
                    if(not dataTypeList.__contains__(fileName)):
                        dataTypeList.append(fileName)
        List1 = []
        Listwo = []
        Lis3 = []
        Lis4 = []
        corelationThreashold = 0.4

        # currentSavePath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(self.objConfig.windowSize)+'\\'

        for participant_number in self.objConfig.ParticipantNumbers:
            currentSavePath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(self.objConfig.windowSize)+'\\' + participant_number + "\\" + position + "\\"
            self.objFileIO.CreatePath(currentSavePath)

            for processedFile in dataTypeList:
                type =processedFile
                Actual_all_averaged = []
                Actual_all_LastSecond = []
                Observed_all = []
                Observed_Withoutcheck_all = []


                # participant_number = "PIS-256"
                previousfullPath = self.objConfig.getPathforFolderName(previousPath + "DataFiles", participant_number,
                                                                       position)
                fileName =processedFile
                splitFileName = fileName.split("_")
                AlgorithmName = splitFileName[len(splitFileName)-3]
                # lOAD Data
                algorithmDataframe = pd.read_csv(previousfullPath + fileName + ".csv")  # read file

                algorithmDataframe = algorithmDataframe.sort_values(by=['WindowCount'], ascending=[True])# sort before other processing

                PulseOximeterClipdataAveraged =  algorithmDataframe['GroundTruth HeartRate Averaged'].tolist()
                PulseOximeterClipdataLastSecond =  algorithmDataframe[' Hr from windows last second'].tolist()

                Observeddata = algorithmDataframe['Computed HeartRate'].tolist()
                ObserveddataWithoutCheck =  algorithmDataframe['bestBpm Without ReliabilityCheck'].tolist()

                # ObserveddataDifference =  algorithmDataframe['HRDifference'].tolist()
                # ObserveddataWithoutCheckDifference =  algorithmDataframe['bestBpmWithoutReliabilityCheckDifference'].tolist()

                # if(AlgorithmName.__contains__("FastICACombined")):
                #     data_tuples = list(zip(PulseOximeterClipdata, Observeddata))
                #     dfARPOS = pd.DataFrame(data_tuples, columns=['PulseOximeterClipdata', 'Observeddata'])
                #     correlation = dfARPOS.corr()
                #     print(type(correlation))

                Actual_all_averaged =Actual_all_averaged + PulseOximeterClipdataAveraged
                Actual_all_LastSecond =Actual_all_LastSecond + PulseOximeterClipdataLastSecond
                Observed_all=Observed_all+Observeddata
                Observed_Withoutcheck_all=Observed_Withoutcheck_all+ObserveddataWithoutCheck


                # ObserveddataWithoutCheck
                mean_squared_error_result, r, p = self.GenerateResultbyAlgorithmType(objPlots,
                                                                                     position,
                                                                                     Actual_all_averaged,
                                                                                     Observed_all,
                                                                                     AlgorithmName,
                                                                                     currentSavePath,
                                                                                     participant_number,
                                                                                     generateChart,
                                                                                     'HR_All_' + type + "_1")
                resultValues = "dataAveraged : mean_squared_error_result = " + str(
                    mean_squared_error_result) + ", r" + str(r) + ", p " + str(p)
                # self.objFileIO.WritedatatoFile(currentSavePath,
                #                                type + "_StatValues", resultValues)
                # print( "Actual Averaged vs ARPOS Computed, " + type + "," + resultValues)
                List1.append("Actual Averaged vs ARPOS Computed, " + type + "," + resultValues)

                # GenerateObservedvsActual
                mean_squared_error_result, r, p = self.GenerateResultbyAlgorithmType(objPlots,
                                                                                     position,
                                                                                     Actual_all_averaged,
                                                                                     Observed_Withoutcheck_all,
                                                                                     AlgorithmName,
                                                                                     currentSavePath,
                                                                                     participant_number,
                                                                                     generateChart,
                                                                                     'HR_' + type + "_WOcheck")

                resultValues = "dataAveraged Wihtout check: mean_squared_error_result = " + str(
                    mean_squared_error_result) + ", r" + str(r) + ", p " + str(p)
                # print(participant_number + ", " + position + ", " + type + "," + resultValues)
                # print( "Actual Averaged vs ARPOS Computed without check, " + type + "," + resultValues)
                Listwo.append( type + "," + resultValues)


                # ObserveddataWithoutCheck
                mean_squared_error_result, r, p = self.GenerateResultbyAlgorithmType(objPlots,
                                                                                     position,
                                                                                     Actual_all_LastSecond,
                                                                                     Observed_all,
                                                                                     AlgorithmName,
                                                                                     currentSavePath,
                                                                                     participant_number,
                                                                                     generateChart,
                                                                                     'HR_All_' + type + "_1")
                resultValues = "data last second : mean_squared_error_result = " + str(
                    mean_squared_error_result) + ", r" + str(r) + ", p " + str(p)
                # self.objFileIO.WritedatatoFile(currentSavePath,
                #                                type + "_StatValues", resultValues)
                # print(participant_number + ", " + position + ", " + type + "," + resultValues)
                # print(  type + "," + resultValues)
                Lis3.append( type + "," + resultValues )

                # GenerateObservedvsActual
                mean_squared_error_result, r, p = self.GenerateResultbyAlgorithmType(objPlots,
                                                                                     position,
                                                                                     Actual_all_LastSecond,
                                                                                     Observed_Withoutcheck_all,
                                                                                     AlgorithmName,
                                                                                     currentSavePath,
                                                                                     participant_number,
                                                                                     generateChart,
                                                                                     'HR_' + type + "_WOcheck")

                resultValues = "data last second Wihtout check: mean_squared_error_result = " + str(
                    mean_squared_error_result) + ", r" + str(r) + ", p " + str(p)
                # print(participant_number + ", " + position + ", " + type + "," + resultValues)
                # print( "Last Second GR vs ARPOS Computed without check, " + type + "," + resultValues)
                Lis4.append(type + "," + resultValues)


                # print("Completed for postion: " + position + " and participant: " + participant_number)

            # print("list1")
            # for item in List1:
            #     print(item)
            self.objFileIO.WriteListDatatoFile(currentSavePath,  "ActualAveraged_vs_ARPOSComputed_StatValues", List1)

            # print("list2")
            # for item in Listwo:
            #     print(item)
            self.objFileIO.WriteListDatatoFile(currentSavePath, "ActualAveraged_vs_ARPOSComputedwithoutcheck_StatValues", Listwo)

            # print("list3")
            # for item in Lis3:
            #     print(item)
            self.objFileIO.WriteListDatatoFile(currentSavePath,  "LastSecondGR_vs_ARPOSComputed_StatValues", Lis3)

            # print("list4")
            # for item in Lis4:
            #     print(item)
            self.objFileIO.WriteListDatatoFile(currentSavePath,  "LastSecondGR_vs_ARPOSComputedwithoutcheck_StatValues", Lis4)

            # print("Completed")
            print("Completed for postion: " + position + " and participant: " + participant_number)

    def CreateRestultsForICAOnly(self,position,algodetail,skinType):
        print( "StatisticalResults --> " + str(self.objConfig.windowSize) + " for " + position + " and algo " + algodetail)
        previousPath = self.objConfig.ProcessingSteps[len(self.objConfig.ProcessingSteps) - 1]

        participantsBestTypeLastSec = {}

        dataTypeList = []

        algodetailupdated =algodetail.replace("_Only","")

        prevdatatypepath = self.objConfig.getPathforFolderName(previousPath + "DataFiles", "PIS-1032",position)#any folder just to get all processed item names
        for fileName in os.listdir(prevdatatypepath):  # Load previous step files, for each file
            fileName =fileName.replace(".csv","")
            if ((not fileName.__contains__("ProcessedCompleted"))):
                if ((not fileName.__contains__("Graphs"))):
                    if(not dataTypeList.__contains__(fileName)):
                        if(fileName.__contains__(algodetailupdated)):
                            dataTypeList.append(fileName)

        ICA_pariticipants_rmse = {}
        for processedFile in dataTypeList:
            participantsRMSE = {}
            for participant_number in self.objConfig.ParticipantNumbers:
                # if(participant_number != "PIS-6888" and participant_number != "PIS-6729" and participant_number != "PIS-5868" and participant_number != "PIS-3252"
                #         and participant_number != "PIS-3186" and participant_number != "PIS-7728" ): ##fps 15 or beard
                # if(participant_number == "PIS-6888" or participant_number == "PIS-6729" or participant_number == "PIS-5868" # or participant_number == "PIS-1949"
                #         or participant_number == "PIS-3186" or participant_number == "PIS-7728" ): ##fps 15 or beard
                skinpigPath = self.objConfig.DiskPath + "SerialisedRawServerData\\UnCompressed\\" + participant_number +"\\"
                skinPigmentaion = self.objFileIO.ReaddatafromFile(skinpigPath,"OtherInformation")[0].replace("Skinpigmentation: ","")
                if(skinPigmentaion == skinType):
                    previousfullPath = self.objConfig.getPathforFolderName(previousPath + "DataFiles", participant_number,
                                                                           position)
                    # lOAD Data
                    algorithmDataframe = pd.read_csv(previousfullPath + processedFile + ".csv")  # read file
                    algorithmDataframe = algorithmDataframe.sort_values(by=['WindowCount'], ascending=[True])# sort before other processing

                    diffval = abs(
                        algorithmDataframe[algorithmDataframe.WindowCount == 0]["HRDifference from averaged"][0])
                    # for 2nd wndw
                    diffval2 = abs(
                        algorithmDataframe[algorithmDataframe.WindowCount == 1]["HRDifference from averaged"][1])
                    # for 3 wndw
                    diffval3 = abs(
                        algorithmDataframe[algorithmDataframe.WindowCount == 2]["HRDifference from averaged"][2])
                    # for 4 wndw
                    diffval4 = abs(
                        algorithmDataframe[algorithmDataframe.WindowCount == 3]["HRDifference from averaged"][3])
                    # for 5 wndw
                    diffval5 = abs(
                        algorithmDataframe[algorithmDataframe.WindowCount == 4]["HRDifference from averaged"][4])

                    if (diffval >= 10):
                        algorithmDataframe = algorithmDataframe.drop(
                            algorithmDataframe[algorithmDataframe.WindowCount == 0].index)
                        if (diffval2 >= 10):
                            algorithmDataframe = algorithmDataframe.drop(
                                algorithmDataframe[algorithmDataframe.WindowCount == 1].index)
                            if (diffval3 >= 10):
                                algorithmDataframe = algorithmDataframe.drop(
                                    algorithmDataframe[algorithmDataframe.WindowCount == 2].index)
                                if (diffval4 >= 10):
                                    algorithmDataframe = algorithmDataframe.drop(
                                        algorithmDataframe[algorithmDataframe.WindowCount == 3].index)
                                    if (diffval5 >= 10):
                                        algorithmDataframe = algorithmDataframe.drop(
                                            algorithmDataframe[algorithmDataframe.WindowCount == 4].index)

                    Actual_data = algorithmDataframe['GroundTruth HeartRate Averaged'].tolist()
                    observed_data = algorithmDataframe['Computed HeartRate'].tolist()

                    slope, intercept, r, p, std_err = scipy.stats.linregress(Actual_data, observed_data)
                    mean_squared_error_result = np.sqrt(
                        mean_squared_error(Actual_data, observed_data))  # mean_squared_error_result

                    participantsRMSE[participant_number] = abs(r)#meanValue
            ICA_pariticipants_rmse[processedFile] =participantsRMSE


        new_columns = ['AlgorithmType', 'PreProcess', 'SkinPigmentation', 'RMSE']
        algorithmDataframe = pd.DataFrame(columns=new_columns)
        FinalbestData = {}
        for k,v in ICA_pariticipants_rmse.items():
            Techniques = k.split("_") #ComputerHRandSPO_1_FFT_M4_Algorithm_PCAICA_PreProcess_7
            algoName=Techniques[len(Techniques)-3]
            PreProcess = Techniques[len(Techniques)-1]
            participantsDictionarydata = v
            rmseMean = sum(participantsDictionarydata.values()) / len(participantsDictionarydata)
            FinalbestData[algoName] = rmseMean
            algorithmDataframe = algorithmDataframe.append({'AlgorithmType': algoName,
                                     'PreProcess': PreProcess,
                                     'SkinPigmentation': skinType,
                                     'RMSE': rmseMean }, ignore_index=True)

        return algorithmDataframe

    def CreateRestultsfromBestFileCases(self,position,algodetail):
        processingStep = "StatisticalResults"
        print(processingStep + " --> " + str(self.objConfig.windowSize) + " for " + position + " and algo " + algodetail)
        previousPath = self.objConfig.ProcessingSteps[len(self.objConfig.ProcessingSteps) - 1]

        participantsBestType = {}
        participantsBestTypeLastSec = {}

        dataTypeList = []
        # dataTypeList.append("ComputerHRandSPO_1_FFT_M4_Algorithm_FastICA_PreProcess_2")
        # dataTypeList.append("ComputerHRandSPO_1_FFT_M4_Algorithm_FastICA_PreProcess_6")
        # dataTypeList.append("ComputerHRandSPO_1_FFT_M4_Algorithm_FastICA_PreProcess_7")
        # dataTypeList.append("ComputerHRandSPO_1_FFT_M4_Algorithm_FastICACombined_PreProcess_2")
        # dataTypeList.append("ComputerHRandSPO_1_FFT_M4_Algorithm_FastICACombined_PreProcess_6")
        # dataTypeList.append("ComputerHRandSPO_1_FFT_M4_Algorithm_FastICACombined_PreProcess_7")

        algodetailupdated =algodetail.replace("_Only","")

        prevdatatypepath = self.objConfig.getPathforFolderName(previousPath + "DataFiles", "PIS-1032",position)#any folder just to get all processed item names
        for fileName in os.listdir(prevdatatypepath):  # Load previous step files, for each file
            fileName =fileName.replace(".csv","")
            if ((not fileName.__contains__("ProcessedCompleted"))):
                if ((not fileName.__contains__("Graphs"))):
                    if(not dataTypeList.__contains__(fileName)):
                        if(fileName.__contains__(algodetailupdated)): #REMOVE FOR nONE TODO
                            if(algodetailupdated == "PCA"):
                                if(fileName.__contains__("PCAICA")):
                                    continue
                                else:
                                    dataTypeList.append(fileName)
                            # elif (algodetailupdated == "FastICA"):
                            #     if (fileName.__contains__("FastICA3Times")):
                            #         continue
                            #     else:
                            #         dataTypeList.append(fileName)
                            else:
                                dataTypeList.append(fileName)

        for participant_number in self.objConfig.ParticipantNumbers:
            currentSavePath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(self.objConfig.windowSize)+'\\' + participant_number + "\\" + position + "\\"
            # self.objFileIO.CreatePath(currentSavePath)
            for processedFile in dataTypeList:
                previousfullPath = self.objConfig.getPathforFolderName(previousPath + "DataFiles", participant_number,
                                                                       position)
                # lOAD Data
                algorithmDataframe = pd.read_csv(previousfullPath + processedFile + ".csv")  # read file
                algorithmDataframe = algorithmDataframe.sort_values(by=['WindowCount'], ascending=[True])# sort before other processing

                diffval = abs(
                    algorithmDataframe[algorithmDataframe.WindowCount == 0]["HRDifference from averaged"][0])
                # for 2nd wndw
                diffval2 = abs(
                    algorithmDataframe[algorithmDataframe.WindowCount == 1]["HRDifference from averaged"][1])
                # for 3 wndw
                diffval3 = abs(
                    algorithmDataframe[algorithmDataframe.WindowCount == 2]["HRDifference from averaged"][2])
                # for 4 wndw
                diffval4 = abs(
                    algorithmDataframe[algorithmDataframe.WindowCount == 3]["HRDifference from averaged"][3])
                # for 5 wndw
                diffval5 = abs(
                    algorithmDataframe[algorithmDataframe.WindowCount == 4]["HRDifference from averaged"][4])

                if (diffval >= 10):
                    algorithmDataframe = algorithmDataframe.drop(
                        algorithmDataframe[algorithmDataframe.WindowCount == 0].index)
                    if (diffval2 >= 10):
                        algorithmDataframe = algorithmDataframe.drop(
                            algorithmDataframe[algorithmDataframe.WindowCount == 1].index)
                        if (diffval3 >= 10):
                            algorithmDataframe = algorithmDataframe.drop(
                                algorithmDataframe[algorithmDataframe.WindowCount == 2].index)
                            if (diffval4 >= 10):
                                algorithmDataframe = algorithmDataframe.drop(
                                    algorithmDataframe[algorithmDataframe.WindowCount == 3].index)
                                if (diffval5 >= 10):
                                    algorithmDataframe = algorithmDataframe.drop(
                                        algorithmDataframe[algorithmDataframe.WindowCount == 4].index)

                Actual_data = algorithmDataframe['GroundTruth HeartRate Averaged'].tolist()
                Actual_dataLastSec = algorithmDataframe[' Hr from windows last second'].tolist()
                observed_data = algorithmDataframe['Computed HeartRate'].tolist()
                diffValues = algorithmDataframe['HRDifference from averaged'].tolist()

                # slope, intercept, r, p, std_err = scipy.stats.linregress(Actual_data, observed_data)
                mean_squared_error_result = np.sqrt(
                    mean_squared_error(Actual_data, observed_data))  # mean_squared_error_result
                mean_squared_error_result_lastSec = np.sqrt(
                    mean_squared_error(Actual_dataLastSec, observed_data))  # mean_squared_error_result

                # algorithmDataframe['HRDifference from averaged'] = algorithmDataframe['HRDifference from averaged'].abs()# abs
                # algorithmDataframe[' LastSecondWindow differenceHR'] = algorithmDataframe[' LastSecondWindow differenceHR'].abs()# abs
                # meanValue = algorithmDataframe['HRDifference from averaged'].mean()
                # meanValueLastSec = algorithmDataframe[' LastSecondWindow differenceHR'].mean()

                # print(participant_number + "--> " + processedFile + " mean : " + str(meanValue))

                participantsBestType[participant_number+","+processedFile] = mean_squared_error_result#meanValue
                participantsBestTypeLastSec[participant_number+","+processedFile] = mean_squared_error_result_lastSec

        FinalbestData = {}
        FinalbestDataLastSec = {}
        for participant_number in self.objConfig.ParticipantNumbers:
            participantsDataMeans={}
            for k,v in participantsBestType.items():
                ksplit = k.split(",")
                algoName=ksplit[1]
                pinumber = ksplit[0]
                meanV = v

                if(participant_number != pinumber):
                    continue

                participantsDataMeans[algoName] = meanV

            participantsDataMeansLastSec = {}
            for k, v in participantsBestTypeLastSec.items():
                ksplit = k.split(",")
                algoName = ksplit[1]
                pinumber = ksplit[0]
                meanV = v

                if (participant_number != pinumber):
                    continue

                participantsDataMeansLastSec[algoName] = meanV

            minValue = min(participantsDataMeans.values())# min(participantsDataMeans.keys(), key=(lambda k: participantsDataMeans[k]))
            minValueKey = min(participantsDataMeans, key=participantsDataMeans.get)

            minValuelastsec = min(participantsDataMeansLastSec.values())  # min(participantsDataMeans.keys(), key=(lambda k: participantsDataMeans[k]))
            minValueKeylastsec = min(participantsDataMeansLastSec, key=participantsDataMeansLastSec.get)

            FinalbestData[participant_number + "," + minValueKey] = minValue
            FinalbestDataLastSec[participant_number + "," + minValueKeylastsec] = minValuelastsec


        new_columns = ['id', 'Overall', 'means', 'SkinPigmenation', 'bestdifference', 'bestmeandiff']
        algorithmDataframe = pd.DataFrame(columns=new_columns)
        for k,v in FinalbestData.items():
            # print(k + "-->" + str(v))
            splitKey = k.split(",")
            participantn = splitKey[0]
            algoDetail = splitKey[1]
            computedObject = self.objFileIO.ReadfromDisk(self.objConfig.DiskPath + "ComputerHRandSPODataFiles_W" + str(self.objConfig.windowSize) + "CorrectHR\\"
                                                   +participantn + "\\" + position + "\\" + algoDetail+ "\\", "ProcessedWindow_5")
            fpsnotes = computedObject.FPSNotes
            filecontent = self.objFileIO.ReaddatafromFile(self.objConfig.DiskPath +"SerialisedRawServerData\\UnCompressed\\" + participantn + "\\", "OtherInformation")
            skinType =filecontent[0].replace("Skinpigmentation: ","")
            algorithmDataframe = algorithmDataframe.append({'id': participantn,
                                     'Overall': algoDetail,
                                     'means': v,
                                     'SkinPigmenation': skinType,
                                        'FPSNotes': fpsnotes,
                                     'bestdifference': 0,'bestmeandiff': 0
                                                            }, ignore_index=True)

        algorithmDataframe.to_csv(
            self.objConfig.DiskPath + previousPath + "DataFiles_W" + str(self.objConfig.windowSize) + "\\"  + "BestMeansbyAlgoTypeDataW" + str(self.objConfig.windowSize) + "_"+ position +  algodetail+".csv")  # Write to file

    def CreateRestultsfromParitcipantIdforTestingoNLY(self,position,algodetail, participant_number):
        print("StatisticalResults" + " --> " + str(self.objConfig.windowSize) + " for " +participant_number + ", " + position + " and algo " + algodetail)
        previousPath = self.objConfig.ProcessingSteps[len(self.objConfig.ProcessingSteps) - 1]

        participantsBestType = {}
        participantsBestTypeLastSec = {}

        dataTypeList = []

        algodetailupdated =algodetail.replace("_Only","")

        prevdatatypepath = self.objConfig.getPathforFolderName(previousPath + "DataFiles", "PIS-1032",position)#any folder just to get all processed item names
        for fileName in os.listdir(prevdatatypepath):  # Load previous step files, for each file
            fileName =fileName.replace(".csv","")
            if ((not fileName.__contains__("ProcessedCompleted"))):
                if ((not fileName.__contains__("Graphs"))):
                    if(not dataTypeList.__contains__(fileName)):
                        if(fileName.__contains__(algodetailupdated)): #REMOVE FOR nONE TODO
                            if(algodetailupdated == "PCA"):
                                if(fileName.__contains__("PCAICA")):
                                    continue
                                else:
                                    dataTypeList.append(fileName)
                            else:
                                dataTypeList.append(fileName)

        for processedFile in dataTypeList:
            previousfullPath = self.objConfig.getPathforFolderName(previousPath + "DataFiles", participant_number,
                                                                   position)
            # lOAD Data
            algorithmDataframe = pd.read_csv(previousfullPath + processedFile + ".csv")  # read file

            algorithmDataframe = algorithmDataframe.sort_values(by=['WindowCount'], ascending=[True])# sort before other processing
            
            Actual_data = algorithmDataframe['GroundTruth HeartRate Averaged'].tolist()
            observed_data = algorithmDataframe['Computed HeartRate'].tolist()
            diffValues = algorithmDataframe['HRDifference from averaged'].tolist()

            slope, intercept, r, p, std_err = scipy.stats.linregress(Actual_data, observed_data)
            mean_squared_error_result = np.sqrt(mean_squared_error(Actual_data, observed_data))  # mean_squared_error_result
            rsqrd = r ** 2
            meanABS_ = np.mean(abs(np.array(diffValues)))
            meanWithoutABS_ = np.mean(np.array(diffValues))
            Liststats  = []
            Liststats.append(r)
            Liststats.append(p)
            Liststats.append(rsqrd)
            Liststats.append(meanABS_)
            Liststats.append(meanWithoutABS_)
            Liststats.append(std_err)
            Liststats.append(mean_squared_error_result)
            participantsBestType[participant_number+","+processedFile] = mean_squared_error_result

        for k,v in participantsBestType.items():
            ksplit = k.split(",")
            algoName=ksplit[1]
            pinumber = ksplit[0]
            rmse=v
            print(pinumber + "--> " + algoName + " : "+str(rmse))
            # print("r : " + str(v[0]))
            # print("p : " + str(v[1]))
            # print("rsqrd : " + str(v[2]))
            # print("meanABS_ : " + str(v[3]))
            # print("meanWithoutABS_ : " + str(v[4]))
            # print("std_err : " + str(v[5]))
            # print("mean_squared_error_result : " + str(v[6]))

    def GenerateDataFileWithAllDatafrombestmeans(self,position,AlgorithmName,skintype,loadfilename,Filtered):
        processingStep = "StatisticalResults"
        print(processingStep + " --> " + str(self.objConfig.windowSize))
        previousPath = self.objConfig.ProcessingSteps[len(self.objConfig.ProcessingSteps) - 1]

        datamEans = pd.read_csv(self.objConfig.DiskPath + previousPath +"DataFiles_W" + str(self.objConfig.windowSize) + "\\" + loadfilename+ ".csv")  # read file
        if(skintype == "White"):
            # datamEans=datamEans.loc[(datamEans['SkinPigmenation'] == skintype)   ]
            array = [skintype, 'Asian']
            datamEans=datamEans.loc[datamEans['SkinPigmenation'].isin(array)]
        else:
            datamEans=datamEans.loc[(datamEans['SkinPigmenation'] == "Brown")  ]

        AlgoNames = datamEans['Overall'].tolist()
        AlgoNames = AlgoNames[: 40]
        participantid = datamEans['id'].tolist()
        participantid = participantid[: 40]
        algorithmDataframe_AllData = pd.DataFrame()

        count = 0
        for item in AlgoNames:
            if(item.__contains__(AlgorithmName)):
                curentPiid = participantid[count]
                print(curentPiid)
                previousfullPath = self.objConfig.getPathforFolderName(previousPath + "DataFiles", curentPiid,
                                                                       position)
                algorithmDataframe = pd.read_csv(previousfullPath + item + ".csv")  # read file

                totalrows = algorithmDataframe.shape[0]
                pilIST= []

                for x in range(0,totalrows):
                    pilIST.append(curentPiid)

                algorithmDataframe['ParticipantId'] = pilIST

                filterFolderName = "UnFiltered"
                if(Filtered):
                    diffval = abs(algorithmDataframe[algorithmDataframe.WindowCount == 0]["HRDifference from averaged"][0])
                    # for 2nd wndw
                    diffval2 = abs(algorithmDataframe[algorithmDataframe.WindowCount == 1]["HRDifference from averaged"][1])
                    # for 3 wndw
                    diffval3 = abs(algorithmDataframe[algorithmDataframe.WindowCount == 2]["HRDifference from averaged"][2])
                    # for 4 wndw
                    diffval4 = abs(algorithmDataframe[algorithmDataframe.WindowCount == 3]["HRDifference from averaged"][3])
                    # for 5 wndw
                    diffval5 = abs(algorithmDataframe[algorithmDataframe.WindowCount == 4]["HRDifference from averaged"][4])

                    # found48 = algorithmDataframe[algorithmDataframe['WindowCount'].str.contains('48')]
                    # found49 = algorithmDataframe[algorithmDataframe['WindowCount'].str.contains('49')]
                    # found50 = algorithmDataframe[algorithmDataframe['WindowCount'].str.contains('50')]

                    # for 48 wndw
                    # diffval48 = abs(algorithmDataframe[algorithmDataframe.WindowCount == 48]["HRDifference from averaged"][48])
                    # # for 49 wndw
                    # diffval49 = abs(algorithmDataframe[algorithmDataframe.WindowCount == 49]["HRDifference from averaged"][49])
                    # # for 50 wndw
                    # diffval50 = abs(algorithmDataframe[algorithmDataframe.WindowCount == 50]["HRDifference from averaged"][50])
                    if(diffval >=10):
                        algorithmDataframe = algorithmDataframe.drop(algorithmDataframe[algorithmDataframe.WindowCount == 0].index)
                        if(diffval2 >=10):
                            algorithmDataframe = algorithmDataframe.drop(algorithmDataframe[algorithmDataframe.WindowCount == 1].index)
                            if(diffval3 >=10):
                                algorithmDataframe = algorithmDataframe.drop(algorithmDataframe[algorithmDataframe.WindowCount == 2].index)
                                if (diffval4 >= 10):
                                    algorithmDataframe = algorithmDataframe.drop(
                                        algorithmDataframe[algorithmDataframe.WindowCount == 3].index)
                                    if (diffval5 >= 10):
                                        algorithmDataframe = algorithmDataframe.drop(
                                            algorithmDataframe[algorithmDataframe.WindowCount == 4].index)
                    # if (diffval50 >= 10):
                    #     algorithmDataframe = algorithmDataframe.drop(
                    #         algorithmDataframe[algorithmDataframe.WindowCount == 50].index)
                    #     if (diffval49 >= 10):
                    #         algorithmDataframe = algorithmDataframe.drop(
                    #             algorithmDataframe[algorithmDataframe.WindowCount == 49].index)
                    #         if (diffval48 >= 10):
                    #             algorithmDataframe = algorithmDataframe.drop(
                    #                 algorithmDataframe[algorithmDataframe.WindowCount == 48].index)


                    filterFolderName= "Filtered"

                if (algorithmDataframe_AllData.empty):
                    algorithmDataframe_AllData = algorithmDataframe
                else:
                    algorithmDataframe_AllData = algorithmDataframe_AllData.append(algorithmDataframe, ignore_index=True)


            count =count+1

        self.objFileIO.CreatePath( self.objConfig.DiskPath + processingStep + "DataFiles_W" + str(self.objConfig.windowSize) + "\\"+position+"\\" + str(filterFolderName) + "\\")
        algorithmDataframe_AllData.to_csv(
            self.objConfig.DiskPath + processingStep + "DataFiles_W" + str(self.objConfig.windowSize) + "\\"+position+"\\" + str(filterFolderName) + "\\"+ skintype +"_"+loadfilename + "_SkinoverAllcommon_" + position + ".csv")  # Write to file

    def GenerateGraphfromOneFileforALlSPO(self,position,generateChart,AlgorithmName,allfileName,skintype,filtered):#WhiteoverAllcommon
        objPlots = Plots()
        ListStatValues = []
        processingStep = "StatisticalResults"
        print(processingStep + " --> " + str(self.objConfig.windowSize) + " , "+ AlgorithmName)
        savepath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(
                self.objConfig.windowSize) + '\\' + position+'\\'
        if(filtered):
            savepath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(
                    self.objConfig.windowSize) + '\\' + position+'\\Filtered\\'

        # self.objFileIO.CreatePath(savepath)
        # lOAD Data
        if(skintype == "All"):

            algorithmDataframe1 = pd.read_csv( savepath+ "White" +allfileName +".csv")  # read file
            algorithmDataframe2 = pd.read_csv( savepath+ "Darker" +allfileName +".csv")  # read file

            alldata = [algorithmDataframe1, algorithmDataframe2]
            algorithmDataframe = pd.concat(alldata)
        else:

            algorithmDataframe = pd.read_csv( savepath+ allfileName +".csv")  # read file

        algorithmDataframe=algorithmDataframe.dropna()
        PulseOximeterClipdataAveraged =  algorithmDataframe['GroundTruth SPO Averaged'].tolist()
        WindowCount =  algorithmDataframe['WindowCount'].tolist()
        Observeddata = algorithmDataframe['Computed SPO'].tolist()
        PulseOximeterClipdataLastSecond =  algorithmDataframe['SPOLastSecond'].tolist()
        ObserveddataWithoutCheck =  algorithmDataframe['best SPO WithoutReliability Check'].tolist()
        AVGdiff =  algorithmDataframe['SPO Difference from averaged'].tolist()
        AVgwithoutcheckdiff =  algorithmDataframe['Original Obtianed Average differenceSPO'].tolist()
        LastSecondWindowdifferenceHR =  algorithmDataframe['LastSecondWindowdifferenceSPO '].tolist()
        LastSecondWindowdifferenceHRwithoutcheck =  algorithmDataframe['OriginalObtianedLastSecondWindowdifferenceSPO'].tolist()

        currentSavePath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(self.objConfig.windowSize) + "\\" + position + "\\" +AlgorithmName + "\\"

        if(filtered):
            currentSavePath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(self.objConfig.windowSize) + "\\" + position + "\\Filtered\\" +AlgorithmName + "\\"
            self.objFileIO.CreatePath(currentSavePath)


        print("Standard Deviation of the sample is % s " % (statistics.stdev(AVGdiff)))
        print("Mean of the sample is % s " % (statistics.mean(AVGdiff)))

        print("Standard Deviation of the sample is % s " % (statistics.stdev(LastSecondWindowdifferenceHR)))
        print("Mean of the sample is % s " % (statistics.mean(LastSecondWindowdifferenceHR)))


        # fig = plt.figure()
        #
        # x = np.linspace(-np.pi, np.pi, len(AVGdiff))
        # y = AVGdiff
        # y2=AVgwithoutcheckdiff
        #
        # ax = plt.gca()
        #
        # ax.plot(x, y)
        # ax.plot(x, y2, color='green')
        # ax.grid(True)
        # ax.spines['left'].set_position('zero')
        # ax.spines['right'].set_color('none')
        # ax.spines['bottom'].set_position('zero')
        # ax.spines['top'].set_color('none')
        #
        # plt.xlim(-np.pi, np.pi)
        #
        # plt.savefig("CenterOriginMatplotlib01.png")
        # plt.show()
        # ObserveddataWithoutCheck
        mean_squared_error_result, r, p = self.GenerateResultbyAlgorithmType(objPlots,
                                                                             position,
                                                                             PulseOximeterClipdataAveraged,
                                                                             Observeddata,
                                                                             AlgorithmName,
                                                                             currentSavePath,
                                                                             "All",
                                                                             generateChart,
                                                                             "SPO_All_" + allfileName+ "1_1")
        # print("Avg SPO vs ARPOS computer r value : " + str(r))
        ListStatValues.append("Avg SPO vs ARPOS computer:" )
        ListStatValues.append("r value" + str(r))
        ListStatValues.append("p value" + str(p))
        ListStatValues.append("mean_squared_error_result value" + str(mean_squared_error_result))

        # GenerateObservedvsActual
        mean_squared_error_result, r, p = self.GenerateResultbyAlgorithmType(objPlots,
                                                                             position,
                                                                             PulseOximeterClipdataAveraged,
                                                                             ObserveddataWithoutCheck,
                                                                             AlgorithmName,
                                                                             currentSavePath,
                                                                             "All",
                                                                             generateChart,
                                                                             'SPO_' + allfileName + "2_WOcheck")

        # print("Avg SPO vs best bpm unchecked r value : " + str(r))
        ListStatValues.append("Avg SPO vs best bpm unchecked:" )
        ListStatValues.append("r value" + str(r) )
        ListStatValues.append("p value" + str(p) )
        ListStatValues.append("mean_squared_error_result value" + str(mean_squared_error_result) )

        # ObserveddataWithoutCheck
        mean_squared_error_result, r, p = self.GenerateResultbyAlgorithmType(objPlots,
                                                                             position,
                                                                             PulseOximeterClipdataLastSecond,
                                                                             Observeddata,
                                                                             AlgorithmName,
                                                                             currentSavePath,
                                                                             "All",
                                                                             generateChart,
                                                                             'SPO_All_' + allfileName + "3_1")

        # print("Last sec GR vs ARPOS computed SPO r value : " + str(r))
        ListStatValues.append("Last sec GR vs ARPOS computed SPO :" )
        ListStatValues.append("r value" + str(r) )
        ListStatValues.append("p value" + str(p) )
        ListStatValues.append("mean_squared_error_result value" + str(mean_squared_error_result) )

        # GenerateObservedvsActual
        mean_squared_error_result, r, p = self.GenerateResultbyAlgorithmType(objPlots,
                                                                             position,
                                                                             PulseOximeterClipdataLastSecond,
                                                                             ObserveddataWithoutCheck,
                                                                             AlgorithmName,
                                                                             currentSavePath,
                                                                             "All",
                                                                             generateChart,
                                                                             'SPO_' + allfileName + "4_WOcheck")

        # print("Last sec GR vs best bpm unchecked r value : " + str(r))
        ListStatValues.append("Last sec GR vs best bpm unchecked r value :" )
        ListStatValues.append("r value" + str(r) )
        ListStatValues.append("p value" + str(p) )
        ListStatValues.append("mean_squared_error_result value" + str(mean_squared_error_result) )

        if(filtered):
            self.objFileIO.WriteListDatatoFile(self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(self.objConfig.windowSize) + "\\" + position +"\\Filtered\\" +AlgorithmName + "\\",
                                               "SPO_stat_values_" + skintype, ListStatValues )
        else:
            self.objFileIO.WriteListDatatoFile(self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(self.objConfig.windowSize) + "\\" + position +"\\" +AlgorithmName + "\\",
                                               "SPO_stat_values_" + skintype, ListStatValues )

        # data_dict = {}
        # data_dict["AVGdiff"]=AVGdiff
        # # data_dict["AVgwithoutcheckdiff"]=AVgwithoutcheckdiff
        # data_dict["LastSecondWindowdifferenceHR"]=LastSecondWindowdifferenceHR
        # data_dict["LastSecondWindowdifferenceHRwithoutcheck"]=LastSecondWindowdifferenceHRwithoutcheck

        # fig, ax = plt.subplots()
        # ax.boxplot(data_dict.values())
        # ax.set_xticklabels(data_dict.keys())
        # # plt.boxplot(AVGdiff,AVgwithoutcheckdiff,LastSecondWindowdifferenceHR,LastSecondWindowdifferenceHRwithoutcheck)
        # plt.savefig(savepath + "boxplot_" + allfileName+ ".png")  # Save here
        # plt.close()

    def GenerateGraphfromOneFileforALl(self,position,generateChart,AlgorithmName,allfileName,skintype,filtered):#WhiteoverAllcommon
        objPlots = Plots()
        ListStatValues = []
        processingStep = "StatisticalResults"
        print(processingStep + " --> " + str(self.objConfig.windowSize))
        savepath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(
                self.objConfig.windowSize) + '\\' + position+'\\'
        if(filtered):
            savepath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(
                    self.objConfig.windowSize) + '\\' + position+'\\Filtered\\'

        # self.objFileIO.CreatePath(savepath)
        # lOAD Data
        if(skintype == "All"):

            algorithmDataframe1 = pd.read_csv( savepath+ "White" +allfileName +".csv")  # read file
            algorithmDataframe2 = pd.read_csv( savepath+ "Darker" +allfileName +".csv")  # read file

            alldata = [algorithmDataframe1, algorithmDataframe2]
            algorithmDataframe = pd.concat(alldata)
        else:

            algorithmDataframe = pd.read_csv( savepath+ allfileName +".csv")  # read file

        algorithmDataframe=algorithmDataframe.dropna()
        PulseOximeterClipdataAveraged =  algorithmDataframe['GroundTruth HeartRate Averaged'].tolist()
        Observeddata = algorithmDataframe['Computed HeartRate'].tolist()
        PulseOximeterClipdataLastSecond =  algorithmDataframe[' Hr from windows last second'].tolist()
        ObserveddataWithoutCheck =  algorithmDataframe['bestBpm Without ReliabilityCheck'].tolist()
        AVGdiff =  algorithmDataframe['HRDifference from averaged'].tolist()
        AVgwithoutcheckdiff =  algorithmDataframe['OriginalObtianedAveragedifferenceHR'].tolist()
        LastSecondWindowdifferenceHR =  algorithmDataframe[' LastSecondWindow differenceHR'].tolist()
        LastSecondWindowdifferenceHRwithoutcheck =  algorithmDataframe[' OriginalObtianed LastSecondWindow differenceHR'].tolist()

        currentSavePath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(self.objConfig.windowSize) + "\\" + position + "\\" +AlgorithmName + "\\"

        if(filtered):
            currentSavePath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(self.objConfig.windowSize) + "\\" + position + "\\Filtered\\" +AlgorithmName + "\\"
            self.objFileIO.CreatePath(currentSavePath)


        print("Standard Deviation of the sample is % s " % (statistics.stdev(AVGdiff)))
        print("Mean of the sample is % s " % (statistics.mean(AVGdiff)))

        print("Standard Deviation of the sample is % s " % (statistics.stdev(LastSecondWindowdifferenceHR)))
        print("Mean of the sample is % s " % (statistics.mean(LastSecondWindowdifferenceHR)))

        # ObserveddataWithoutCheck
        mean_squared_error_result, r, p = self.GenerateResultbyAlgorithmType(objPlots,
                                                                             position,
                                                                             PulseOximeterClipdataAveraged,
                                                                             Observeddata,
                                                                             AlgorithmName,
                                                                             currentSavePath,
                                                                             "All",
                                                                             generateChart,
                                                                             "HR_All_" + allfileName+ "1_1")
        # print("Avg HR vs ARPOS computer r value : " + str(r))
        ListStatValues.append("Avg HR vs ARPOS computer:" )
        ListStatValues.append("r value" + str(r))
        ListStatValues.append("p value" + str(p))
        ListStatValues.append("mean_squared_error_result value" + str(mean_squared_error_result))

        # GenerateObservedvsActual
        mean_squared_error_result, r, p = self.GenerateResultbyAlgorithmType(objPlots,
                                                                             position,
                                                                             PulseOximeterClipdataAveraged,
                                                                             ObserveddataWithoutCheck,
                                                                             AlgorithmName,
                                                                             currentSavePath,
                                                                             "All",
                                                                             generateChart,
                                                                             'HR_' + allfileName + "2_WOcheck")

        # print("Avg HR vs best bpm unchecked r value : " + str(r))
        ListStatValues.append("Avg HR vs best bpm unchecked:" )
        ListStatValues.append("r value" + str(r) )
        ListStatValues.append("p value" + str(p) )
        ListStatValues.append("mean_squared_error_result value" + str(mean_squared_error_result) )

        # ObserveddataWithoutCheck
        mean_squared_error_result, r, p = self.GenerateResultbyAlgorithmType(objPlots,
                                                                             position,
                                                                             PulseOximeterClipdataLastSecond,
                                                                             Observeddata,
                                                                             AlgorithmName,
                                                                             currentSavePath,
                                                                             "All",
                                                                             generateChart,
                                                                             'HR_All_' + allfileName + "3_1")

        # print("Last sec GR vs ARPOS computed hr r value : " + str(r))
        ListStatValues.append("Last sec GR vs ARPOS computed hr :" )
        ListStatValues.append("r value" + str(r) )
        ListStatValues.append("p value" + str(p) )
        ListStatValues.append("mean_squared_error_result value" + str(mean_squared_error_result) )

        # GenerateObservedvsActual
        mean_squared_error_result, r, p = self.GenerateResultbyAlgorithmType(objPlots,
                                                                             position,
                                                                             PulseOximeterClipdataLastSecond,
                                                                             ObserveddataWithoutCheck,
                                                                             AlgorithmName,
                                                                             currentSavePath,
                                                                             "All",
                                                                             generateChart,
                                                                             'HR_' + allfileName + "4_WOcheck")

        # print("Last sec GR vs best bpm unchecked r value : " + str(r))
        ListStatValues.append("Last sec GR vs best bpm unchecked r value :" )
        ListStatValues.append("r value" + str(r) )
        ListStatValues.append("p value" + str(p) )
        ListStatValues.append("mean_squared_error_result value" + str(mean_squared_error_result) )

        if(filtered):
            self.objFileIO.WriteListDatatoFile(self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(self.objConfig.windowSize) + "\\" + position +"\\Filtered\\" +AlgorithmName + "\\",
                                               "stat_values_" + skintype, ListStatValues )
        else:
            self.objFileIO.WriteListDatatoFile(self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(self.objConfig.windowSize) + "\\" + position +"\\" +AlgorithmName + "\\",
                                               "stat_values_" + skintype, ListStatValues )

        # data_dict = {}
        # data_dict["AVGdiff"]=AVGdiff
        # # data_dict["AVgwithoutcheckdiff"]=AVgwithoutcheckdiff
        # data_dict["LastSecondWindowdifferenceHR"]=LastSecondWindowdifferenceHR
        # data_dict["LastSecondWindowdifferenceHRwithoutcheck"]=LastSecondWindowdifferenceHRwithoutcheck

        # fig, ax = plt.subplots()
        # ax.boxplot(data_dict.values())
        # ax.set_xticklabels(data_dict.keys())
        # # plt.boxplot(AVGdiff,AVgwithoutcheckdiff,LastSecondWindowdifferenceHR,LastSecondWindowdifferenceHRwithoutcheck)
        # plt.savefig(savepath + "boxplot_" + allfileName+ ".png")  # Save here
        # plt.close()

    def BlandAltmanPlotMain(self,position,skintype,algoName):
        savepathResting1 = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(self.objConfig.windowSize) + '\\' + position + '\\'
        # algorithmDataframe = pd.read_csv(savepathResting1 + "Finalised\\" + skintype +"overAllcommonFinalised" + ".csv")  # read file
        algorithmDataframe = pd.read_csv(savepathResting1 + "Filtered\\" +skintype +"_BestMeansbyAlgoTypeDataW" +str(self.objConfig.windowSize)  +
                                         "_" + position + "_Only"+ algoName + "_SkinoverAllcommon_" + position + ".csv")  # read file
        algorithmDataframe = algorithmDataframe.dropna()
        PulseOximeterClipdataLastSEc =  algorithmDataframe[' LastSecondWindow differenceHR'].tolist()
        PulseOximeterClipdataAveraged =  algorithmDataframe['GroundTruth HeartRate Averaged'].tolist()
        Observeddata = algorithmDataframe['Computed HeartRate'].tolist()
        #
        # algorithmDataframe = pd.read_csv(
        #     savepathResting1 + "White_BestMeansbyAlgoTypeDataW15_Resting1_OnlyFastICA_SkinoverAllcommon_Resting1" + ".csv")  # read file
        # algorithmDataframe = algorithmDataframe.dropna()
        # AVGdiff = algorithmDataframe['HRDifference from averaged'].tolist()
        # AVgwithoutcheckdiff = algorithmDataframe['OriginalObtianedAveragedifferenceHR'].tolist()
        # LastSecondWindowdifferenceHR = algorithmDataframe[' LastSecondWindow differenceHR'].tolist()
        # LastSecondWindowdifferenceHRwithoutcheck = algorithmDataframe[
        #     ' OriginalObtianed LastSecondWindow differenceHR'].tolist()

        # create Bland-Altman plot

        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        plt.figure(figsize=(14, 14))
        plt.rc('font', size=14)
        plt.tight_layout()
        plt.title('Bland-Altman Plot')

        # make boxplot with Seaborn
        self.bland_altman_plot(PulseOximeterClipdataAveraged, Observeddata)

        plt.savefig(self.objConfig.DiskPath + "\\StatisticalResultsDataFiles_W" + str(
            self.objConfig.windowSize) + "\\" + "bland_altman_plot_" + skintype + "_" + position + "_" +algoName+ ".png")
        print("StatisticalResultsDataFiles_W" + str(
            self.objConfig.windowSize) + "\\" + "bland_altman_plot_" + skintype + "_" + position + "_" +algoName+ ".png")


        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        plt.figure(figsize=(14, 14))
        plt.rc('font', size=14)
        plt.tight_layout()
        plt.title('Bland-Altman Plot')

        # make boxplot with Seaborn
        self.bland_altman_plot(PulseOximeterClipdataLastSEc, Observeddata)

        plt.savefig(self.objConfig.DiskPath + "\\StatisticalResultsDataFiles_W" + str(
            self.objConfig.windowSize) + "\\" + "bland_altman_plot_LastSec_" + skintype + "_" + position + "_" +algoName+ ".png")
        print("StatisticalResultsDataFiles_W" + str(
            self.objConfig.windowSize) + "\\" + "bland_altman_plot_LastSec_" + skintype + "_" + position + "_" +algoName+ ".png")

    def bland_altman_plot(self,data1, data2, *args, **kwargs):
        data1 = np.asarray(data1)
        data2 = np.asarray(data2)
        mean = np.mean([data1, data2], axis=0)
        diff = data1 - data2  # Difference between data1 and data2
        md = np.mean(diff)  # Mean of the difference
        sd = np.std(diff, axis=0)  # Standard deviation of the difference

        plt.scatter(mean, diff, *args, **kwargs)
        plt.axhline(md, color='gray', linestyle='--')
        plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
        plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
        plt.show()

    def DiffPlotMain(self,skintype,position, diffType,saveType):
        algorithmDataframe = pd.DataFrame()

        for algorithm in self.objConfig.AlgoList:
            ##Boxplot RMSE over resting1, resting2 and after exc
            AVGdiff,GroundTruth,GRLastSecond,Computed,WithoutReliability = self.DiffPlot(skintype,position,algorithm,diffType)

            algoList = []
            for x in range (0,len(AVGdiff)):
                algoList.append(algorithm)

            if (algorithmDataframe.empty):
                algorithmDataframe["Algorithm"] = algoList
                algorithmDataframe[skintype+"AVGdiff"] = AVGdiff
                # algorithmDataframe["DarkRMSE"] = rmseArray_Dark
            else:
                for item in AVGdiff:
                    s = pd.Series([algorithm, item], index=['Algorithm', skintype+'AVGdiff'])
                    algorithmDataframe = algorithmDataframe.append(s, ignore_index=True)

        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        plt.figure(figsize=(14, 14))
        plt.rc('font', size=14)
        plt.tight_layout()

        # make boxplot with Seaborn
        bplot = sns.boxplot(y='Algorithm', x=skintype+'AVGdiff',
                            data=algorithmDataframe,
                            width=0.5,
                            palette="colorblind", showfliers=False)
        plt.savefig(self.objConfig.DiskPath + "\\StatisticalResultsDataFiles_W" +str(self.objConfig.windowSize) + "\\" + "boxplotal" + saveType + "agorithm_"+skintype+"_" + position+".png")
        print("StatisticalResultsDataFiles_W" +str(self.objConfig.windowSize) + "\\" + "boxplotalgorithm_"+skintype+"_" + position+".png")
        
    def DiffPlotMainSPO(self,skintype,position, diffType,saveType):
        algorithmDataframe = pd.DataFrame()
        algorithm = "FastICA"
        ##Boxplot RMSE over resting1, resting2 and after exc
        AVGdiff,GroundTruth,GRLastSecond,Computed,WithoutReliability = self.DiffPlot(skintype,position,algorithm,diffType)

        count = 0
        for item in GroundTruth:
            temp = pd.DataFrame(
                {
                    'Type': 'GroundTruthAveraged',
                    'value': item
                }, index=[0]
            )
            algorithmDataframe = pd.concat([algorithmDataframe, temp])
            count = count + 1

        count = 0
        for item in GRLastSecond:
            temp = pd.DataFrame(
                {
                    'Type': 'GroundTruthLastSecond',
                    'value': item
                }, index=[0]
            )
            algorithmDataframe = pd.concat([algorithmDataframe, temp])
            count = count + 1

        count = 0
        for item in Computed:
            temp = pd.DataFrame(
                {
                    'Type': 'ARPOSComputed',
                    'value': item
                }, index=[0]
            )
            algorithmDataframe = pd.concat([algorithmDataframe, temp])
            count = count + 1

        count = 0
        for item in WithoutReliability:
            temp = pd.DataFrame(
                {
                    'Type': 'ARPOSComputedwithoutReliabilityCheck',
                    'value': item
                }, index=[0]
            )
            algorithmDataframe = pd.concat([algorithmDataframe, temp])
            count = count + 1

        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        plt.figure(figsize=(14, 14))
        plt.rc('font', size=14)
        plt.tight_layout()

        # make boxplot with Seaborn
        bplot = sns.boxplot(y='Type', x='value',
                            data=algorithmDataframe,
                            width=0.5,
                            palette="colorblind", showfliers=False)
        plt.savefig(self.objConfig.DiskPath + "\\StatisticalResultsDataFiles_W" +str(self.objConfig.windowSize) + "\\" + "SPOplotICA_"+skintype+"_" + position+".png")
        print("StatisticalResultsDataFiles_W" +str(self.objConfig.windowSize) + "\\" + "boxplotalgorithm_"+skintype+"_" + position+".png")

    def DiffPlot(self,skintype,position,algorithmName, differenceType):#WhiteoverAllcommon
        savepathResting1 = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(
                self.objConfig.windowSize) + '\\' + position+'\\Filtered\\'

        # lOAD Data
        if(skintype == "All"):
            algorithmDataframe1 = pd.read_csv(savepathResting1 + "White_BestMeansbyAlgoTypeDataW" + str(self.objConfig.windowSize) + "_" + position +
                                             "_Only" + algorithmName + "_SkinoverAllcommon_" + position  + ".csv")  # read file
            algorithmDataframe2 = pd.read_csv(savepathResting1 + "Darker_BestMeansbyAlgoTypeDataW" + str(self.objConfig.windowSize) + "_" + position +
                                             "_Only" + algorithmName + "_SkinoverAllcommon_" + position  + ".csv")  # read file

            alldata = [algorithmDataframe1, algorithmDataframe2]
            algorithmDataframe = pd.concat(alldata)
        else:
            algorithmDataframe = pd.read_csv(savepathResting1 + skintype +"_BestMeansbyAlgoTypeDataW" + str(self.objConfig.windowSize) + "_" + position +
                                             "_Only" + algorithmName + "_SkinoverAllcommon_" + position  + ".csv")  # read file
        algorithmDataframe = algorithmDataframe.dropna()
        AVGdiff = algorithmDataframe[differenceType].tolist()
        if(differenceType.__contains__("SPO")):
            GroundTruth = algorithmDataframe["GroundTruth SPO Averaged"].tolist()
            GRLastSecond = algorithmDataframe["SPOLastSecond"].tolist()
            Computed = algorithmDataframe["Computed SPO"].tolist()
            WithoutReliability = algorithmDataframe["best SPO WithoutReliability Check"].tolist()
        else:
            GroundTruth = algorithmDataframe["GroundTruth HeartRate Averaged"].tolist()
            GRLastSecond = algorithmDataframe[" Hr from windows last second"].tolist()
            Computed = algorithmDataframe["Computed HeartRate"].tolist()
            WithoutReliability = algorithmDataframe["bestBpm Without ReliabilityCheck"].tolist()
        return AVGdiff,GroundTruth,GRLastSecond,Computed,WithoutReliability

    def RMSEPlotMain(self):
        algorithmDataframe = pd.DataFrame()
        dictionalgoMeanDark = {}
        dictionalgoMeanWhite = {}
        for algorithm in self.objConfig.AlgoList:
            ##Boxplot RMSE over resting1, resting2 and after exc
            rmseArray_White, rmseArray_Dark = self.RMSEPlot(algorithm)

            dictionalgoMeanWhite[algorithm] = np.mean(rmseArray_White)
            dictionalgoMeanDark[algorithm] = np.mean(rmseArray_Dark)

            if (algorithmDataframe.empty):
                algorithmDataframe["Algorithm"] = [algorithm, algorithm, algorithm]
                algorithmDataframe["WhiteRMSE"] = rmseArray_White
                algorithmDataframe["DarkRMSE"] = rmseArray_Dark
            else:
                s = pd.Series([algorithm, rmseArray_White[0], rmseArray_Dark[0]],
                              index=['Algorithm', 'WhiteRMSE', 'DarkRMSE'])
                algorithmDataframe = algorithmDataframe.append(s, ignore_index=True)
                s = pd.Series([algorithm, rmseArray_White[1], rmseArray_Dark[1]],
                              index=['Algorithm', 'WhiteRMSE', 'DarkRMSE'])
                algorithmDataframe = algorithmDataframe.append(s, ignore_index=True)
                s = pd.Series([algorithm, rmseArray_White[2], rmseArray_Dark[2]],
                              index=['Algorithm', 'WhiteRMSE', 'DarkRMSE'])
                algorithmDataframe = algorithmDataframe.append(s, ignore_index=True)

        # make boxplot with Seaborn
        # bplot = sns.boxplot(y='Algorithm', x='WhiteRMSE',
        #                     data=algorithmDataframe,
        #                     width=0.5,
        #                     palette="colorblind")
        # plt.show()

        for k, v in dictionalgoMeanWhite.items():
            print(k + " for White pigmentation : " + str(v))

        for k, v in dictionalgoMeanDark.items():
            print(k + " for Darker skin pigmentation : " + str(v))

        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        plt.figure(figsize=(14, 14))
        plt.rc('font', size=14)
        plt.tight_layout()

        bplot = sns.boxplot(y='Algorithm', x='WhiteRMSE',
                            data=algorithmDataframe,
                            width=0.5,
                            palette="colorblind")

        plt.savefig(self.objConfig.DiskPath + "\\StatisticalResultsDataFiles_W" + str(
            self.objConfig.windowSize) + "\\" + "boxplotalgorithmRMSE_White.png")
        print("StatisticalResultsDataFiles_W" + str(
            self.objConfig.windowSize) + "\\" + "boxplotalgorithmRMSE_White.png")

        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        plt.figure(figsize=(14, 14))
        plt.rc('font', size=14)
        plt.tight_layout()

        bplot = sns.boxplot(y='Algorithm', x='DarkRMSE',
                            data=algorithmDataframe,
                            width=0.5,
                            palette="colorblind")

        plt.savefig(self.objConfig.DiskPath + "\\StatisticalResultsDataFiles_W" + str(
            self.objConfig.windowSize) + "\\" + "boxplotalgorithmRMSE_Darker.png")
        print("StatisticalResultsDataFiles_W" + str(
            self.objConfig.windowSize) + "\\" + "boxplotalgorithmRMSE_Darker.png")

    def RMSEPlot(self,AlgorithmName):#WhiteoverAllcommon
        objPlots = Plots()
        ListStatValues = []
        processingStep = "StatisticalResults"
        print(processingStep + " --> " + str(self.objConfig.windowSize))
        savepathResting1 = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(
                self.objConfig.windowSize) + '\\' + "Resting1"+'\\Filtered\\' + AlgorithmName+'\\'
        savepathResting2 = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(
                self.objConfig.windowSize) + '\\' + "Resting2"+'\\Filtered\\'+ AlgorithmName+'\\'
        savepathAfterExc = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(
                self.objConfig.windowSize) + '\\' + "AfterExcersize"+'\\Filtered\\'+ AlgorithmName+'\\'

        rmseArray_White = []
        rmseArray_Dark = []

        # lOAD Data
        ##Retsing1
        darkerValues = self.objFileIO.ReaddatafromFile(savepathResting1,"stat_values_Darker")
        whiteValues = self.objFileIO.ReaddatafromFile(savepathResting1,"stat_values_White")
        #Avg HR vs ARPOS computer:
        rmseDarker = darkerValues[3].replace("mean_squared_error_result value","")
        rmseWhtie = whiteValues[3].replace("mean_squared_error_result value","")
        rmseDarker = rmseDarker.replace("\n","")
        rmseWhtie = rmseWhtie.replace("\n","")
        rmseArray_White.append(float(rmseWhtie))
        rmseArray_Dark.append(float(rmseDarker))

        ##Retsing2
        darkerValues = self.objFileIO.ReaddatafromFile(savepathResting2,"stat_values_Darker")
        whiteValues = self.objFileIO.ReaddatafromFile(savepathResting2,"stat_values_White")
        #Avg HR vs ARPOS computer:
        rmseDarker = darkerValues[3].replace("mean_squared_error_result value","")
        rmseWhtie = whiteValues[3].replace("mean_squared_error_result value","")
        rmseDarker = rmseDarker.replace("\n","")
        rmseWhtie = rmseWhtie.replace("\n","")
        rmseArray_White.append(float(rmseWhtie))
        rmseArray_Dark.append(float(rmseDarker))

        ##AfterExcersize
        darkerValues = self.objFileIO.ReaddatafromFile(savepathAfterExc,"stat_values_Darker")
        whiteValues = self.objFileIO.ReaddatafromFile(savepathAfterExc,"stat_values_White")
        #Avg HR vs ARPOS computer:
        rmseDarker = darkerValues[3].replace("mean_squared_error_result value","")
        rmseWhtie = whiteValues[3].replace("mean_squared_error_result value","")
        rmseDarker = rmseDarker.replace("\n","")
        rmseWhtie = rmseWhtie.replace("\n","")
        rmseArray_White.append(float(rmseWhtie))
        rmseArray_Dark.append(float(rmseDarker))

        return rmseArray_White,rmseArray_Dark

    def RvalPlotMainIndividual(self,position):
        algorithmDataframe = pd.DataFrame()

        rmseArray_White = []
        rmseArray_Dark = []
        allAlgos = self.objConfig.AlgoList

        for algorithm in self.objConfig.AlgoList:
            objPlots = Plots()
            savepathResting1 = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(
                self.objConfig.windowSize) + '\\' + position + '\\Filtered\\' + algorithm + '\\'

            # lOAD Data
            ##Retsing1
            darkerValues = self.objFileIO.ReaddatafromFile(savepathResting1, "stat_values_Darker")
            whiteValues = self.objFileIO.ReaddatafromFile(savepathResting1, "stat_values_White")
            # Avg HR vs ARPOS computer:
            rmseDarker = darkerValues[1].replace("r value", "")
            rmseWhtie = whiteValues[1].replace("r value", "")
            rmseDarker = rmseDarker.replace("\n", "")
            rmseWhtie = rmseWhtie.replace("\n", "")
            rmseArray_White.append(float(rmseWhtie))
            rmseArray_Dark.append(float(rmseDarker))

        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        fig = plt.figure()
        plt.tight_layout()
        plt.xticks(rotation=90)
        plt.bar(allAlgos, rmseArray_White)

        plt.savefig(self.objConfig.DiskPath + "\\StatisticalResultsDataFiles_W" + str(
            self.objConfig.windowSize) + "\\" + "boxplotalgorithmrVal_White_" + position + ".png")
        print("StatisticalResultsDataFiles_W" + str(
            self.objConfig.windowSize) + "\\" + "boxplotalgorithmrVal_White_" + position + ".png")

        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        fig = plt.figure()
        plt.tight_layout()
        plt.xticks(rotation=90)
        plt.bar(allAlgos, rmseArray_Dark)

        plt.savefig(self.objConfig.DiskPath + "\\StatisticalResultsDataFiles_W" + str(
            self.objConfig.windowSize) + "\\" + "boxplotalgorithmrVal_Darker_" + position + ".png")
        print("StatisticalResultsDataFiles_W" + str(
            self.objConfig.windowSize) + "\\" + "boxplotalgorithmrVal_Darker_" + position + ".png")

    def RvalPlotMain(self):
        algorithmDataframe = pd.DataFrame()

        dictionalgoMeanWhite={}
        dictionalgoMeanDark={}

        for algorithm in self.objConfig.AlgoList:
            ##Boxplot RMSE over resting1, resting2 and after exc
            rmseArray_White, rmseArray_Dark = self.RvalPlot(algorithm)

            dictionalgoMeanWhite[algorithm] = np.mean(rmseArray_White)
            dictionalgoMeanDark[algorithm] = np.mean(rmseArray_Dark)

            if (algorithmDataframe.empty):
                algorithmDataframe["Algorithm"] = [algorithm, algorithm, algorithm]
                algorithmDataframe["White"] = rmseArray_White
                algorithmDataframe["Darker"] = rmseArray_Dark
            else:
                s = pd.Series([algorithm, rmseArray_White[0], rmseArray_Dark[0]],
                              index=['Algorithm', 'White', 'Darker'])
                algorithmDataframe = algorithmDataframe.append(s, ignore_index=True)
                s = pd.Series([algorithm, rmseArray_White[1], rmseArray_Dark[1]],
                              index=['Algorithm', 'White', 'Darker'])
                algorithmDataframe = algorithmDataframe.append(s, ignore_index=True)
                s = pd.Series([algorithm, rmseArray_White[2], rmseArray_Dark[2]],
                              index=['Algorithm', 'White', 'Darker'])
                algorithmDataframe = algorithmDataframe.append(s, ignore_index=True)

        # make boxplot with Seaborn
        # bplot = sns.boxplot(y='Algorithm', x='WhiteRMSE',
        #                     data=algorithmDataframe,
        #                     width=0.5,
        #                     palette="colorblind")
        # plt.show()

        for k, v in dictionalgoMeanWhite.items():
            print(k + " for White pigmentation : " + str(v))

        for k, v in dictionalgoMeanDark.items():
            print(k + " for Darker skin pigmentation : " + str(v))

        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        plt.figure(figsize=(14, 14))
        plt.rc('font', size=14)
        plt.tight_layout()

        bplot = sns.boxplot(y='Algorithm', x='White',
                            data=algorithmDataframe,
                            width=0.5,
                            palette="colorblind")

        plt.savefig(self.objConfig.DiskPath + "\\StatisticalResultsDataFiles_W" + str(
            self.objConfig.windowSize) + "\\" + "boxplotalgorithmrVal_White.png")
        print("StatisticalResultsDataFiles_W" + str(
            self.objConfig.windowSize) + "\\" + "boxplotalgorithmrVal_White.png")

        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        plt.figure(figsize=(14, 14))
        plt.rc('font', size=14)
        plt.tight_layout()

        bplot = sns.boxplot(y='Algorithm', x='Darker',
                            data=algorithmDataframe,
                            width=0.5,
                            palette="colorblind")

        plt.savefig(self.objConfig.DiskPath + "\\StatisticalResultsDataFiles_W" + str(
            self.objConfig.windowSize) + "\\" + "boxplotalgorithmrVal_Darker.png")
        print("StatisticalResultsDataFiles_W" + str(
            self.objConfig.windowSize) + "\\" + "boxplotalgorithmrVal_Darker.png")

    def RvalPlot(self,AlgorithmName):#WhiteoverAllcommon
        objPlots = Plots()
        ListStatValues = []
        processingStep = "StatisticalResults"
        print(processingStep + " --> " + str(self.objConfig.windowSize))
        savepathResting1 = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(
                self.objConfig.windowSize) + '\\' + "Resting1"+'\\Filtered\\' + AlgorithmName+'\\'
        savepathResting2 = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(
                self.objConfig.windowSize) + '\\' + "Resting2"+'\\Filtered\\'+ AlgorithmName+'\\'
        savepathAfterExc = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(
                self.objConfig.windowSize) + '\\' + "AfterExcersize"+'\\Filtered\\'+ AlgorithmName+'\\'

        rmseArray_White = []
        rmseArray_Dark = []

        # lOAD Data
        ##Retsing1
        darkerValues = self.objFileIO.ReaddatafromFile(savepathResting1,"stat_values_Darker")
        whiteValues = self.objFileIO.ReaddatafromFile(savepathResting1,"stat_values_White")
        #Avg HR vs ARPOS computer:
        rmseDarker = darkerValues[1].replace("r value","")
        rmseWhtie = whiteValues[1].replace("r value","")
        rmseDarker = rmseDarker.replace("\n","")
        rmseWhtie = rmseWhtie.replace("\n","")
        rmseArray_White.append(float(rmseWhtie))
        rmseArray_Dark.append(float(rmseDarker))

        ##Retsing2
        darkerValues = self.objFileIO.ReaddatafromFile(savepathResting2,"stat_values_Darker")
        whiteValues = self.objFileIO.ReaddatafromFile(savepathResting2,"stat_values_White")
        #Avg HR vs ARPOS computer:
        rmseDarker = darkerValues[1].replace("r value","")
        rmseWhtie = whiteValues[1].replace("r value","")
        rmseDarker = rmseDarker.replace("\n","")
        rmseWhtie = rmseWhtie.replace("\n","")
        rmseArray_White.append(float(rmseWhtie))
        rmseArray_Dark.append(float(rmseDarker))

        ##AfterExcersize
        darkerValues = self.objFileIO.ReaddatafromFile(savepathAfterExc,"stat_values_Darker")
        whiteValues = self.objFileIO.ReaddatafromFile(savepathAfterExc,"stat_values_White")
        #Avg HR vs ARPOS computer:
        rmseDarker = darkerValues[1].replace("r value","")
        rmseWhtie = whiteValues[1].replace("r value","")
        rmseDarker = rmseDarker.replace("\n","")
        rmseWhtie = rmseWhtie.replace("\n","")
        rmseArray_White.append(float(rmseWhtie))
        rmseArray_Dark.append(float(rmseDarker))

        return rmseArray_White,rmseArray_Dark

    def ComputeStatResultsForParticiapnts(self,position,generateChart):
        objPlots = Plots()
        processingStep = "StatisticalResults"
        print(processingStep + " --> " + str(self.objConfig.windowSize))
        previousPath = self.objConfig.ProcessingSteps[len(self.objConfig.ProcessingSteps) - 1]

        dataTypeList = []

        dataDictionarybyType = {}

        prevdatatypepath = self.objConfig.getPathforFolderName(previousPath + "DataFiles", "PIS-1032",position)#any folder just to get all processed item names
        for fileName in os.listdir(prevdatatypepath):  # Load previous step files, for each file
            fileName =fileName.replace(".csv","")
            if ((not fileName.__contains__("ProcessedCompleted"))):
                if ((not fileName.__contains__("Graphs"))):
                    if(not dataTypeList.__contains__(fileName)):
                        dataTypeList.append(fileName)

        objDataList =[]
        for processedFile in dataTypeList:
            type =processedFile
            Actual_all_averaged = []
            Actual_all_LastSecond = []
            Observed_all = []
            Observed_Withoutcheck_all = []
            Observed_all_Difference_avged= []
            Observed_all_Difference_LastWindow= []

            fileName = processedFile
            splitFileName = fileName.split("_")
            AlgorithmName = splitFileName[len(splitFileName) - 3]

            for participant_number in self.objConfig.ParticipantNumbers:
                currentSavePath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(
                    self.objConfig.windowSize) + '\\' + participant_number + "\\" + position + "\\"
                self.objFileIO.CreatePath(currentSavePath)

                # participant_number = "PIS-256"
                previousfullPath = self.objConfig.getPathforFolderName(previousPath + "DataFiles", participant_number,
                                                                       position)

                # lOAD Data
                algorithmDataframe = pd.read_csv(previousfullPath + fileName + ".csv")  # read file

                algorithmDataframe = algorithmDataframe.sort_values(by=['WindowCount'], ascending=[True])# sort before other processing

                PulseOximeterClipdataAveraged =  algorithmDataframe['GroundTruth HeartRate Averaged'].tolist()
                PulseOximeterClipdataLastSecond =  algorithmDataframe[' Hr from windows last second'].tolist()
                Observeddata = algorithmDataframe['Computed HeartRate'].tolist()
                ObserveddataWithoutCheck =  algorithmDataframe['bestBpm Without ReliabilityCheck'].tolist()
                ObserveddataDifferenceAVG =  algorithmDataframe['HRDifference from averaged'].tolist()
                ObserveddataDifferenceLastWindow =  algorithmDataframe[' LastSecondWindow differenceHR'].tolist()
                # ObserveddataWithoutCheckDifference =  algorithmDataframe['bestBpmWithoutReliabilityCheckDifference'].tolist()

                # if(AlgorithmName.__contains__("FastICACombined")):
                # data_tuples = list(zip(algorithmDataframe['WindowCount'].tolist(),PulseOximeterClipdataAveraged, PulseOximeterClipdataLastSecond,
                #                        Observeddata,ObserveddataWithoutCheck,ObserveddataDifferenceAVG,ObserveddataDifferenceLastWindow))
                # dfARPOS = pd.DataFrame(data_tuples, columns=['WindowCount', 'PulseOximeterClipdataAveraged','PulseOximeterClipdataLastSecond',
                #                                              'Observeddata','ObserveddataWithoutCheck','ObserveddataDifferenceAVG','ObserveddataDifferenceLastWindow'])
                #     correlation = dfARPOS.corr()
                #     print(type(correlation))

                Actual_all_averaged =Actual_all_averaged + PulseOximeterClipdataAveraged
                Actual_all_LastSecond =Actual_all_LastSecond + PulseOximeterClipdataLastSecond
                Observed_all=Observed_all+Observeddata
                Observed_all_Difference_avged = Observed_all_Difference_avged +ObserveddataDifferenceAVG
                Observed_all_Difference_LastWindow = Observed_all_Difference_LastWindow + ObserveddataDifferenceLastWindow
                Observed_Withoutcheck_all=Observed_Withoutcheck_all+ObserveddataWithoutCheck


            currentSavePath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(self.objConfig.windowSize) + "\\"
            # ObserveddataWithoutCheck
            mean_squared_error_result, r, p = self.GenerateResultbyAlgorithmType(objPlots,
                                                                                 position,
                                                                                 Actual_all_averaged,
                                                                                 Observed_all,
                                                                                 AlgorithmName,
                                                                                 currentSavePath,
                                                                                 participant_number,
                                                                                 generateChart,
                                                                                 'HR_All_' + type + "_1")

            # GenerateObservedvsActual
            mean_squared_error_result, r, p = self.GenerateResultbyAlgorithmType(objPlots,
                                                                                 position,
                                                                                 Actual_all_averaged,
                                                                                 Observed_Withoutcheck_all,
                                                                                 AlgorithmName,
                                                                                 currentSavePath,
                                                                                 participant_number,
                                                                                 generateChart,
                                                                                 'HR_' + type + "_WOcheck")


            # ObserveddataWithoutCheck
            mean_squared_error_result, r, p = self.GenerateResultbyAlgorithmType(objPlots,
                                                                                 position,
                                                                                 Actual_all_LastSecond,
                                                                                 Observed_all,
                                                                                 AlgorithmName,
                                                                                 currentSavePath,
                                                                                 participant_number,
                                                                                 generateChart,
                                                                                 'HR_All_' + type + "_1")

            # GenerateObservedvsActual
            mean_squared_error_result, r, p = self.GenerateResultbyAlgorithmType(objPlots,
                                                                                 position,
                                                                                 Actual_all_LastSecond,
                                                                                 Observed_Withoutcheck_all,
                                                                                 AlgorithmName,
                                                                                 currentSavePath,
                                                                                 participant_number,
                                                                                 generateChart,
                                                                                 'HR_' + type + "_WOcheck")


            print("Completed for type: " + type )
            dataDictionarybyType[type] = Observed_all_Difference_avged
            objData = DataTypeObject
            objData.color = ["#" + ''.join([random.choice('ABCDEF0123456789') for i in range(6)])]
            objData.label = type
            objData.datapoints = Observed_all
            objData.DifferenceValues = Observed_all_Difference_avged
            objData.DifferenceValuesLastWindow = ObserveddataDifferenceLastWindow
            objDataList.append(objData)

        currentSavePath = self.objConfig.DiskPath + "StatisticalResultsDataFiles_W" + str(self.objConfig.windowSize)+ "\\"

        self.Plotdataboxplot(dataDictionarybyType,dataTypeList,currentSavePath,"boxPlotoverAll")

        self.GenearateHistorgram(currentSavePath,objDataList,
                                 position,position + "_HistogramoverAllComputedARPOSValues",False)
        self.GenearateHistorgram(currentSavePath,objDataList,
                                 position,position + "_HistogramoverAllComputedARPOSValues_LastSec",True)

    def Plotdataboxplot(self, dictionarydataList,dataTypeList,DirectoryPath,fileName):
        plt.boxplot([dictionarydataList[dataTypeList[0]],dictionarydataList[dataTypeList[1]],dictionarydataList[dataTypeList[2]],dictionarydataList[dataTypeList[3]],
                     dictionarydataList[dataTypeList[4]],dictionarydataList[dataTypeList[5]],dictionarydataList[dataTypeList[6]],
                     dictionarydataList[dataTypeList[7]],dictionarydataList[dataTypeList[8]]])

        plt.savefig(DirectoryPath + fileName + ".png")  # Save here
        plt.close()

        plt.clf()

        plt.boxplot([dictionarydataList[dataTypeList[9]], dictionarydataList[dataTypeList[10]],
                     dictionarydataList[dataTypeList[11]], dictionarydataList[dataTypeList[12]],
                     dictionarydataList[dataTypeList[13]], dictionarydataList[dataTypeList[14]],
                     dictionarydataList[dataTypeList[15]], dictionarydataList[dataTypeList[16]],
                     dictionarydataList[dataTypeList[17]]])
        plt.savefig(DirectoryPath + fileName + "2.png")  # Save here
        plt.close()

        plt.clf()

        plt.boxplot([dictionarydataList[dataTypeList[18]], dictionarydataList[dataTypeList[19]],
                     dictionarydataList[dataTypeList[20]], dictionarydataList[dataTypeList[21]],
                     dictionarydataList[dataTypeList[22]], dictionarydataList[dataTypeList[23]],
                     dictionarydataList[dataTypeList[24]], dictionarydataList[dataTypeList[25]],
                     dictionarydataList[dataTypeList[26]]])
        plt.savefig(DirectoryPath + fileName + "3.png")  # Save here
        plt.close()

        plt.clf()

        plt.boxplot([dictionarydataList[dataTypeList[27]], dictionarydataList[dataTypeList[28]],
                     dictionarydataList[dataTypeList[29]], dictionarydataList[dataTypeList[30]],
                     dictionarydataList[dataTypeList[31]], dictionarydataList[dataTypeList[32]],
                     dictionarydataList[dataTypeList[33]], dictionarydataList[dataTypeList[34]],
                     dictionarydataList[dataTypeList[35]]])
        plt.savefig(DirectoryPath + fileName + "3.png")  # Save here
        plt.close()

    def GenearateHistorgram(self,DirectoryPath,itemList,position,fileName,lastsec):
        ##General over all
        plt.ioff()
        kwargs = dict(alpha=0.5, bins=100, density=True, stacked=True)
        plt.clf()
        fig = plt.figure(figsize=(16, 12))
        plt.rc('font', size=18)
        plt.tight_layout()
        if(lastsec):
            for item in itemList:
                plt.hist(item.DifferenceValues, **kwargs,  label=item.label)#color=item.color,
        else:
            for item in itemList:
                plt.hist(item.DifferenceValuesLastWindow, **kwargs, label=item.label)
            # plt.hist(x3, **kwargs, color='red', label='Sensor3')
            # plt.hist(x2, **kwargs, color='green', label='Wellue')
            # plt.hist(x4, **kwargs, color='purple', label='Watch')
        plt.gca().set(title='HR difference among different commercial pulse oximeter devices', ylabel='Frequency',
                      xlabel='HR (BPM) Difference (Clinical - Commercial)')
        # plt.legend()
        plt.savefig(DirectoryPath + fileName + ".png")  # Save here
        plt.close()
        plt.clf()
        plt.close()

    def GenerateResultbyAlgorithmType(self,objPlots,position,Actual_data, observed_data,Algorithm,SavePath,PIno,generateChart,fName):

        mean_squared_error_result = np.sqrt(mean_squared_error(Actual_data, observed_data))
        # print(position + " HR "+ Algorithm + " RMSE:", mean_squared_error_result)
        r, p = scipy.stats.pearsonr(Actual_data, observed_data)  ##Final
        # print(Algorithm +' HR '+position + ', r : ' + str(r))
        # print(Algorithm +' HR  '+position + ', p : ' + str(float(p)))
        # print("{:f}".format(float(p)))
        # print(np.corrcoef(Actual_data, observed_data)[0, 1])

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Actual_data, observed_data)
        rsqrd= r_value ** 2

        if (generateChart):
            objPlots.GenerateObservedvsActual(Actual_data,
                                              observed_data, SavePath, fName,r)
        return mean_squared_error_result, r, p

    def GenerateValuesOnlyStats(self, actualName,observedName,diffName,algorithmDataframe,dataTypeName):
        Actual_data = algorithmDataframe[actualName].tolist()
        observed_data = algorithmDataframe[observedName].tolist()
        diffValues = algorithmDataframe[diffName].tolist()

        rms = np.sqrt(mean_squared_error(Actual_data, observed_data)) #mean_squared_error_result
        # rms not squared= mean_squared_error(Actual_data, observed_data, squared=False)

        r, p = scipy.stats.pearsonr(Actual_data, observed_data)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Actual_data, observed_data)
        rsqrd= r_value ** 2
        meanABS_ = np.mean(abs(np.array(diffValues)))
        meanWithoutABS_ = np.mean(np.array(diffValues))
        print(dataTypeName+":")
        print("r= "+ str(r))
        print("p= "+ str(p))
        print("meanABS_= "+ str(meanABS_))
        print("meanWithoutABS= "+ str(meanWithoutABS_))
        print("slope= "+ str(slope))
        print("intercept= "+ str(intercept))
        print("std_err= "+ str(std_err))
        # print("mean_squared_error_result= "+ str(mean_squared_error_result))
        print("rms= "+ str(rms))
        print("rsqrd= "+ str(rsqrd))
        print()

        return r, p, meanABS_, meanWithoutABS_, slope, intercept,std_err,rsqrd,rms

    def LoadComputedResults(self):
        try:
            restPath = "SaveResultstoDiskDataFiles\\PIS-1032\\Resting1\\"
            for fileName in os.listdir(self.objConfig.DiskPath + restPath):
                if ((not fileName.__contains__("ProcessedCompleted"))):
                    if ((not fileName.__contains__("Graphs"))):
                        filePath = os.path.join(self.objConfig.DiskPath + restPath, fileName)
                        computerData = pd.read_csv(filePath)

                        computerData = computerData[(computerData.HRDifference < 3) & (
                                    computerData.HRDifference > -3)]  # (computerData.ComputedHeartRate >74|computerData.bestBpmWithoutReliabilityCheck >74)

                        # computerData = computerData[(computerData.HRDifference <6) | (computerData.bestBpmWithoutReliabilityCheckDifference <6)
                        # & computerData[(computerData.bestSnrString>=2.8)]]# (computerData.ComputedHeartRate >74|computerData.bestBpmWithoutReliabilityCheck >74)

                        if (not computerData.empty):
                            print(filePath)
        except:
            print("Error " + filePath)

class DataTypeObject:
    datapoints = []
    DifferenceValues= []
    DifferenceValuesLastWindow= []
    color=""
    label=""
