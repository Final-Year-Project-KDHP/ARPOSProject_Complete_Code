import glob
import math
import os
from datetime import datetime
import pyCompare

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from numpy import linspace

from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import scipy.stats
from scipy.stats import chisquare
import scipy.stats as stats
import CommonMethods
from Configurations import Configurations
from FileIO import FileIO
from SQLResults.SQLConfig import SQLConfig
from SaveGraphs import Plots
from boxPlotMethodComparision import BoxPlot
import plotly.express as px


class GeneratedDataFiltering:
    objConfig = None
    objFile = None
    AcceptableDifference = 3

    # Constructor
    def __init__(self, skinGroup='None'):
        self.objConfig = Configurations()
        self.objFile = FileIO()

    def Getdata(self, fileName, AcceptableDifference, participant_number, position):  # "None", loadpath,methodtype
        filepath = self.objConfig.DiskPath + 'Result\\' + participant_number + '\\' + position + '\\' + fileName + '\\' + 'HeartRate_' + fileName + '.txt'  # HeartRate_FastICA_FFT-M1_FL-6_RS-1_PR-1_SM-False
        # + algotype + '\\' + filename + "-"+ methodtype+ "_Fl_"+ str(filtertype) + "_Rs_" + str(resulttype)  + "_Pr_" + str(processtype)+ "_Sm_" + str(isSmooth)+ ".txt" #HRdata-M1_1_1
        # HRdata-M1_Fl_1_Rs_1_Pr_1_Sm_False

        Filedata = open(filepath, "r")
        data = Filedata.read().split("\n")
        generatedresult = []
        isAcceptableData = False
        diffValueList = []
        AcceptabledifferenceValue = -20
        for row in data:
            dataRow = row.split(",\t")
            # 0 index for windowCount, 1 index for GroundTruth, 2 index for generatedresult by arpos, 3 index for diference
            differenceValue = int(dataRow[2])  # change index here
            #  if 5 <= 7 and 5 >= -7:
            negativeAcceptableDifference = -1 * (AcceptableDifference)
            if (
                    differenceValue <= AcceptableDifference and differenceValue >= negativeAcceptableDifference):  # ((differenceValue >= AcceptableDifference) ): #or (differenceValue <= negativeAcceptableDifference )
                isAcceptableData = True
                AcceptabledifferenceValue = differenceValue
            else:
                isAcceptableData = False
                break

            # isAcceptableData =True

        if (isAcceptableData):
            # filename = Filedata.name.split("HRdata-")
            generatedresult.append(fileName)
            diffValueList.append(AcceptabledifferenceValue)

        return generatedresult, isAcceptableData, diffValueList

    def CompareFiles(self, ListFilesP1, ListFilesP2):  # participantnumber1, participantnumber2,
        # full_path_Save_P1 = "E:\\StudyData\\Result\\" + participantnumber1 + "\\Resting1\\BestDataFiles\\"
        # full_path_Save_P2 = "E:\\StudyData\\Result\\" + participantnumber2 + "\\Resting1\\BestDataFiles\\"

        # HRdata-M1_Fl_1_Rs_1_Pr_1_Sm_False
        generatedresult = []
        for row in ListFilesP1:
            Row1_P1_Value = row
            for row2 in ListFilesP2:
                if (Row1_P1_Value == row2):
                    generatedresult.append(row)

        # using list comprehension
        # to remove duplicated
        # from list
        result = []
        [result.append(x) for x in generatedresult if x not in result]

        for value in result:
            print(value)

    def processBestResults(self, filepath, filename):
        for position in self.objConfig.hearratestatus:
            # dataFile = filepath + filename
            dataFile = filepath + filename + "_" + position + ".txt"
            # SPOFile = filepath + "SPO.txt"

            # read data from files
            HrFiledata = open(dataFile, "r")
            # SPOFiledata = open(SPOFile, "r")

            HrFileNames = HrFiledata.read().split("\n")
            # SpoGr = SPOFiledata.read().split("\n")
            HrFiledata.close()

            _FastICA = []
            _None = []
            _PCA = []
            _ICAPCA = []
            _Jade = []

            for item in HrFileNames:
                if (item.__contains__("FastICA_")):
                    Filename = item.replace("FastICA_", "")
                    _FastICA.append(Filename)  # FastICA_ # M1_Fl_1_Rs_1_Pr_3_Sm_True.txt
                elif (item.__contains__("None")):
                    Filename = item.replace("None_", "")
                    _None.append(Filename)  # FastICA_ # M1_Fl_1_Rs_1_Pr_3_Sm_True.txt
                elif (item.__contains__("ICAPCA_")):
                    Filename = item.replace("ICAPCA_", "")
                    _ICAPCA.append(Filename)  # FastICA_ # M1_Fl_1_Rs_1_Pr_3_Sm_True.txt
                elif (item.__contains__("PCA_")):  # MATCHES WITH ICA PCA So check in end
                    Filename = item.replace("PCA_", "")
                    _PCA.append(Filename)  # FastICA_ # M1_Fl_1_Rs_1_Pr_3_Sm_True.txt
                elif (item.__contains__("Jade_")):
                    Filename = item.replace("Jade_", "")
                    _Jade.append(Filename)

            # Getcommon filenames only
            result = []
            [result.append(x) for x in _None if x not in result]
            [result.append(x) for x in _FastICA if x not in result]
            [result.append(x) for x in _ICAPCA if x not in result]
            [result.append(x) for x in _PCA if x not in result]

            result.sort()
            full_path_Save = self.objConfig.DiskPath + 'Result\\'
            # Wirte data
            RHr = open(full_path_Save + "BestCommonFiles" + "_" + position + ".txt", "w+")

            for item in result:
                RHr.write(item + "\n")

            RHr.close()

    def Run(self, AcceptableDifference):
        fullistResting1 = []
        fullistResting2 = []
        fullistAfterExcersize = []
        fullistResting1Values = []
        fullistResting2Values = []
        fullistAfterExcersizeValues = []

        Resting1Store = {}
        Resting2Store = {}
        AfterExcersizeStore = {}
        # self.objConfig.ParticipantNumbers = ['PIS-8073' , 'PIS-2047', 'PIS-4014', 'PIS-1949', 'PIS-3186', 'PIS-7381']

        for participant_number in self.objConfig.ParticipantNumbers:
            Resting1List = []
            Resting2List = []
            AfterExcList = []
            Resting1ListValues = []
            Resting2ListValues = []
            AfterExcListValues = []

            for position in self.objConfig.hearratestatus:
                loadpath = self.objConfig.DiskPath + 'Result\\' + participant_number + '\\' + position + '\\'
                datalist = []
                datalistValues = []

                for preprocesstype in self.objConfig.preprocesses:
                    for algoType in self.objConfig.AlgoList:
                        for fftype in self.objConfig.fftTypeList:
                            for resulttype in self.objConfig.resulttypeList:
                                for filtertype in self.objConfig.filtertypeList:
                                    for isSmooth in self.objConfig.Smoothen:
                                        fileName = algoType + "_FFT-" + str(fftype) + "_FL-" + str(
                                            filtertype) + "_RS-" + str(resulttype) + "_PR-" + str(
                                            preprocesstype) + "_SM-" + str(isSmooth)
                                        print(fileName)
                                        generatedresult, isAcceptableData, diffValueList = self.Getdata(fileName,
                                                                                                        AcceptableDifference,
                                                                                                        participant_number,
                                                                                                        position)
                                        # generatedresult,isAcceptableData =self.Getdata(algoType, loadpath,fftype,"HRdata",filtertype,resulttype,AcceptableDifference,preprocesstype,isSmooth )
                                        # filepath = loadpath  + algotype + '\\' + filename + "-"+ methodtype+ "_"+ filtertype + "_" + resulttype+ ".txt" #HRdata-M1_1_1
                                        if (isAcceptableData):
                                            datalist.append(generatedresult)
                                            datalistValues.append(diffValueList)

                # datalist = np.array(datalist)
                if (position == "Resting1"):
                    Resting1List = datalist
                    Resting1ListValues = datalistValues
                elif (position == "Resting2"):
                    Resting2List = datalist
                    Resting2ListValues = datalistValues
                else:
                    AfterExcList = datalist
                    AfterExcListValues = datalistValues

            fullistResting1 = fullistResting1 + Resting1List
            fullistResting2 = fullistResting2 + Resting2List
            fullistAfterExcersize = fullistAfterExcersize + AfterExcList

            fullistResting1Values = fullistResting1Values + Resting1ListValues
            fullistResting2Values = fullistResting2Values + Resting2ListValues
            fullistAfterExcersizeValues = fullistAfterExcersizeValues + AfterExcListValues
            # Resting1Store[participant_number] =Resting1List
            # Resting2Store[participant_number] =Resting2List
            # AfterExcersizeStore[participant_number] =AfterExcList

            # if (piIndex == 0):
            #     fullistResting1 = Resting1List
            #     fullistResting2 = Resting2List
            #     fullistAfterExcersize = AfterExcList
            # else:
            #     fullistResting1 = np.intersect1d(fullistResting1, Resting1List)
            #     fullistResting2 = np.intersect1d(fullistResting2, Resting2List)
            #     fullistAfterExcersize = np.intersect1d(fullistAfterExcersize, AfterExcList)
            # piIndex = piIndex + 1

        # fullistResting1 = list(Resting1Store.values())
        Resting1result = []
        Resting1resultValues = []
        index = 0
        for item in fullistResting1:
            if item not in Resting1result:
                Resting1result.append(item)
                ItemValue = fullistResting1Values[index]
                Resting1resultValues.append(ItemValue)
            index = index + 1

        # [Resting1result.append(x) for x in fullistResting1 if x not in Resting1result]

        Resting2result = []
        Resting2resultValues = []
        # [Resting2result.append(x) for x in fullistResting2 if x not in Resting2result]
        index = 0
        for item in fullistResting2:
            if item not in Resting2result:
                Resting2result.append(item)
                ItemValue = fullistResting2Values[index]
                Resting2resultValues.append(ItemValue)
            index = index + 1

        AfterExcersizeresult = []
        AfterExcersizeresultValues = []
        index = 0
        for item in fullistAfterExcersize:
            if item not in AfterExcersizeresult:
                AfterExcersizeresult.append(item)
                ItemValue = fullistAfterExcersizeValues[index]
                AfterExcersizeresultValues.append(ItemValue)
            index = index + 1
        # [AfterExcersizeresult.append(x) for x in fullistAfterExcersize if x not in AfterExcersizeresult]
        #
        # for x in result:
        #     print(x)

        # fullistResting2 = list(Resting2Store.values())
        # result = []
        # [result.append(x) for x in fullistResting2 if x not in result]
        #
        #
        # fullistAfterExcersize = list(AfterExcersizeStore.values())
        # result = []
        # [result.append(x) for x in fullistAfterExcersize if x not in result]

        full_path_Save = self.objConfig.DiskPath
        # Wirte data
        RHr = open(full_path_Save + 'Result\\' + "BestDataFiles_Resting1.txt", "w+")
        RHr2 = open(full_path_Save + 'Result\\' + "BestDataFiles_Resting2.txt", "w+")
        RHr3 = open(full_path_Save + 'Result\\' + "BestDataFiles_AfterExcersize.txt", "w+")

        for item in Resting1result:
            RHr.write(item[0] + "\n")

        for item in Resting2result:
            RHr2.write(item[0] + "\n")

        for item in AfterExcersizeresult:
            RHr3.write(item[0] + "\n")

        # for item in fullistResting2:
        #     RHr2.write(item[0]  + "\n")
        #
        # for item in fullistAfterExcersize:
        #     RHr3.write(item[0]  + "\n")

        RHr.close()
        RHr2.close()
        RHr3.close()

        # objboxplot = BoxPlot('Europe_WhiteSkin_Group')
        # objboxplot.RunBoxplotforCommonResultsFromList("Resting1", True, Resting1result,True)  # Set True to generate differnce box plot
        # for x in fullist:
        #     print(x)
        # CompareFiles(P1List,P2List) Manual comparison between two participants data list containing filenames showing data within acceptablediffernce
        # finallist = set(np.array(P1List)) & set(np.array(P2List)) #& set(c)
        # P1List =np.array(P1List)
        # P2List =np.array(P2List)
        # fullist = set(P1List).intersection(P2List)
        # fullist = np.intersect1d(P1List, P2List)

        # set(a).intersection(b, c)

    """
    Gemerate cases:
    """
    def GenerateCases2(self):
        CaseList = []
        for preprocesstype in self.objConfig.preprocesses:
            for algoType in self.objConfig.AlgoList:
                for fftype in self.objConfig.fftTypeList:
                    for resulttype in self.objConfig.resulttypeList:
                        for filtertype in self.objConfig.filtertypeList:
                            for isSmooth in self.objConfig.Smoothen:
                                fileName = "HRSPOwithLog_HRSPOwithLog_" + str(algoType) + "_PR-" + str(preprocesstype) + \
                                           "_FFT-" + str(fftype) + "_FL-" + str(filtertype) + "_RS-" + str(resulttype) \
                                           + "_SM-" + str(isSmooth)
                                CaseList.append(fileName)
        return CaseList

    """
       Gemerate cases:
       """
    def GenerateCasesNewMethod(self, region):
        CaseList = []
        for preprocesstype in self.objConfig.preprocesses:
            for algoType in self.objConfig.AlgoList:
                for fftype in self.objConfig.fftTypeList:
                    for resulttype in self.objConfig.resulttypeList:
                        for filtertype in self.objConfig.filtertypeList:
                            for isSmooth in self.objConfig.Smoothen:
                                fileName = "ResultType_RS-" + str(resulttype) + "_Filtered_FL-" + str(
                                    filtertype) + "_" + region + "_FFTtype-" + str(fftype) + "_algotype-" + str(
                                    algoType) + \
                                           '_PreProcessType-' + str(preprocesstype) + "_Smoothed-" + str(isSmooth)
                                print(fileName)
                                CaseList.append(fileName)

        return CaseList

    def getCase(self, fileName):
        SavePath = self.objConfig.SavePath +'ComputedFinalResult\\'
        filepath = SavePath + fileName + '.txt'  # HeartRate_FastICA_FFT-M1_FL-6_RS-1_PR-1_SM-False

        pathExsists = self.objFile.FileExits(filepath)
        data = None
        # already generated
        if (pathExsists):
            Filedata = open(filepath, "r")
            data = Filedata.read().split("\n")[0]
            Filedata.close()
        return data

    def getCasebyPath(self, SavePath,fileName):
        filepath = SavePath + fileName + '.txt'  # HeartRate_FastICA_FFT-M1_FL-6_RS-1_PR-1_SM-False

        pathExsists = self.objFile.FileExits(filepath)
        data = None
        # already generated
        if (pathExsists):
            Filedata = open(filepath, "r")
            data = Filedata.read().split("\n")[0]
            Filedata.close()
        return data

    def getCaseNew(self, fileName, participant_number, position):
        SavePath = self.objConfig.DiskPath + '\\ProcessedData\\' + participant_number + '\\' + position + '\\FinalComputedResult\\'  # + fileName + '\\'
        filepath = SavePath + fileName + '.txt'  # HeartRate_FastICA_FFT-M1_FL-6_RS-1_PR-1_SM-False 'HeartRate_' +

        pathExsists = self.objFile.FileExits(filepath)  # + ".txt"
        data = None
        # already generated
        if (pathExsists):
            Filedata = open(filepath, "r")
            data = Filedata.read().split("\n")[0]
            Filedata.close()
        return data

    def IsAcceptableDifference(self, differenceValue):  # "None", loadpath,methodtype
        #  if 5 <= 7 and 5 >= -7:
        negativeAcceptableDifference = -1 * (self.AcceptableDifference)
        if (differenceValue <= self.AcceptableDifference and differenceValue >= negativeAcceptableDifference):
            return True
        else:
            return False

    def splitDataRow(self, DataRow, participant_number, position):  # "None", loadpath,methodtype
        dataRow = DataRow.split(",\t")
        # 0 index for GroundTruth, 1 index for generatedresult by arpos, 2 index for diference
        generatedResult = int(dataRow[2].replace('ComputedHeartRate: ',''))  # change index here
        HrGr = int(dataRow[1].replace('GroundTruthHeartRate: ','')) #CommonMethods.GetGroundTruth(participant_number, position, self.objConfig.DiskPath, int(60))
        # SpoGr= int(dataRow[4].replace('ComputedHeartRate: ',''))
        # avgGroundTruthStored = int(dataRow[0])
        avgGroundTruth = HrGr#np.mean(HrGr)
        # if(avgGroundTruthStored != round(avgGroundTruth)):
        #     differenceValue=int(dataRow[2])
        # else:
        diffval =dataRow[3].replace('HRDifference: ','')
        diffval = float((diffval))
        # if(diffval<0):
        #     diffval = -1*(diffval)
        differenceValue =  diffval#avgGroundTruth - generatedResult
        errorrate = (differenceValue / avgGroundTruth) * 100
        return errorrate #

    def RunCasewise(self,position,useAcceptableDiff):
        CaseList, LoadfNameList = self.GenerateCases()  # GET Directories or generate below
        # CaseList = self.GenerateCases() # or generate

        df1 = pd.DataFrame({
            'CaseNames': CaseList
        })

        for participant_number in self.objConfig.ParticipantNumbers:
            df1[participant_number] = None
        # print(df1)
        # for position in self.objConfig.hearratestatus:
        RowIndex = 0  #
        for case in CaseList:
            ColumnIndex = 1  # so doesntt replace case names
            for participant_number in self.objConfig.ParticipantNumbers:
                # for position in self.objConfig.hearratestatus:
                self.objConfig.setSavePath(participant_number,position,'ProcessedDatabyProcessType')
                CaseData = self.getCase(case)
                if (CaseData != None):
                    if(CaseData == ''):
                        df1.iloc[RowIndex, ColumnIndex] = 'NoHR' #nmeans does not work for this case for this person
                        #needs to be checked if not same for all participants
                    else:
                        differenceVal = self.splitDataRow(CaseData, participant_number, position)  # ROW,COlum
                        if(useAcceptableDiff):
                            isAcceptable = self.IsAcceptableDifference(differenceVal)
                            if (isAcceptable):
                                df1.iloc[RowIndex, ColumnIndex] = differenceVal
                        else:
                            df1.iloc[RowIndex, ColumnIndex] = differenceVal
                else:
                    df1.iloc[RowIndex, ColumnIndex] = 'NotGenerated'
                # else:
                #     df1.iloc[RowIndex, ColumnIndex] = None
                ColumnIndex = ColumnIndex + 1
                # print(df1)
            RowIndex = RowIndex + 1

        # write dataFrame to SalesRecords CSV file
        df1.to_csv(self.objConfig.DiskPath + "CaseWiseParticipantsResults_" + position + "__DiffOf" + str(self.AcceptableDifference) + ".csv")
        t = 0

    def getBestCasesFromCSV(self, position):
        df1 = pd.read_csv(self.objConfig.DiskPath + "CaseWiseParticipantsResults_" + position + "__DiffOf" + str(self.AcceptableDifference) + ".csv")
        NoHRlist =[]
        ListCaseCount = {}
        for index, row in df1.iterrows():
            currentCase = row[1]
            addCase = True
            mincount = 0

            for column in df1:
                if (column.__contains__('Unnamed') or column.__contains__('Case')):
                    continue
                colIndex = df1.columns.get_loc(column)  # df1[column] just column in df1 get entire columns values
                colValueInRow = df1.iloc[index, colIndex]
                if(colValueInRow == 'NoHR'):
                    NoHRlist.append(currentCase)
                else:
                    colValueInRow = float(colValueInRow)
                    if (not math.isnan(colValueInRow)):
                        mincount = mincount + 1

            ListCaseCount[currentCase] = mincount

        MaxCount=0
        ListCasesForSelection = []
        for key, value in ListCaseCount.items():
            if (value > MaxCount):
                MaxCount = value
                # ListCasesForSelection.append(key)


        for key, value in ListCaseCount.items():
            if(value >= MaxCount):
                ListCasesForSelection.append(key)
                # ListCasesForSelection.remove(key)

        # MaxCount = self.getDuplicateValue(ListCaseCount)
        NoHRlist = list(dict.fromkeys(NoHRlist))
        print('')
        #Rerun for cases:
        for item in NoHRlist:
            print(item)

        print('')
        print('Best Cases')
        ListCasesForSelection = list(dict.fromkeys(ListCasesForSelection))
        for item in ListCasesForSelection:
            if(item.__contains__('FastICAComponents3')):
                ListCasesForSelection.remove(item)

        for item in ListCasesForSelection:
            print(item)

        ###ReGeneraete sheet as
        # totalLengofCaes = len(ListCasesForSelection)
        # totalCharts = 8 # int(round(totalLengofCaes/7))
        # TotalListsOfCases = []
        # initialindex=0
        # list1 = ListCasesForSelection[initialindex:totalCharts]
        # initialindex=totalCharts
        # totalCharts = totalCharts +8
        # list2 = ListCasesForSelection[initialindex:totalCharts]
        # initialindex=totalCharts
        # totalCharts = totalCharts +8
        # list3 = ListCasesForSelection[initialindex:totalCharts]
        # initialindex=totalCharts
        # totalCharts = totalCharts +8
        # list4 = ListCasesForSelection[initialindex:totalCharts]
        # initialindex=totalCharts
        # totalCharts = totalCharts +8
        # list5 = ListCasesForSelection[initialindex:totalCharts]
        # initialindex=totalCharts
        # totalCharts = totalCharts +8
        # list6 = ListCasesForSelection[initialindex:totalCharts]
        # list7 = ListCasesForSelection
        n=8
        for x in range(0, len(ListCasesForSelection),n):
            list1 = ListCasesForSelection[x:x+n]
            self.CreateCSVdataFromList(list1,position,x,df1)
            a=0
        # self.CreateCSVdataFromList(list2,position,2,df1)
        # self.CreateCSVdataFromList(list3,position,3,df1)
        # self.CreateCSVdataFromList(list4,position,4,df1)
        # self.CreateCSVdataFromList(list5,position,5,df1)
        # self.CreateCSVdataFromList(list6,position,6,df1)
        # self.CreateCSVdataFromList(list7,position,7,df1)


        t = 0
        # BestCaseList = []
        # for index, row in df1.iterrows():
        #     currentCase = row[1]
        #     addCase= True
        #     for column in df1:
        #         if(column.__contains__('Unnamed') or column.__contains__('Case')):
        #             continue
        #         colIndex = df1.columns.get_loc(column) #df1[column] just column in df1 get entire columns values
        #         colValueInRow =  df1.iloc[index, colIndex]
        #         if(np.isnan(colValueInRow)):
        #             addCase = False
        #
        #     if(addCase):
        #         BestCaseList.append(currentCase)
        #
        # for item in BestCaseList:
        #     print(item)

    def getfromDataFrameCSV(self, position,GroupWise):
        df1 = pd.read_csv(self.objConfig.DiskPath + "CaseWiseFromFolder_Results_" + position + "_All.csv")
        CaseList, LoadfNameList = self.GenerateCases()
        x=0
        SelectedCases = []
        count=0
        for case in CaseList:
            print(count)
            if(GroupWise):
                if(x<9):
                    SelectedCases.append(case)
                else:
                    GroupcaseData =df1[df1['Techniques'].isin(SelectedCases)]
                    self.MakeBoxPlotbyInput(GroupcaseData, "caseGroup_" + str(count) , "CaseGroupWise")
                    x = 0
                    SelectedCases = []
                x= x+1
            else:
                caseData = df1.where(df1.Techniques == case)
                Gnereate=False
                for item in caseData.iterrows():
                    diff =item[1]['Differences']
                    if(diff<100):
                        Gnereate=True
                    else:
                        Gnereate=False
                        break
                if(Gnereate):
                    self.MakeBoxPlotbyInput(caseData, "case", "CaseDiffWise_Under" + str(15))
            count = count + 1

            # caseData.to_csv(
            #     self.objConfig.DiskPath + "BoxPlotCSV\\BoxPlotImages\\CaseGroupWise\\" + case + ".csv")

    def CustomCaseListMain(self):
        # CustomCases = self.objFile.ReaddatafromFile(self.objConfig.DiskPath,'NoHrFilesCases')
        CustomCases = []

        CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M4_FL-3_RS-1_SM-False')
        CustomCases.append('HRSPOwithLog_FastICA_PR-3_FFT-M3_FL-7_RS-1_SM-False')
        CustomCases.append('HRSPOwithLog_FastICA_PR-3_FFT-M4_FL-3_RS-1_SM-False')
        CustomCases.append('HRSPOwithLog_FastICA_PR-3_FFT-M4_FL-7_RS-1_SM-True')
        CustomCases.append('HRSPOwithLog_FastICA_PR-4_FFT-M4_FL-7_RS-1_SM-False')
        CustomCases.append('HRSPOwithLog_FastICA_PR-4_FFT-M4_FL-7_RS-1_SM-True')
        CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M3_FL-7_RS-1_SM-True')
        CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M4_FL-7_RS-1_SM-True')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-1_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-1_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-6_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-6_RS-2_SM-False')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M1_FL-1_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M1_FL-1_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M1_FL-6_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M1_FL-6_RS-2_SM-False')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M1_FL-1_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M1_FL-2_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M1_FL-6_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M1_FL-6_RS-2_SM-False')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-3_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-3_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-5_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-5_RS-2_SM-False')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M2_FL-3_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M2_FL-3_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M2_FL-5_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M2_FL-5_RS-2_SM-False')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M2_FL-3_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M2_FL-3_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M2_FL-5_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M2_FL-5_RS-2_SM-False')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-3_FFT-M2_FL-3_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-3_FFT-M2_FL-3_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-3_FFT-M2_FL-5_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-3_FFT-M2_FL-5_RS-2_SM-False')
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-4_FFT-M2_FL-3_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-4_FFT-M2_FL-3_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-4_FFT-M2_FL-5_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-4_FFT-M2_FL-5_RS-2_SM-False')
        #
        #
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M2_FL-3_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M2_FL-3_RS-2_SM-False')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M2_FL-5_RS-2_SM-True')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-5_FFT-M2_FL-5_RS-2_SM-False')
        # FOR White skin
        ###second with rs=2 only
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-2_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M2_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M4_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M1_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M1_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M1_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M2_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M2_FL-2_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M2_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M2_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M4_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCAICA_PR-6_FFT-M5_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M1_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M1_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M1_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M2_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M2_FL-2_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M2_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M2_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M4_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-6_FFT-M5_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-2_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_PCA_PR-6_FFT-M2_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M1_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M1_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M1_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M2_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M2_FL-2_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M2_FL-3_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M2_FL-5_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_None_PR-6_FFT-M4_FL-1_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-7_FFT-M1_FL-6_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICA_PR-7_FFT-M5_FL-6_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-7_FFT-M1_FL-6_RS-2_SM-TRUE')
        # CustomCases.append('HRSPOwithLog_FastICAComponents3Times_PR-7_FFT-M5_FL-6_RS-2_SM-TRUE')
        ################First attempt below

        self.CaseList = []
        for case in CustomCases:
            case = case.replace('\n','')
            caseSplit = case.split('_')
            algoType = caseSplit[1]
            preprocesstype = caseSplit[2].replace('PR-','')
            fftype = caseSplit[3].replace('FFT-','')
            filtertype = caseSplit[4].replace('FL-','')
            resulttype = caseSplit[5].replace('RS-','')
            isSmooth = caseSplit[6].replace('SM-','')
            fileName = algoType + "_PR-" + str(preprocesstype) + "_FFT-" + str(
                fftype) + "_FL-" + str(filtertype) \
                       + "_RS-" + str(resulttype) + "_SM-" + str(isSmooth)
            self.CaseList.append(fileName)

    def PlotfromCustomCases(self,  GroupWise):
        fileName =  "CaseWiseFromFolder_Results_Resting1_CustomCases2_UpSampleData-False"
        df1 = pd.read_csv(self.objConfig.DiskPath +fileName + ".csv")
        df1.head()
        self.CustomCaseListMain()
        x = 0
        SelectedCases = []
        count = 0
        for case in self.CaseList:
            print(count)
            if (GroupWise):
                if (x < 9):
                    SelectedCases.append(case)
                else:
                    GroupcaseData = df1[df1['Techniques'].isin(SelectedCases)]
                    self.MakeBoxPlotbyInput(GroupcaseData, "caseGroup_" + str(count),fileName+ ".csv")
                    x = 0
                    SelectedCases = []
                x = x + 1
            else:
                caseData = df1.where(df1.Techniques == case)
                caseData = caseData.dropna()
                Gnereate = False
                for item in caseData.iterrows():
                    diff = np.abs(item[1]['Differences'])
                    if (diff < 100):
                        Gnereate = True
                    else:
                        Gnereate = False
                        break
                if (Gnereate):
                    self.MakeBoxPlotbyInput(caseData, "case", fileName+'_Individual' ,fileName+ '_' + str(count))
            count = count + 1

    def PlotFromSQLQuery(self,objSQLConfig):
        #Param
        skinGroup='SouthAsian_BrownSkin_Group'
        PreProcess = '6'
        FFT='M4'
        Filter='1'
        Result='2'
        Smoothen='True'
        sqlResultData = objSQLConfig.ReadDataTableByTechniqueParameters(skinGroup,PreProcess,FFT,Filter,Result,Smoothen)

        #by algo
        fullPath = "E:\\ARPOS_Server_Data\\Server_Study_Data\\Plots\\AllParticipants\\"
        self.objFile.CreatePath(fullPath)
        fileName="boxPlot_OverallParticipants"
        self.MakeBoxPlotfromSQLData(sqlResultData,fullPath,fileName,'AlgorithmType','OriginalDifference',"Box Plot for over all participants")

        dataGroupWise = objSQLConfig.getTableQueryGroupWiseDifferences(skinGroup,PreProcess,FFT,Filter,Result,Smoothen)
        fileName="boxPlot_OverallParticipants_GroupWise"
        self.MakeBoxPlotGroupWisefromSQLData(dataGroupWise, fullPath,fileName,'differenceType','DifferenceHR',"AlgorithmType")

        Position = 'Resting1'
        fileName="barPlot_OverallParticipants"
        dataTimetoExecute = objSQLConfig.getTableQueryTimetoRun(skinGroup,PreProcess,FFT,Filter,Result,Smoothen,Position)
        # dataTimetoExecute = dataTimetoExecute.explode('totalTimeinMilliSeconds')
        # dataTimetoExecute['totalTimeinMilliSeconds'] = dataTimetoExecute['totalTimeinMilliSeconds'].astype('totalTimeinMilliSeconds')
        self.MakeBarTimePlotfromSQLData(dataTimetoExecute,fullPath,fileName)#TODO: VERIFY DATA

    def PlotFromSQLQueryFinal(self,objSQLConfig,skinGroup,type,TechniqueId,HeartRateStatus,UpSampled):
        TechniqueDetail = objSQLConfig.getTechniqueDetailFromId(TechniqueId)
        PreProcess=TechniqueDetail['PreProcess'][0]
        FFT=TechniqueDetail['FFT'][0]
        Filter=TechniqueDetail['Filter'][0]
        Result=TechniqueDetail['Result'][0]
        Smoothen=TechniqueDetail['Smoothen'][0]
        sqlResultData = objSQLConfig.ReadDataTableByTechniqueAndHeartRateStatus(skinGroup,str(PreProcess),FFT,str(Filter),str(Result),str(Smoothen),HeartRateStatus,str(UpSampled))
        #by algo
        folderName=''
        if(UpSampled == '1'):
            folderName = 'UpSampled'
        else:
            folderName = 'NOTUpSampled'
        fullPath = "E:\\ARPOS_Server_Data\\Server_Study_Data\\Plots\\" + skinGroup + '\\'+folderName+'\\'
        self.objFile.CreatePath(fullPath)
        fileName="boxPlot_" +skinGroup + "_" +HeartRateStatus +"_"+type + '_' + str(TechniqueId)
        self.MakeBoxPlotfromSQLData(sqlResultData,fullPath,fileName,'AlgorithmType',type,fileName)
        a=0

    def PlotFromSQLQuerySouthAsian(self,objSQLConfig,skinGroup,type,TechniqueId,HeartRateStatus,UpSampled,AttemptType):
        sqlResultData = objSQLConfig.GetBestSouthAsianCases(skinGroup, HeartRateStatus, str(UpSampled), str(AttemptType), TechniqueId)
        #by algo
        folderName=''
        if(UpSampled == '1'):
            folderName = 'UpSampled'
        else:
            folderName = 'NOTUpSampled'
        fullPath = "E:\\ARPOS_Server_Data\\Server_Study_Data\\Plots\\" + skinGroup + '\\'+folderName+'\\'+ HeartRateStatus+'\\'
        self.objFile.CreatePath(fullPath)
        fileName="boxPlot_" +skinGroup + "_" +HeartRateStatus +"_"+type + '_' + str(TechniqueId)
        self.MakeBoxPlotfromSQLData(sqlResultData,fullPath,fileName,'AlgorithmType',type,fileName)
        a=0

    def PlotFromSQLQueryAll(self,objSQLConfig,type,TechniqueId,HeartRateStatus,UpSampled,AttemptType):
        sqlResultData = objSQLConfig.GetBestAmongAll(HeartRateStatus, str(UpSampled), str(AttemptType), TechniqueId)
        lst = [0] * len(sqlResultData['HeartRateValue'].tolist())
        mean_squared_error_result = np.sqrt(mean_squared_error(lst, (sqlResultData['HeartRateDifference'].tolist())))#sqlResultData['GroundTruthHeartRate'].tolist()
        # print(position + " HR "+ Algorithm + " RMSE:", mean_squared_error_result)#HeartRateValue
        r, p = scipy.stats.pearsonr(lst, (sqlResultData['HeartRateDifference'].tolist()))  ##Final
        print(str(mean_squared_error_result))
        print(str(r))
        print(str(p))
        a=0
        #
        # #by algo
        # folderName=''
        # if(UpSampled == '1'):
        #     folderName = 'UpSampled'
        # else:
        #     folderName = 'NOTUpSampled'
        # fullPath = "E:\\ARPOS_Server_Data\\Server_Study_Data\\Plots\\All\\"+folderName+"\\"+ HeartRateStatus+"\\"
        # self.objFile.CreatePath(fullPath)
        # fileName="boxPlot_All_" +HeartRateStatus +"_"+type + '_' + str(TechniqueId)
        # self.MakeBoxPlotfromSQLData(sqlResultData,fullPath,fileName,'AlgorithmType',type,fileName)
        #
        # ####tIME PLOT
        # dataTimetoExecute = objSQLConfig.getTableQueryTimetoRunForAll(HeartRateStatus, str(UpSampled), str(AttemptType), TechniqueId)
        # fileName="BarplotTime_All_" +HeartRateStatus +"_" + str(TechniqueId)
        # self.MakeBarTimePlotfromSQLData(dataTimetoExecute, fullPath, fileName)

    def PlotFromSQLQueryGetUpSampledVSNotSampledDataSpecific(self,objSQLConfig,TechniqueId,HeartRateStatus,UpSampled,AttemptType):
        sqlResultData = objSQLConfig.GetUpSampledVSNotSampledDataSpecific(HeartRateStatus, str(UpSampled), str(AttemptType), TechniqueId)
        folderName='Comparison'
        fullPath = "E:\\ARPOS_Server_Data\\Server_Study_Data\\Plots\\All\\"+folderName+"\\"+ HeartRateStatus+"\\"
        self.objFile.CreatePath(fullPath)
        #For group
        fileName="boxPlot_All_" +HeartRateStatus +"_NotVSUpSampled_" + str(TechniqueId)
        self.MakeBoxPlotGroupWisefromSQLData(sqlResultData, fullPath,fileName,'SampleType','differenceHR',"AlgorithmType")
        #For idnivudal
        # fileName="boxPlot_All_" +HeartRateStatus +"_UpSampled_" + str(TechniqueId)
        # self.MakeBoxPlotfromSQLData(sqlResultData,fullPath,fileName,'AlgorithmType','differenceHR',fileName)
        #
        # fileName="boxPlot_All_" +HeartRateStatus +"_NOTSampled_" + str(TechniqueId)
        # self.MakeBoxPlotfromSQLData(sqlResultData,fullPath,fileName,'AlgorithmType','upSampledDiff',fileName)

    def appdendataTime(self,FinalDataResults,dtParticipant,InputValue):

        FinalDataResults = FinalDataResults.append({'AlgorithmType': 'FastICA', 'MiliTotaltime': dtParticipant["FastICA"], 'InputValue': InputValue},
                       ignore_index=True)
        FinalDataResults = FinalDataResults.append({'AlgorithmType': 'FastICA3Times', 'MiliTotaltime': dtParticipant["FastICA3Times"], 'InputValue': InputValue},
                       ignore_index=True)
        FinalDataResults = FinalDataResults.append({'AlgorithmType': 'FastICACombined', 'MiliTotaltime': dtParticipant["FastICACombined"], 'InputValue': InputValue},
                       ignore_index=True)
        FinalDataResults = FinalDataResults.append({'AlgorithmType': 'None', 'MiliTotaltime': dtParticipant["None"], 'InputValue': InputValue},
                       ignore_index=True)
        FinalDataResults = FinalDataResults.append({'AlgorithmType': 'PCA', 'MiliTotaltime': dtParticipant["PCA"], 'InputValue': InputValue},
                       ignore_index=True)
        FinalDataResults = FinalDataResults.append({'AlgorithmType': 'PCACombined', 'MiliTotaltime': dtParticipant["PCACombined"], 'InputValue': InputValue},
                       ignore_index=True)
        FinalDataResults = FinalDataResults.append({'AlgorithmType': 'PCAICA', 'MiliTotaltime': dtParticipant["PCAICA"], 'InputValue': InputValue},
                       ignore_index=True)
        FinalDataResults = FinalDataResults.append({'AlgorithmType': 'PCAICACombined', 'MiliTotaltime': dtParticipant["PCAICACombined"], 'InputValue': InputValue},
                       ignore_index=True)
        FinalDataResults = FinalDataResults.append({'AlgorithmType': 'JadeCombined', 'MiliTotaltime': dtParticipant["JadeCombined"], 'InputValue': InputValue},
                       ignore_index=True)

        return FinalDataResults

    def PlotFromSQLTimeAll(self,objSQLConfig,TechniqueId,HeartRateStatus,UpSampled,AttemptType):
        #by algo
        folderName=''
        fileName="All_" +HeartRateStatus +"_TimePlot2byInput_" + str(TechniqueId)
        if(UpSampled == '1'):
            folderName = 'UpSampled'
        else:
            folderName = 'NOTUpSampled'
        fullPath = "E:\\ARPOS_Server_Data\\Server_Study_Data\\Plots\\All\\"+folderName+"\\"+ HeartRateStatus+"\\"
        self.objFile.CreatePath(fullPath)
        ####tIME PLOT
        dataTimetoExecute = objSQLConfig.GetAllTimeForTechniqueId(HeartRateStatus, str(UpSampled), str(AttemptType), TechniqueId)
        for i, row in dataTimetoExecute.iterrows():
            totalTime = dataTimetoExecute.at[i, 'totalTime']
            militotalTimeSplit = totalTime.split(':')
            militotalTime = float(militotalTimeSplit[2]) #.split('.')[1]
            # militotalTime = round(militotalTime)
            # dataTimetoExecute.at[i,'MiliTotaltime'] = militotalTime
            dataTimetoExecute.loc[i,'MiliTotaltime']  =militotalTime
            # dataTimetoExecute['MiliTotaltime'] = militotalTime
            a=0

        FinalDataResults = pd.DataFrame()
        FinalDataResults = pd.DataFrame(columns=['AlgorithmType', 'MiliTotaltime', 'InputValue'])

        Condition1 = dataTimetoExecute['ParticipantId'] == 'PIS-1118'
        OneParticipant = dataTimetoExecute.where(Condition1).copy(deep=True)
        OneParticipant=OneParticipant.dropna()
        OneParticipant = OneParticipant.groupby(['AlgorithmType'])['MiliTotaltime'].sum()
        FinalDataResults =self.appdendataTime(FinalDataResults,OneParticipant,1)


        Condition2 =  (dataTimetoExecute['ParticipantId'] == "PIS-4014") \
                        | (dataTimetoExecute['ParticipantId'] == "PIS-2212") \
                        | (dataTimetoExecute['ParticipantId'] == "PIS-7728")
        ThreeParticipant = dataTimetoExecute.where(Condition2).copy(deep=True)
        ThreeParticipant=ThreeParticipant.dropna()
        ThreeParticipant = ThreeParticipant.groupby(['AlgorithmType'])['MiliTotaltime'].sum()
        FinalDataResults =self.appdendataTime(FinalDataResults,ThreeParticipant,3)

        Condition3 =  (dataTimetoExecute['ParticipantId'] == "PIS-4014") \
                        | (dataTimetoExecute['ParticipantId'] == "PIS-2212") \
                        | (dataTimetoExecute['ParticipantId'] == "PIS-7728") \
                      | (dataTimetoExecute['ParticipantId'] == "PIS-7180") \
                        | (dataTimetoExecute['ParticipantId'] == "PIS-1032") \
                        | (dataTimetoExecute['ParticipantId'] == "PIS-7381")
        SixParticipant = dataTimetoExecute.where(Condition3).copy(deep=True)
        SixParticipant=SixParticipant.dropna()
        SixParticipant = SixParticipant.groupby(['AlgorithmType'])['MiliTotaltime'].sum()
        FinalDataResults =self.appdendataTime(FinalDataResults,SixParticipant,6)

        Condition4 = (dataTimetoExecute['ParticipantId'] == "PIS-4014") \
                    | (dataTimetoExecute['ParticipantId'] == "PIS-2212") \
                     | (dataTimetoExecute['ParticipantId'] == "PIS-7728") \
                     | (dataTimetoExecute['ParticipantId'] == "PIS-7180") \
                    |(dataTimetoExecute['ParticipantId'] == "PIS-1032") \
                     | (dataTimetoExecute['ParticipantId'] == "PIS-7381") \
                    |(dataTimetoExecute['ParticipantId'] == "PIS-396") \
                | (dataTimetoExecute['ParticipantId'] == "PIS-3186") \
                    |(dataTimetoExecute['ParticipantId'] == "PIS-5868P2") \
                    | (dataTimetoExecute['ParticipantId'] == "PIS-3252P2")
        TenParticipant = dataTimetoExecute.where(Condition4).copy(deep=True)
        TenParticipant=TenParticipant.dropna()
        TenParticipant = TenParticipant.groupby(['AlgorithmType'])['MiliTotaltime'].sum()
        FinalDataResults =self.appdendataTime(FinalDataResults,TenParticipant,10)

        Condition5 = (dataTimetoExecute['ParticipantId'] == "PIS-4014") \
                        | (dataTimetoExecute['ParticipantId'] == "PIS-2212") \
                        | (dataTimetoExecute['ParticipantId'] == "PIS-7728") \
                     | (dataTimetoExecute['ParticipantId'] == "PIS-7180") \
                     | (dataTimetoExecute['ParticipantId'] == "PIS-1032") \
                     | (dataTimetoExecute['ParticipantId'] == "PIS-7381") \
                        | (dataTimetoExecute['ParticipantId'] == "PIS-396") \
                        |(dataTimetoExecute['ParticipantId'] == "PIS-3186") \
                        | (dataTimetoExecute['ParticipantId'] == "PIS-5868P2") \
                        | (dataTimetoExecute['ParticipantId'] == "PIS-3252P2") \
                        | (dataTimetoExecute['ParticipantId'] == "PIS-6888") \
                     | (dataTimetoExecute['ParticipantId'] == "PIS-667") \
                        | (dataTimetoExecute['ParticipantId'] == "PIS-1949") \
                        |(dataTimetoExecute['ParticipantId'] == "PIS-8308P2") \
                     | (dataTimetoExecute['ParticipantId'] == "PIS-3807")
        fifteenParticipant = dataTimetoExecute.where(Condition5).copy(deep=True)
        fifteenParticipant=fifteenParticipant.dropna()
        fifteenParticipant = fifteenParticipant.groupby(['AlgorithmType'])['MiliTotaltime'].sum()
        FinalDataResults =self.appdendataTime(FinalDataResults,fifteenParticipant,15)

        Condition6 = (dataTimetoExecute['ParticipantId'] == "PIS-4014") \
                    |(dataTimetoExecute['ParticipantId'] == "PIS-2212") \
                     | (dataTimetoExecute['ParticipantId'] == "PIS-7728") \
                    |(dataTimetoExecute['ParticipantId'] == "PIS-7180") \
                     | (dataTimetoExecute['ParticipantId'] == "PIS-1032") \
                     | (dataTimetoExecute['ParticipantId'] == "PIS-7381") \
                     | (dataTimetoExecute['ParticipantId'] == "PIS-396") \
                     | (dataTimetoExecute['ParticipantId'] == "PIS-3186") \
                     | (dataTimetoExecute['ParticipantId'] == "PIS-5868P2") \
                    | (dataTimetoExecute['ParticipantId'] == "PIS-3252P2") \
                    | (dataTimetoExecute['ParticipantId'] == "PIS-6888") \
                    | (dataTimetoExecute['ParticipantId'] == "PIS-667") \
                    |(dataTimetoExecute['ParticipantId'] == "PIS-1949") \
                    | (dataTimetoExecute['ParticipantId'] == "PIS-8308P2") \
                     | (dataTimetoExecute['ParticipantId'] == "PIS-3807") \
                    | (dataTimetoExecute['ParticipantId'] == "PIS-5456") \
                    | (dataTimetoExecute['ParticipantId'] == "PIS-2047") \
                    | (dataTimetoExecute['ParticipantId'] == "PIS-4709") \
                    | (dataTimetoExecute['ParticipantId'] == "PIS-8308") \
                    | (dataTimetoExecute['ParticipantId'] == "PIS-6729")
        twentyParticipant = dataTimetoExecute.where(Condition6).copy(deep=True)
        twentyParticipant = twentyParticipant.dropna()
        twentyParticipant = twentyParticipant.groupby(['AlgorithmType'])['MiliTotaltime'].sum()
        FinalDataResults =self.appdendataTime(FinalDataResults,twentyParticipant,20)

        Condition7 =  (dataTimetoExecute['ParticipantId'] == "PIS-4014") \
                      | (dataTimetoExecute['ParticipantId'] == "PIS-2212") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-7728") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-7180") \
                      | (dataTimetoExecute['ParticipantId'] == "PIS-1032") \
                      | (dataTimetoExecute['ParticipantId'] == "PIS-7381") \
                      | (dataTimetoExecute['ParticipantId'] == "PIS-396") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-3186") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-5868P2") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-3252P2") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-6888") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-667") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-1949") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-8308P2") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-3807") \
                      | (dataTimetoExecute['ParticipantId'] == "PIS-5456") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-2047") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-4709") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-8308") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-6729") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-2740") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-3252") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-1118") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-6327") \
          | (dataTimetoExecute['ParticipantId'] == "PIS-5868")
        twentyfiveParticipant = dataTimetoExecute.where(Condition7).copy(deep=True)
        twentyfiveParticipant = twentyfiveParticipant.dropna()
        twentyfiveParticipant = twentyfiveParticipant.groupby(['AlgorithmType'])['MiliTotaltime'].sum()
        FinalDataResults =self.appdendataTime(FinalDataResults,twentyfiveParticipant,25)

        AllSummed = dataTimetoExecute.groupby(['AlgorithmType'])['MiliTotaltime'].sum() #Over all participants 31
        FinalDataResults =self.appdendataTime(FinalDataResults,AllSummed,31)
        ## get for one two etc then send to plot
        fileName="BarplotTime_All_" +HeartRateStatus +"_" + str(TechniqueId)
        self.MakeLinePlotTimefromSQLData(FinalDataResults, fullPath, fileName,'InputValue','MiliTotaltime',"AlgorithmType")

    def GenerateCasesMain(self):
        self.CaseList = []
        for preprocesstype in self.objConfig.preprocesses:
            for algoType in self.objConfig.AlgoList:
                for fftype in self.objConfig.fftTypeList:
                    for resulttype in self.objConfig.resulttypeList:
                        for filtertype in self.objConfig.filtertypeList:
                            for isSmooth in self.objConfig.Smoothen:
                                fileName = algoType + "_PR-" + str(preprocesstype) + "_FFT-" + str(
                                    fftype) + "_FL-" + str(filtertype) \
                                           + "_RS-" + str(resulttype) + "_SM-" + str(isSmooth)
                                if (fileName not in self.CaseList):
                                    self.CaseList.append(fileName)


    def RunCasewiseFromFolderCustomCases(self, position):
        # self.CustomCaseListMain()  # GET Directories or generate below
        self.GenerateCasesMain()
        UpSampleData = False
        dfFinal = []
        for case in self.CaseList:
            for participant_number in self.objConfig.ParticipantNumbers:
                FolderNameforSave = 'ProcessedDataWindows'
                if (self.objConfig.RunAnalysisForEntireSignalData):
                    if (UpSampleData):
                        FolderNameforSave = 'ProcessedDataUpSampled'  # 'ProcessedData'byProcessType ,ProcessedDataRevised,ProcessedDataUpSampled
                    else:
                        FolderNameforSave = 'ProcessedData'  # 'ProcessedData'byProcessType ,ProcessedDataRevised,ProcessedDataUpSampled

                self.objConfig.setSavePath(participant_number, position, FolderNameforSave)#ProcessedDataRevised,ProcessedData,ProcessedDataUpSampled
                CaseData = self.getCasebyPath(self.objConfig.SavePath + 'Result\\', 'HRSPOwithLog_'+ case)#+ '_UpSampleData-' + str(UpSampleData)
                if (CaseData != None):
                    if (CaseData == ''):
                        print(str(case) + '_' + participant_number + '= NoHR')  # nmeans does not work for this case for this person
                        # needs to be checked if not same for all participants
                    else:
                        differenceVal = self.splitDataRow(CaseData, participant_number, position)  # ROW,COlum
                        dfFinal.append(
                            {
                                'Techniques': case,
                                'Participants': participant_number,
                                'Differences': differenceVal
                            })
                else:
                    print(str(case) + '_' + participant_number + '= NotGenerated')

        dfFinal = pd.DataFrame(dfFinal)
        dfFinal.to_csv(self.objConfig.DiskPath + "CaseWiseFromFolder_Results_" + position + "_CustomCases_ProcessedDataRevised.csv")


    def RunCasewiseFromFolder(self,position,useAcceptableDiff):
        CaseList, LoadfNameList = self.GenerateCases()  # GET Directories or generate below

        dfFinal = []
        for case in CaseList:
            for participant_number in self.objConfig.ParticipantNumbers:
                self.objConfig.setSavePath(participant_number,position,'ProcessedDatabyProcessType')
                CaseData = self.getCase(case)
                if (CaseData != None):
                    if(CaseData == ''):
                        print(str(case) + '_' + participant_number + '= NoHR') #nmeans does not work for this case for this person
                        #needs to be checked if not same for all participants
                    else:
                        differenceVal = self.splitDataRow(CaseData, participant_number, position)  # ROW,COlum
                        dfFinal.append(
                            {
                                'Techniques': case,
                                'Participants': participant_number,
                                'Differences': differenceVal
                            })
                        # if(useAcceptableDiff):
                        #     isAcceptable = self.IsAcceptableDifference(differenceVal)
                        #     if (isAcceptable):
                        #         df1.iloc[RowIndex, ColumnIndex] = differenceVal
                        # else:
                        #     df1.iloc[RowIndex, ColumnIndex] = differenceVal
                else:
                    print(str(case) + '_' + participant_number + '= NotGenerated')


        dfFinal = pd.DataFrame(dfFinal)
        dfFinal.to_csv(self.objConfig.DiskPath + "CaseWiseFromFolder_Results_" + position + "_All"  + ".csv")

        t = 0

    def getBlankCases(self,position):
        CaseList, LoadfNameList = self.GenerateCases()  # GET Directories or generate below
        # CaseList = self.GenerateCases() # or generate

        df1 = pd.DataFrame({
            'CaseNames': CaseList
        })
        NoHrFiles = []
        NotGenerated = []

        for participant_number in self.objConfig.ParticipantNumbers:
            df1[participant_number] = None
        RowIndex = 0  #
        for case in CaseList:
            ColumnIndex = 1  # so doesntt replace case names
            for participant_number in self.objConfig.ParticipantNumbers:
                # for position in self.objConfig.hearratestatus:
                self.objConfig.setSavePath(participant_number,position,'ProcessedDatabyProcessType')
                CaseData = self.getCase(case)
                if (CaseData != None):
                    if(CaseData == ''):
                        NoHrFiles.append(case)
                        # df1.iloc[RowIndex, ColumnIndex] = 'NoHR' #nmeans does not work for this case for this person
                else:
                    NotGenerated.append(case)
                ColumnIndex = ColumnIndex + 1
            RowIndex = RowIndex + 1

        print('')
        print('NotGenerated')
        # for item in NotGenerated:
        #     print(item)

        self.objFile.WriteListDatatoFile(self.objConfig.DiskPath,'NotGeneratedCases',NotGenerated)

        print('')
        print('NoHrFiles')
        # for item in NoHrFiles:
        #     print(item)

        self.objFile.WriteListDatatoFile(self.objConfig.DiskPath,'NoHrFilesCases',NoHrFiles)

    def getDuplicateValue(self, ini_dict):
        # finding duplicate values
        flipped = {}
        for key, value in ini_dict.items():
            if value not in flipped:
                flipped[value] = 1
            else:
                val = flipped.get(value)
                flipped[value] = val + 1
        # print(str(max(flipped.values())))
        # printing result
        # print("final_dictionary", str(flipped))
        key = [fps for fps, count in flipped.items() if count == max(flipped.values())]
        return key[0]

    def getBestCasesFromCSV(self, position):
        df1 = pd.read_csv(self.objConfig.DiskPath + "CaseWiseParticipantsResults_" + position + "__DiffOf" + str(self.AcceptableDifference) + ".csv")
        NoHRlist =[]
        ListCaseCount = {}
        for index, row in df1.iterrows():
            currentCase = row[1]
            addCase = True
            mincount = 0

            for column in df1:
                if (column.__contains__('Unnamed') or column.__contains__('Case')):
                    continue
                colIndex = df1.columns.get_loc(column)  # df1[column] just column in df1 get entire columns values
                colValueInRow = df1.iloc[index, colIndex]
                if(colValueInRow == 'NoHR'):
                    NoHRlist.append(currentCase)
                else:
                    colValueInRow = float(colValueInRow)
                    if (not math.isnan(colValueInRow)):
                        mincount = mincount + 1

            ListCaseCount[currentCase] = mincount

        MaxCount=0
        ListCasesForSelection = []
        for key, value in ListCaseCount.items():
            if (value > MaxCount):
                MaxCount = value
                # ListCasesForSelection.append(key)


        for key, value in ListCaseCount.items():
            if(value >= MaxCount):
                ListCasesForSelection.append(key)
                # ListCasesForSelection.remove(key)

        # MaxCount = self.getDuplicateValue(ListCaseCount)
        NoHRlist = list(dict.fromkeys(NoHRlist))
        print('')
        #Rerun for cases:
        for item in NoHRlist:
            print(item)

        print('')
        print('Best Cases')
        ListCasesForSelection = list(dict.fromkeys(ListCasesForSelection))
        for item in ListCasesForSelection:
            if(item.__contains__('FastICAComponents3')):
                ListCasesForSelection.remove(item)

        for item in ListCasesForSelection:
            print(item)

        ###ReGeneraete sheet as
        # totalLengofCaes = len(ListCasesForSelection)
        # totalCharts = 8 # int(round(totalLengofCaes/7))
        # TotalListsOfCases = []
        # initialindex=0
        # list1 = ListCasesForSelection[initialindex:totalCharts]
        # initialindex=totalCharts
        # totalCharts = totalCharts +8
        # list2 = ListCasesForSelection[initialindex:totalCharts]
        # initialindex=totalCharts
        # totalCharts = totalCharts +8
        # list3 = ListCasesForSelection[initialindex:totalCharts]
        # initialindex=totalCharts
        # totalCharts = totalCharts +8
        # list4 = ListCasesForSelection[initialindex:totalCharts]
        # initialindex=totalCharts
        # totalCharts = totalCharts +8
        # list5 = ListCasesForSelection[initialindex:totalCharts]
        # initialindex=totalCharts
        # totalCharts = totalCharts +8
        # list6 = ListCasesForSelection[initialindex:totalCharts]
        # list7 = ListCasesForSelection
        n=8
        for x in range(0, len(ListCasesForSelection),n):
            list1 = ListCasesForSelection[x:x+n]
            self.CreateCSVdataFromList(list1,position,x,df1)
            a=0
        # self.CreateCSVdataFromList(list2,position,2,df1)
        # self.CreateCSVdataFromList(list3,position,3,df1)
        # self.CreateCSVdataFromList(list4,position,4,df1)
        # self.CreateCSVdataFromList(list5,position,5,df1)
        # self.CreateCSVdataFromList(list6,position,6,df1)
        # self.CreateCSVdataFromList(list7,position,7,df1)


        t = 0
        # BestCaseList = []
        # for index, row in df1.iterrows():
        #     currentCase = row[1]
        #     addCase= True
        #     for column in df1:
        #         if(column.__contains__('Unnamed') or column.__contains__('Case')):
        #             continue
        #         colIndex = df1.columns.get_loc(column) #df1[column] just column in df1 get entire columns values
        #         colValueInRow =  df1.iloc[index, colIndex]
        #         if(np.isnan(colValueInRow)):
        #             addCase = False
        #
        #     if(addCase):
        #         BestCaseList.append(currentCase)
        #
        # for item in BestCaseList:
        #     print(item)
    def CreateCSVdataFromList(self,ListCasesForSelection,position,listno,df1):
        dfFinal =[]

        RowIndex = 0  #
        for case in ListCasesForSelection:
            ColumnIndex = 1  # so doesntt replace case names
            for participant_number in self.objConfig.ParticipantNumbers:
                # for position in self.objConfig.hearratestatus:
                self.objConfig.setSavePath(participant_number, position, 'ProcessedDatabyProcessType')
                CaseData = self.getCase(case)
                if (CaseData != None):
                    if (CaseData == ''):
                        print('df1.iloc[' + RowIndex + ', ' + ColumnIndex +'] NoHR')  # nmeans does not work for this case for this person
                        # needs to be checked if not same for all participants
                    else:
                        differenceVal = self.splitDataRow(CaseData, participant_number, position)  # ROW,COlum
                        df1.iloc[RowIndex, ColumnIndex] = differenceVal
                        dfFinal.append(
                            {
                                'Techniques': case,
                                'Participants': participant_number,
                                'Differences': differenceVal
                            })
                else:
                    print('df1.iloc['+RowIndex+', '+ColumnIndex+'] = ' + 'NotGenerated')
                # else:
                #     df1.iloc[RowIndex, ColumnIndex] = None
                ColumnIndex = ColumnIndex + 1
                # print(df1)
            RowIndex = RowIndex + 1

        dfFinal =pd.DataFrame(dfFinal)

        # write dataFrame to SalesRecords CSV file
        dfFinal.to_csv(self.objConfig.DiskPath + "BoxPlotCSV\\BoxPlotdataBestCaseParticipantsResults_" + position + "_DiffOf" + str(self.AcceptableDifference) +  "_listno-"+str(listno) +".csv")


    """
       LoadFiles:
       Load file from path
       """
    def LoadFiles(self, filepath):
        CaseList = []
        for path, subdirs, files in os.walk(filepath):
            for filename in subdirs:
                if (filename not in CaseList):
                    CaseList.append(filename)
                # a.write(str(f) + os.linesep)
        return CaseList

    def RunParticipantWiseAll(self):
        CaseList = []
        for participant_number in self.objConfig.ParticipantNumbers:
            for position in self.objConfig.hearratestatus:
                CaseSublist = []

                loadpath = self.objConfig.DiskPath + 'Result\\' + participant_number + '\\' + position + '\\'
                print(loadpath)
                CaseSublist = self.LoadFiles(loadpath)

                for name in CaseSublist:
                    if (name not in CaseList):
                        CaseList.append(name)

        finallist = 0
        return CaseList

        # file = open(loadpath + filename + ".txt", "r")
        # Lines = file.readlines()
        # file.close()

    def RunAllCaseParticipantwiseCaseasRow(self):
        CaseList = self.GenerateCases()  # or generate

        df1 = pd.DataFrame({
            'CaseNames': CaseList
        })
        # self.objConfig.ParticipantNumbers =["PIS-8073","PIS-2047","PIS-4014","PIS-1949","PIS-3186","PIS-7381","PIS-5937"]
        # self.objConfig.Participantnumbers_SkinGroupTypes =["Europe_WhiteSkin_Group","Europe_WhiteSkin_Group","Europe_WhiteSkin_Group","Europe_WhiteSkin_Group","Europe_WhiteSkin_Group","Europe_WhiteSkin_Group","Europe_WhiteSkin_Group"]

        for participant_number in self.objConfig.ParticipantNumbers:
            df1[participant_number] = None

        for position in self.objConfig.hearratestatus:
            RowIndex = 0  #
            for case in CaseList:
                ColumnIndex = 1  # so doesntt replace case names
                for participant_number in self.objConfig.ParticipantNumbers:
                    # for position in self.objConfig.hearratestatus:
                    CaseData = self.getCaseNew(case, participant_number, position)
                    if (CaseData != None):
                        differenceVal = self.splitDataRow(CaseData, participant_number, position)  # ROW,COlum
                        isAcceptable = self.IsAcceptableDifference(differenceVal)
                        if (isAcceptable):
                            df1.iloc[RowIndex, ColumnIndex] = differenceVal
                    else:
                        df1.iloc[RowIndex, ColumnIndex] = 'NotGenerated'
                    # else:
                    #     df1.iloc[RowIndex, ColumnIndex] = None
                    ColumnIndex = ColumnIndex + 1
                    # print(df1)
                RowIndex = RowIndex + 1

            # write dataFrame to SalesRecords CSV file
            df1.to_csv(
                "E:\\ARPOS_Server_Data\\Server_Study_Data\\Europe_WhiteSkin_Group\\Result\\PIResults_" + position + ".csv")
        t = 0

    def RunAllCaseParticipantwiseCaseasCol(self):
        CaseList = self.GenerateCases()  # or generate

        df1 = pd.DataFrame({
            'ParticipantNumbers': self.objConfig.ParticipantNumbers
        })
        # self.objConfig.ParticipantNumbers =["PIS-8073","PIS-2047","PIS-4014","PIS-1949","PIS-3186","PIS-7381","PIS-5937"]
        # self.objConfig.Participantnumbers_SkinGroupTypes =["Europe_WhiteSkin_Group","Europe_WhiteSkin_Group","Europe_WhiteSkin_Group","Europe_WhiteSkin_Group","Europe_WhiteSkin_Group","Europe_WhiteSkin_Group","Europe_WhiteSkin_Group"]

        for case in CaseList:
            df1[case] = None

        for position in self.objConfig.hearratestatus:
            RowIndex = 0  #
            for participant_number in self.objConfig.ParticipantNumbers:
                ColumnIndex = 1  # so doesntt replace case names
                for case in CaseList:
                    # for position in self.objConfig.hearratestatus:
                    CaseData = self.getCaseNew(case, participant_number, position)
                    if (CaseData != None):
                        differenceVal = self.splitDataRow(CaseData, participant_number, position)  # ROW,COlum
                        isAcceptable = self.IsAcceptableDifference(differenceVal)
                        if (isAcceptable):
                            df1.iloc[RowIndex, ColumnIndex] = differenceVal
                    else:
                        df1.iloc[RowIndex, ColumnIndex] = 'NotGenerated'
                    # else:
                    #     df1.iloc[RowIndex, ColumnIndex] = None
                    ColumnIndex = ColumnIndex + 1
                    # print(df1)
                RowIndex = RowIndex + 1

            # write dataFrame to SalesRecords CSV file
            df1.to_csv(
                "E:\\ARPOS_Server_Data\\Server_Study_Data\\Europe_WhiteSkin_Group\\Result\\PIResults_" + position + ".csv")
        t = 0

    # a
    def MakeBoxPlotbyInput(self,Input,filename,CaseFolder,fileName):
        self.objFile.CreatePath(self.objConfig.DiskPath + "BoxPlotCSV\\BoxPlotImages\\"+CaseFolder+"\\")
        Input.to_csv(self.objConfig.DiskPath + "BoxPlotCSV\\BoxPlotImages\\"+CaseFolder+"\\" + fileName + '.csv')
        # Input.to_csv(fileNamecsv)
                # GroupcaseData.to_csv(
                #     self.objConfig.DiskPath + "BoxPlotCSV\\BoxPlotImages\\CaseGroupWise\\" + case + ".csv")

        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        ####SNS
        sns.set_style('whitegrid')
        sns.set(rc={"figure.figsize": (10, 8)})  # width=3, #height=4
        ax = sns.boxplot(x='Techniques', y='Differences', data=Input)  # rotation=45, horizontalalignment='right',
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax = sns.stripplot(x="Techniques", y="Differences", data=Input)
        plt.savefig(self.objConfig.DiskPath + "BoxPlotCSV\\BoxPlotImages\\"+CaseFolder+"\\" + fileName + ".jpg")


    def MakeBoxPlotfromSQLData(self,dataResult,fullPath,fileName,x, y,title):
        sns.set(font_scale=1.5) # Overaall font size
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        ####SNS
        sns.set_style('whitegrid')
        plt.figure(figsize=(14, 9))  # this creates a figure 8 inch wide, 4 inch high
        # sns.set(rc={"figure.figsize": (14, 10)})  # width=3, #height=4
        ax = sns.boxplot(x=x, y=y, data=dataResult)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax = sns.stripplot(x=x, y=y, data=dataResult)

        plt.title(title, size=18)

        plt.xlabel("Algorithms")
        plt.ylabel("OriginalDifference")
        # for tick in ax.xaxis.get_major_ticks():
        #     tick.label.set_fontsize(15)
        #
        # for tick in ax.yaxis.get_major_ticks():
        #     tick.label.set_fontsize(15)

        plt.tight_layout()

        plt.savefig(fullPath + fileName + ".jpg")

    def MakeBoxPlotGroupWisefromSQLData(self, dataResult, fullPath, fileName,x, y, hue):

        sns.set(font_scale=1.5)  # Overaall font size
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        ####SNS
        plt.figure(figsize=(14, 9))  # this creates a figure 8 inch wide, 4 inch high
        sns.set_style('whitegrid')

        ax = sns.boxplot(x=x, y=y, hue=hue, data=dataResult)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax = sns.stripplot(x=x, y=y, hue=hue, data=dataResult,
                           palette="Set2")  # , size=6, marker="D",edgecolor="gray", alpha=.25
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        [ax.axvline(x, color='black', linestyle='--') for x in [0.5, 1.5, 2.5]]
        plt.show()

        plt.title("Box Plot for over all participants Origival vs revised vs upsampled", size=18)

        plt.xlabel("Algorithms")
        plt.ylabel("OriginalDifference")

        plt.tight_layout()

        plt.savefig(fullPath + fileName + ".jpg")
    def MakeLinePlotTimefromSQLData(self,dataResult,fullPath,fileName,x,y,huet):
        sns.set(font_scale=1.5)  # Overaall font size
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        ####SNS
        plt.figure(figsize=(14, 9))  # this creates a figure 8 inch wide, 4 inch high
        sns.set_style('whitegrid')
        sns.lineplot(
            data=dataResult,
            x=x, y=y, hue=huet
        )
        # plt.title('Time taken (ms) to execute ' + fileName)
        plt.xlabel('Number of Participants', size=15)
        plt.ylabel('Execution time (ms)', size=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15)
        plt.tight_layout()
        plt.savefig(fullPath + fileName + ".jpg")


    def MakeBarTimePlotfromSQLData(self, dataResult, fullPath, fileName):

        sns.set(font_scale=1.5)  # Overaall font size
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        ####SNS
        plt.figure(figsize=(14, 9))  # this creates a figure 8 inch wide, 4 inch high
        sns.set_style('whitegrid')
        sns.barplot(data=dataResult, x="AlgorithmType", y="totalTimeinMilliSeconds")
        plt.title('Time taken (MILLISECONDS) to execute ' + fileName)
        plt.xlabel('Algorithm types')
        plt.ylabel('Average time over all participants')
        plt.tight_layout()
        plt.savefig(fullPath + fileName + ".jpg")

    def TestBoxPlot(self):
        # filename = "BoxPlotdataBestCaseParticipantsResults_Resting1_DiffOf10_listno-0"
        # filename = "BoxPlotdataBestCaseParticipantsResults_Resting2_DiffOf10_listno-0"
        # filename = "BoxPlotdataBestCaseParticipantsResults_Resting2_DiffOf10_listno-8"
        # filename = "BoxPlotdataBestCaseParticipantsResults_Resting2_DiffOf10_listno-16"
        # filename = "BoxPlotdataBestCaseParticipantsResults_Resting2_DiffOf10_listno-24"
        # filename = "BoxPlotdataBestCaseParticipantsResults_Resting2_DiffOf10_listno-32"
        # filename = "BoxPlotdataBestCaseParticipantsResults_Resting2_DiffOf10_listno-40"
        # filename = "BoxPlotdataBestCaseParticipantsResults_Resting2_DiffOf10_listno-48"
        # filename = "BoxPlotdataBestCaseParticipantsResults_Resting2_DiffOf10_listno-48Amended"
        # filename = "BoxPlotdataBestCaseParticipantsResults_Resting2_DiffOf10_listno-56"
        # filename = "BoxPlotdataBestCaseParticipantsResults_Resting2_DiffOf10_listno-64"
        # filename = "BoxPlotdataBestCaseParticipantsResults_Resting2_DiffOf10_listno-72"
        # filename = "30fpsGroupWiseBestTechniques2SM-FALSE"
        # filename = "ParticipantsResultWhiteandOtherAsianUpSampled""
        filename = "CaseWiseFromFolder_Results_Resting1_CustomCases_UpSampleData-True"
        # filename = "ParticipantsResultWhiteandOtherAsianUpSampled"
        # df = pd.read_csv(self.objConfig.DiskPath + "BoxPlotCSV\\"+ filename+ ".csv")  # read file
        # df = pd.read_csv(self.objConfig.DiskPath + "BoxPlotCSV\\FinalBestCases\\"+ filename+ ".csv")  # read file
        df = pd.read_csv(self.objConfig.DiskPath + filename+ ".csv")  # read file
        df.head()

        ####SNS
        sns.set_style('whitegrid')
        sns.set(rc={"figure.figsize": (10, 8)})  # width=3, #height=4
        ax = sns.boxplot(x='Techniques', y='Differences',  data=df)#rotation=45, horizontalalignment='right',
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax = sns.stripplot(x="Techniques", y="Differences", data=df)
        # plt.savefig(self.objConfig.DiskPath + "BoxPlotCSV\\BoxPlotImages\\"+ filename+ ".jpg")
        # plt.savefig(self.objConfig.DiskPath + "BoxPlotCSV\\FinalBestCases\\"+ filename+ ".jpg")
        plt.savefig(self.objConfig.DiskPath + filename+ ".jpg")

    def PlotSingalRegion(self, region, blue, green, red, grey, Irchannel, ColorEstimatedFPS, IREstimatedFPS):
        objPlots = Plots()
        objPlots.ColorEstimatedFPS = ColorEstimatedFPS
        objPlots.IREstimatedFPS = IREstimatedFPS
        objPlots.plotGraphAllwithParam('E:\\TestGraphs\\', 'signal_' + region, None, None,
                                       blue, green, red,
                                       grey, Irchannel,
                                       "Time(s)",
                                       "Amplitude")
        del objPlots

    def PlotSignal(self):
        objProcess = \
            self.objFile.ReadfromDisk(
                'E:\\ARPOS_Server_Data\\Server_Study_Data\\Europe_WhiteSkin_Group\\ProcessedDatabyProcessType\\PIS-8073\\Resting1\\Algorithm_WindowsBinaryFiles\\',
                'ResultSignal_Algorithm-FastICAComponents3_PreProcess-1')

        self.PlotSingalRegion('lips', objProcess.get('lips').regionWindowBlueData,
                              objProcess.get('lips').regionWindowGreenData, objProcess.get('lips').regionWindowRedData,
                              objProcess.get('lips').regionWindowGreyData, objProcess.get('lips').regionWindowIRData,
                              objProcess.get('lips').ColorEstimatedFPS, objProcess.get('lips').IREstimatedFPS)
        self.PlotSingalRegion('forehead', objProcess.get('forehead').regionWindowBlueData,
                              objProcess.get('forehead').regionWindowGreenData,
                              objProcess.get('forehead').regionWindowRedData,
                              objProcess.get('forehead').regionWindowGreyData,
                              objProcess.get('forehead').regionWindowIRData,
                              objProcess.get('lips').ColorEstimatedFPS, objProcess.get('lips').IREstimatedFPS)
        self.PlotSingalRegion('leftcheek', objProcess.get('leftcheek').regionWindowBlueData,
                              objProcess.get('leftcheek').regionWindowGreenData,
                              objProcess.get('leftcheek').regionWindowRedData,
                              objProcess.get('leftcheek').regionWindowGreyData,
                              objProcess.get('leftcheek').regionWindowIRData,
                              objProcess.get('lips').ColorEstimatedFPS, objProcess.get('lips').IREstimatedFPS)
        self.PlotSingalRegion('rightcheek', objProcess.get('rightcheek').regionWindowBlueData,
                              objProcess.get('rightcheek').regionWindowGreenData,
                              objProcess.get('rightcheek').regionWindowRedData,
                              objProcess.get('rightcheek').regionWindowGreyData,
                              objProcess.get('rightcheek').regionWindowIRData,
                              objProcess.get('lips').ColorEstimatedFPS, objProcess.get('lips').IREstimatedFPS)

    def PlotbyInput(self,Technique1,TimeLog):
        plt.scatter(Technique1,TimeLog)
        # plt.legend(Technique1)
        plt.title('title')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig('E:\\TestGraphs\\linePlotTimeLog.jpg')#show()

    def LinePlot(self):
        # df = pd.read_csv("E:\\TestResultLine.csv")  # read file
        # df.head()
        # sns.lineplot(x="Techniques", y="TimeLog", hue='Participants', data=df)
        # plt.show()
        ###FOr window use same but do average?
        Technique1 = ['T1', 'T2', 'T3']  # Over All pariticpants
        TimeLog = [(0.006982, 0.006982, 0.004986, 0.004986), (0.043882, 0.043882, 0.043882, 0.043882),
                   (0.8, 0.4, 0.6, 0.8)]  # (P1,P2,P3,P4)
        plt.plot(Technique1, TimeLog, marker='o')
        plt.legend(['T1', 'T2', 'T3'])
        plt.title('Unemployment Rate Vs Year')
        plt.xlabel('Year')
        plt.ylabel('Unemployment Rate')
        plt.show()

    def TestBoxPlotWindow(self):
        fullPathFiles = "D:\\"#"D:\\ARPOS_Server_Data\\Server_Study_Data\\AllParticipants\\SaveResultstoDiskDataFiles\\PIS-1032\\Resting1\\"
        # self.objFile.CreatePath(fullPathFiles + "Graphs\\")
        FirstDf = pd.read_csv(fullPathFiles + "Test2.csv")  # read file

        # for fileName in os.listdir(fullPathFiles):
        #     if(fileName.__contains__("ComputerHR")):
        #         FirstDf = pd.read_csv(fullPathFiles + fileName)  # read file
        #         if(not FirstDf.empty):
        #             self.GenerateObservedvsActual("Resting1", FirstDf['GroundTruthHeartRate'].tolist(),
        #                                      FirstDf['ComputedHeartRate'].tolist(),
        #                                      fullPathFiles + "Graphs\\",fileName)
        # FirstDf = pd.DataFrame()
        # for fileName in os.listdir(fullPathFiles):
        #     if(not fileName.__contains__("ProcessedCompleted")):
        #         if (fileName.__contains__("_PreProcess_1")):
        #             if(fileName.__contains__("FastICA_") or fileName.__contains__("PCA_") or fileName.__contains__("PCAICA_")): #
        #                 techniqueName = fileName.replace(".csv", "")
        #                 techniqueNameSplit = techniqueName.split("_")
        #                 techniqueName = techniqueNameSplit[len(techniqueNameSplit) - 3] # + "_Pre" + techniqueNameSplit[len(techniqueNameSplit) - 1]
        #
        #                 if(FirstDf.empty):
        #                     FirstDf = pd.read_csv(fullPathFiles+fileName)  # read file
        #                     FirstDf = FirstDf.assign(Techniques=techniqueName)
        #                 else:
        #                     dfnext = pd.read_csv(fullPathFiles+fileName)  # read file
        #                     dfnext.head()
        #                     dfnext = dfnext.assign(Techniques=techniqueName)
        #
        #                     FirstDf=FirstDf.append(dfnext)
        #                     a=0
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()

        FirstDf = FirstDf.reset_index(drop=True)

        FirstDf.head()
        # FirstDf = FirstDf.where(FirstDf['WindowCount']<=3)
        # FirstDf=FirstDf.dropna()
        # define figure size
        sns.set(rc={"figure.figsize": (10, 8)})  # width=6, height=5
        # Draw a vertical boxplot grouped
        # by a categorical variable:
        ###MEthod2#####
        # fig, ax = plt.subplots(1, sharex=False, sharey=False, gridspec_kw={'hspace': 0}, figsize=(10, 5))
        # sns.boxplot(x="Window", y="Differences", hue="Techniques", data=df, palette="PRGn")
        # [ax.axvline(x, color='black', linestyle='--') for x in [0.5,1.5,2.5]]
        # plt.show()
        # sns.lineplot(x="WindowCount", y="HRDifference", hue="Techniques", data=FirstDf)
        # plt.savefig(fullPathFiles + "windowLinefig.jpg")
        #
        # plt.switch_backend('agg')
        # plt.ioff()
        # plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        sns.set(font_scale=1.5)  # Overaall font size
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        #
        # sns.set_style('whitegrid')
        # sns.set(rc={"figure.figsize": (10, 8)})  # width=3, #height=4
        # ax = sns.boxplot(x='Techniques', y='HRDifference',  data=FirstDf)#rotation=45, horizontalalignment='right',
        # # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        # ax = sns.stripplot(x="Techniques", y="HRDifference", data=FirstDf)
        # # plt.savefig(self.objConfig.DiskPath + "BoxPlotCSV\\BoxPlotImages\\"+ filename+ ".jpg")
        # # plt.savefig(self.objConfig.DiskPath + "BoxPlotCSV\\FinalBestCases\\"+ filename+ ".jpg")
        # plt.savefig(fullPathFiles + "windowLinefig2.jpg")
        ###MEthod1#####
        sns.set_style('whitegrid')

        plt.figure(figsize=(14, 9))  # this creates a figure 8 inch wide, 4 inch high
        ax = sns.boxplot(x='WindowCount', y='HRDifference', hue="Techniques", data=FirstDf)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax = sns.stripplot(x="WindowCount", y="HRDifference", hue="Techniques", data=FirstDf,
                           palette="Set2")  # , size=6, marker="D",edgecolor="gray", alpha=.25
        handles, labels = ax.get_legend_handles_labels()
        # plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        [ax.axvline(x, color='black', linestyle='--') for x in [0.5, 1.5, 2.5]]

        plt.title("title", size=18)
        plt.tight_layout()
        plt.savefig(fullPathFiles+"windowfigTest.jpg")
        a=0

        #########method3
        # sns.stripplot(x="WindowCount", y="HRDifference", hue="Techniques",
        #               data=FirstDf, jitter=True,
        #               palette="Set2", split=True, linewidth=1, edgecolor='gray')
        #
        # # Get the ax object to use later.
        # ax = sns.boxplot(x="WindowCount", y="HRDifference", hue="Techniques",
        #                  data=FirstDf, palette="Set2", fliersize=0)
        #
        # # Get the handles and labels. For this example it'll be 2 tuples
        # # of length 4 each.
        # handles, labels = ax.get_legend_handles_labels()
        #
        # # When creating the legend, only use the first two elements
        # # to effectively remove the last two.
        # l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.savefig(fullPathFiles+"windowfig4.jpg")

    # def GenerateObservedvsActual(self,position_type,  Actual_HR_AllValues_Resting, Observed_HR_AllValues_Resting,
    #                              path,fileName):
    #     ###PLOT chart3
    #     plt.ioff()
    #     plt.clf()
    # 
    #     actual = []
    #     observed = []
    # 
    #     actual = Actual_HR_AllValues_Resting
    #     observed = Observed_HR_AllValues_Resting
    # 
    #     actualArry = []
    #     # iterate over the list
    #     for val in actual:
    #         actualArry.append(int(float(val)))
    # 
    #     observedArry = []
    #     # iterate over the list
    #     for val in observed:
    #         observedArry.append(int(float(val)))
    # 
    #     rng = np.random.RandomState(0)
    #     sizes = 1000 * rng.rand(len(Actual_HR_AllValues_Resting))
    #     true_value = actualArry
    #     observed_value = observedArry
    #     plt.figure(figsize=(10, 10))
    #     plt.rc('font', size=16)
    #     plt.scatter(true_value, observed_value, c='crimson', s=sizes, alpha=0.3)
    #     # plt.yscale('log')
    #     # plt.xscale('log')
    # 
    #     p1 = max(max(observed_value), max(true_value))
    #     p2 = min(min(observed_value), min(true_value))
    #     plt.plot([p1, p2], [p1, p2], 'b-')
    #     plt.xlabel('Commercial Heart Rate (bpm)', fontsize=15)
    #     plt.ylabel('ARPOS Heart Rate (bpm)', fontsize=15)
    #     plt.tick_params(axis='both', which='major', labelsize=13)
    #     plt.tick_params(axis='both', which='minor', labelsize=12)
    #     plt.axis('equal')
    #     plt.savefig(path + fileName + "_ActualvsObserved"  + ".png")
    #     plt.close()

    def GenerateCases(self):
        CaseList = []
        LoadfNameList = []
        for preprocesstype in self.objConfig.preprocesses:
            for algoType in self.objConfig.AlgoList:
                for isSmooth in self.objConfig.Smoothen:
                    for fftype in self.objConfig.fftTypeList:
                        for filtertype in self.objConfig.filtertypeList:
                            for resulttype in self.objConfig.resulttypeList:
                                fileName = "HRSPOwithLog_" + str(algoType) + "_PR-" + str(
                                    preprocesstype) + "_FFT-" + str(fftype) \
                                           + "_FL-" + str(filtertype) + "_RS-" + str(resulttype) + "_SM-" + str(
                                    isSmooth)
                                LoadName = 'ResultSignal_Result-' + str(resulttype) + '_PreProcess-' + str(
                                    preprocesstype) \
                                           + '_Algorithm-' + str(algoType) + '_Smoothen-' + str(isSmooth) \
                                           + '_FFT-' + str(fftype) + '_Filter-' + str(filtertype)
                                if (fileName not in CaseList):  # not os.path.exists(self.ProcessedDataPath + fName):
                                    CaseList.append(fileName)
                                    LoadfNameList.append(LoadName)
        return CaseList, LoadfNameList

    def CheckIfGenerated(self, filePath, fileName):
        # objFile= FileIO()
        # SavePath = self.objConfig.SavePath + fileName + '\\'
        # pathExsists = objFile.FileExits(filePath + fileName + ".txt")
        # data= None
        # # already generated
        # if (pathExsists):
        file = open(filePath + fileName + ".txt", "r")  # objFile.ReaddatafromFile(filePath,fileName)[0]
        data = file.read()
        file.close()
        return data

    def CheckIfGeneratedAllLines(self, filePath, fileName):
        # objFile= FileIO()
        # SavePath = self.objConfig.SavePath + fileName + '\\'
        # pathExsists = objFile.FileExits(filePath + fileName + ".txt")
        # data= None
        # # already generated
        # if (pathExsists):
        file = open(filePath + fileName + ".txt", "r")  # objFile.ReaddatafromFile(filePath,fileName)[0]
        data = file.readlines()
        file.close()
        return data

    def getDataforpiandpositionWindow(self, ParticipantNumber, position, AcceptanceDiff, CasesAll):
        FileValues = {}
        for item in CasesAll:
            fullpath = 'E:\\ARPOS_Server_Data\\Server_Study_Data\\Europe_WhiteSkin_Group\\ProcessedDatabyProcessType\\' + ParticipantNumber + '\\' + position + '\\ComputedFinalResult\\'
            fileName = item
            FileContent = self.CheckIfGeneratedAllLines(fullpath, fileName)
            if (len(FileContent) > 0):
                for itemFile in FileContent:
                    FileContentSplit = itemFile.split(' ,\t')
                    # WindowCount: 0 ,	GroundTruthHeartRate: 67 ,	ComputedHeartRate: 100 ,	HRDifference: -33 ,
                    diff = FileContentSplit[3].replace('HRDifference: ', '')
                    diff = np.abs(float(diff))
                    if (diff <= AcceptanceDiff):
                        FileValues[fileName] = FileContentSplit
        return FileValues

    def getDataforpiandposition(self, ParticipantNumber, position, AcceptanceDiff, CasesAll):
        FileNames = []
        for item in CasesAll:
            fullpath = 'E:\\ARPOS_Server_Data\\Server_Study_Data\\Europe_WhiteSkin_Group\\ProcessedData\\' + ParticipantNumber + '\\' + position + '\\Result\\'
            fileName = 'HRSPOwithLog_' + item
            FileContent = self.CheckIfGenerated(fullpath, fileName)
            if (len(FileContent) > 0):
                FileContentSplit = FileContent.split(' ,\t')
                # WindowCount: 0 ,	GroundTruthHeartRate: 67 ,	ComputedHeartRate: 100 ,	HRDifference: -33 ,
                diff = FileContentSplit[3].replace('HRDifference: ', '')
                diff = np.abs(float(diff))
                if (diff <= AcceptanceDiff):
                    FileNames.append(item)
        return FileNames

    def ShowEmptyResults(self, ParticipantNumber, position, CasesAll):
        FileNames = []
        for item in CasesAll:
            fullpath = 'E:\\ARPOS_Server_Data\\Server_Study_Data\\Europe_WhiteSkin_Group\\ProcessedData\\' + ParticipantNumber + '\\' + position + '\\Result\\'
            fileName = 'HRSPOwithLog_' + item
            FileContent = self.CheckIfGenerated(fullpath, fileName)
            if (len(FileContent) > 0):
                skip = 0
            else:
                FileNames.append(fileName)
        return FileNames
        # for x in FileNames:
        #     print(x)

    def GetBestFilesForParticipantEntireSingal(self, AcceptanceDiff):
        position = 'Resting1'
        CasesAll = self.GenerateCases2()
        ParticipantNumber = 'PIS-8073'
        FileName1 = self.getDataforpiandposition(ParticipantNumber, position, AcceptanceDiff, CasesAll)
        position = 'Resting2'
        FileName2 = self.getDataforpiandposition(ParticipantNumber, position, AcceptanceDiff, CasesAll)

        CommonFiles = []
        for item in CasesAll:
            if (FileName1.__contains__(item) and FileName2.__contains__(item)):
                CommonFiles.append(item)

        for item in CommonFiles:
            print(item)

    def LogTime(self,hour,minute,second,microsecond):
        logTime = datetime(datetime.now().year, datetime.now().month, datetime.now().day,
                           hour, minute,
                           second, microsecond)
        return logTime

    def GetBestFilesForAllParticipantWindowAllSingal(self, AcceptanceDiff,position):
        CommonFiles=[]
        AllFileNames=[]
        for ParticipantNumber in self.objConfig.ParticipantNumbers:
            CasesAll = self.GenerateCases2()
            FileContent = self.getDataforpiandpositionWindow(ParticipantNumber, position, AcceptanceDiff, CasesAll)
            FileNames= list(FileContent.keys())
            AllFileNames.append(FileNames)

        ##Remove Duplicates

    def GetBestFilesForParticipantWindowAllSingal(self, AcceptanceDiff, ParticipantNumber,RunforType):
        position = 'Resting1'
        CasesAll = self.GenerateCases2()
        FileContent1 = self.getDataforpiandpositionWindow(ParticipantNumber, position, AcceptanceDiff, CasesAll)

        position = 'Resting2'
        FileContent2 = self.getDataforpiandpositionWindow(ParticipantNumber, position, AcceptanceDiff, CasesAll)

        if(RunforType =='TimePlot'):
            Technique_All = []
            TimeLog = []
            for k, v in FileContent1.items():
                Technique_All.append(k)
                timeStamp= v[9].replace('TotalWindowCalculationTime: ','')
                timeStampSplit = timeStamp.split(':')#0:00:00.032910
                secondSplit = timeStampSplit[2].split('.')
                LogTime = self.LogTime(int(timeStampSplit[0]),int(timeStampSplit[1]),int(secondSplit[0]),int(secondSplit[1]))
                TimeLog.append(LogTime)

            # create DataFrame
            df = pd.DataFrame({'techq': Technique_All,
                               'value': LogTime})

            sns.lineplot( x='value',hue='techq', data=df)
            plt.show()
            # self.PlotbyInput(Technique_All,TimeLog)
            a=0
        elif(RunforType =='getCommonFiles'):
            FileName1= list(FileContent1.keys())
            FileName2=list(FileContent2.keys())
            CommonFiles = list(set(FileName1) & set(FileName2))
            PCAICAFiles = []
            FastICAFiles = []
            PCA_Files = []
            None_Files = []
            Jade_Files = []
            Other = []
            for item in CommonFiles:
                fileName = item  # item.replace('HRSPOwithLog_HRSPOwithLog_','')
                if (fileName.__contains__('PCAICA')):
                    PCAICAFiles.append(fileName)
                elif (fileName.__contains__('FastICA')):
                    FastICAFiles.append(fileName)
                elif (fileName.__contains__('PCA_')):
                    PCA_Files.append(fileName)
                elif (fileName.__contains__('None')):
                    None_Files.append(fileName)
                elif (fileName.__contains__('Jade')):
                    Jade_Files.append(fileName)
                else:
                    Other.append(fileName)

            print('Other')
            for item in Other:
                fileName = item.replace('HRSPOwithLog_HRSPOwithLog_', '')
                print(fileName)

            print('PCAICAFiles')
            for item in PCAICAFiles:
                fileName = item.replace('HRSPOwithLog_HRSPOwithLog_', '')
                print(fileName)

            print('FastICAFiles')
            for item in FastICAFiles:
                fileName = item.replace('HRSPOwithLog_HRSPOwithLog_', '')
                if (fileName.__contains__('FastICAComponents3_')):
                    skip = 0
                else:
                    print(fileName)

            print('PCA_Files')
            for item in PCA_Files:
                fileName = item.replace('HRSPOwithLog_HRSPOwithLog_', '')
                print(fileName)

            print('None_Files')
            for item in None_Files:
                fileName = item.replace('HRSPOwithLog_HRSPOwithLog_', '')
                print(fileName)

            print('Jade_Files')
            for item in Jade_Files:
                fileName = item.replace('HRSPOwithLog_HRSPOwithLog_', '')
                print(fileName)

    def getEmptyResults(self):
        position = 'Resting1'
        CasesAll = self.GenerateCases2()
        ParticipantNumber = 'PIS-8073'
        FileNames1 = self.ShowEmptyResults(ParticipantNumber, position, CasesAll)
        # print('FileNames1')
        # for item in FileNames1:
        #     print(item)
        position = 'Resting2'
        FileNames2 = self.ShowEmptyResults(ParticipantNumber, position, CasesAll)
        # print('FileNames2')
        # for item in FileNames2:
        #     print(item)
        CommonFiles = list(set(FileNames1) & set(FileNames2))
        # CommonFiles = []
        # for item in CasesAll:
        #     if (FileNames1.__contains__(item) and FileNames2.__contains__(item)):
        #         CommonFiles.append(item)
        #
        for item in CommonFiles:
            print(item)
    #
    # def WriteCases(self):
    #     CaseList= []
    #     dfFinal=[]
    #     for preprocesstype in self.objConfig.preprocesses:
    #         for algoType in self.objConfig.AlgoList:
    #             for isSmooth in self.objConfig.Smoothen:
    #                 for fftype in self.objConfig.fftTypeList:
    #                     for filtertype in self.objConfig.filtertypeList:
    #                         for resulttype in self.objConfig.resulttypeList:
    #                             fileName = "HRSPOwithLog_" + str(algoType) + "_PR-" + str(
    #                                 preprocesstype) + "_FFT-" + str(fftype) \
    #                                        + "_FL-" + str(filtertype) + "_RS-" + str(resulttype) + "_SM-" + str(
    #                                 isSmooth)
    #                             if (fileName not in CaseList):
    #                                 CaseList.append(fileName)
    #                                 dfFinal.append(
    #                                     {
    #                                         'algoType': algoType,
    #                                         'preprocesstype': preprocesstype,
    #                                         'fftype': fftype,
    #                                         'filtertype': filtertype,
    #                                         'resulttype': resulttype,
    #                                         'isSmooth': isSmooth
    #                                     })
    #     dfFinal = pd.DataFrame(dfFinal)
    #     dfFinal.to_csv(self.objConfig.DiskPath + "AllCases.csv")

    def PlotComparisonMethodMean(self):

        method1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        method2 = [1.03, 2.05, 2.79, 3.67, 5.00, 5.82, 7.16, 7.69, 8.53, 10.38, 11.11, 12.17, 13.47, 13.83, 15.15, 16.12, 16.94, 18.09, 19.13, 19.54]
        pyCompare.blandAltman(method1, method2,)
        plt.show()


    def AllPlotsforComputedResults(self):
        fullPathFiles = "D:\\"#"D:\\ARPOS_Server_Data\\Server_Study_Data\\AllParticipants\\SaveResultstoDiskDataFiles\\PIS-1032\\Resting1\\"
        # self.objFile.CreatePath(fullPathFiles + "Graphs\\")
        FirstDf = pd.read_csv(fullPathFiles + "Test2.csv")  # read file

        # for fileName in os.listdir(fullPathFiles):
        #     if(fileName.__contains__("ComputerHR")):
        #         FirstDf = pd.read_csv(fullPathFiles + fileName)  # read file
        #         if(not FirstDf.empty):
        #             self.GenerateObservedvsActual("Resting1", FirstDf['GroundTruthHeartRate'].tolist(),
        #                                      FirstDf['ComputedHeartRate'].tolist(),
        #                                      fullPathFiles + "Graphs\\",fileName)
        # FirstDf = pd.DataFrame()
        # for fileName in os.listdir(fullPathFiles):
        #     if(not fileName.__contains__("ProcessedCompleted")):
        #         if (fileName.__contains__("_PreProcess_1")):
        #             if(fileName.__contains__("FastICA_") or fileName.__contains__("PCA_") or fileName.__contains__("PCAICA_")): #
        #                 techniqueName = fileName.replace(".csv", "")
        #                 techniqueNameSplit = techniqueName.split("_")
        #                 techniqueName = techniqueNameSplit[len(techniqueNameSplit) - 3] # + "_Pre" + techniqueNameSplit[len(techniqueNameSplit) - 1]
        #
        #                 if(FirstDf.empty):
        #                     FirstDf = pd.read_csv(fullPathFiles+fileName)  # read file
        #                     FirstDf = FirstDf.assign(Techniques=techniqueName)
        #                 else:
        #                     dfnext = pd.read_csv(fullPathFiles+fileName)  # read file
        #                     dfnext.head()
        #                     dfnext = dfnext.assign(Techniques=techniqueName)
        #
        #                     FirstDf=FirstDf.append(dfnext)
        #                     a=0
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()

        FirstDf = FirstDf.reset_index(drop=True)

        FirstDf.head()
        # FirstDf = FirstDf.where(FirstDf['WindowCount']<=3)
        # FirstDf=FirstDf.dropna()
        # define figure size
        sns.set(rc={"figure.figsize": (10, 8)})  # width=6, height=5
        # Draw a vertical boxplot grouped
        # by a categorical variable:
        ###MEthod2#####
        # fig, ax = plt.subplots(1, sharex=False, sharey=False, gridspec_kw={'hspace': 0}, figsize=(10, 5))
        # sns.boxplot(x="Window", y="Differences", hue="Techniques", data=df, palette="PRGn")
        # [ax.axvline(x, color='black', linestyle='--') for x in [0.5,1.5,2.5]]
        # plt.show()
        # sns.lineplot(x="WindowCount", y="HRDifference", hue="Techniques", data=FirstDf)
        # plt.savefig(fullPathFiles + "windowLinefig.jpg")
        #
        # plt.switch_backend('agg')
        # plt.ioff()
        # plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        sns.set(font_scale=1.5)  # Overaall font size
        plt.switch_backend('agg')
        plt.ioff()
        plt.rcParams.update({'figure.max_open_warning': 0})
        plt.clf()
        #
        # sns.set_style('whitegrid')
        # sns.set(rc={"figure.figsize": (10, 8)})  # width=3, #height=4
        # ax = sns.boxplot(x='Techniques', y='HRDifference',  data=FirstDf)#rotation=45, horizontalalignment='right',
        # # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        # ax = sns.stripplot(x="Techniques", y="HRDifference", data=FirstDf)
        # # plt.savefig(self.objConfig.DiskPath + "BoxPlotCSV\\BoxPlotImages\\"+ filename+ ".jpg")
        # # plt.savefig(self.objConfig.DiskPath + "BoxPlotCSV\\FinalBestCases\\"+ filename+ ".jpg")
        # plt.savefig(fullPathFiles + "windowLinefig2.jpg")
        ###MEthod1#####
        sns.set_style('whitegrid')

        plt.figure(figsize=(14, 9))  # this creates a figure 8 inch wide, 4 inch high
        ax = sns.boxplot(x='WindowCount', y='HRDifference', hue="Techniques", data=FirstDf)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax = sns.stripplot(x="WindowCount", y="HRDifference", hue="Techniques", data=FirstDf,
                           palette="Set2")  # , size=6, marker="D",edgecolor="gray", alpha=.25
        handles, labels = ax.get_legend_handles_labels()
        # plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        [ax.axvline(x, color='black', linestyle='--') for x in [0.5, 1.5, 2.5]]

        plt.title("title", size=18)
        plt.tight_layout()
        plt.savefig(fullPathFiles+"windowfigTest.jpg")
        a=0

        #########method3
        # sns.stripplot(x="WindowCount", y="HRDifference", hue="Techniques",
        #               data=FirstDf, jitter=True,
        #               palette="Set2", split=True, linewidth=1, edgecolor='gray')
        #
        # # Get the ax object to use later.
        # ax = sns.boxplot(x="WindowCount", y="HRDifference", hue="Techniques",
        #                  data=FirstDf, palette="Set2", fliersize=0)
        #
        # # Get the handles and labels. For this example it'll be 2 tuples
        # # of length 4 each.
        # handles, labels = ax.get_legend_handles_labels()
        #
        # # When creating the legend, only use the first two elements
        # # to effectively remove the last two.
        # l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.savefig(fullPathFiles+"windowfig4.jpg")

# Execute method to get filenames which have good differnce
# AcceptableDifference = 3 # Max Limit of acceptable differnce
objFilterData = GeneratedDataFiltering()#Europe_WhiteSkin_Group,OtherAsian_OtherSkin_Group,SouthAsian_BrownSkin_Group
# objFilterData.AcceptableDifference = 10
##OLD ROUGH
# objFilterData.Run(AcceptableDifference)
# objFilterData.RunAllCaseParticipantwise() S
# objFilterData.RunAllCaseParticipantwiseCaseasCol()## RUN THIS TO GENERATE CSV FOR CASE
# objFilterData.RunParticipantWiseAll()
# objFilterData.GetBestFilesForParticipantEntireSingal(objFilterData.AcceptableDifference)
# objFilterData.getEmptyResults()
# Only run after best files are generated
# objFilterData.processBestResults("E:\\ARPOS_Server_Data\\Server_Study_Data\\Europe_WhiteSkin_Group\\Result\\","BestDataFiles") #"E:\\StudyData\\Result\\BestDataFiles_Resting1.txt"

###GETING BEST FILES AMONG RESTING 1 AND 2
# objFilterData.GetBestFilesForParticipantWindowAllSingal(objFilterData.AcceptableDifference,'PIS-8073','TimePlot')

###Generate csv with cases for positon for all particiipants
# objFilterData.getBlankCases('AfterExcersize')
# objFilterData.RunCasewiseFromFolderCustomCases('Resting1')#TODO: RUN FOR OTHER SKIN TYPES
# objFilterData.RunCasewiseFromFolderCustomCases('Resting2')#TODO: RUN FOR OTHER SKIN TYPES
# objFilterData.RunCasewiseFromFolderCustomCases('AfterExcersize')#TODO: RUN FOR OTHER SKIN TYPES
# objFilterData.PlotfromCustomCases(False)
# objFilterData.RunCasewiseFromFolder('Resting1',True)#TODO: RUN FOR OTHER SKIN TYPES
# objFilterData.RunCasewiseFromFolder('Resting2',True)
# objFilterData.RunCasewiseFromFolder('AfterExcersize',True)
# objFilterData.getfromDataFrameCSV('Resting1',False)
# objFilterData.RunCasewise('Resting1',True) # Generate cases for differnec TODO: Find cases common among all resting1, 2 and after exc
# objFilterData.RunCasewisebyParticipant('Resting1',True,'PIS-4014') # Generate cases for differnec TODO: Find cases common among all resting1, 2 and after exc
# objFilterData.RunCasewise('AfterExcersize',True) # Generate cases for differnec TODO: Find cases common among all resting1, 2 and after exc
# objFilterData.getBestCasesFromCSV('Resting2')# to get bext cases
# objFilterData.getBestCasesFromCSV('Resting1')# to get bext cases
# objFilterData.getBestCasesFromCSV('AfterExcersize')# to get bext cases
# objFilterData.WriteCases()
####PLOTSS
# objFilterData.PlotComparisonMethodMean()
# objFilterData.LinePlot() # FOR PERFORMANCE TIME LOG
# objFilterData.TestBoxPlot()# enitere signal
# objFilterData.TestBoxPlotWindow()#WINDOWs
# objFilterData.AllPlotsforComputedResults()
# objFilterData.PlotSignal() # to plot graph
#Parameters

# NotUpSapmled =[9406,10414,9154,9658,9910,9532,9763,3837,9091,3333,10540,3585,9175,9616,9259,4341,
#                9343,10351,9847,9112,9931,3081,9427,9784,9511,10435,10036,10519,4299,9280,9595,
#                9364,9679,9868,10372,10372]
# UpSampled='0'

# UpSampledList = [9406,10414,9154,9658,9910,9532,10540,9112,9175,9616,9931,9280,10036,9364,9427,9868,9679,9784,10372,10435,9091,10351,9595,9343,9847]
# UpSampled='1'

# UpSampledList= []
# NotUpsampledList = []
UpSampled =0
#
# loadedlsit=objFilterData.objFile.ReaddatafromFile('C:\\Users\\pp62\\PycharmProjects\\ARPOSProject\\BestCaseTechniqueIds\\','SouthAsianNOTupSampledCases')
#
# for item in loadedlsit:
#     NotUpsampledList.append(item.replace('\n',''))
#
# print(NotUpsampledList)

objSQLConfig = SQLConfig()
#objSQLConfig,skinGroup,PreProcess,FFT,Filter,Result,Smoothen,HeartRateStatus,type
# objFilterData.PlotFromSQLQueryFinal(objSQLConfig,'SouthAsian_BrownSkin_Group','5','M3','5', '3','False','Resting1','SPODifference')
# for hrstatus in objFilterData.objConfig.hearratestatus:
#     for techid in NotUpsampledList:
#         # objFilterData.PlotFromSQLQueryFinal(objSQLConfig,'SouthAsian_BrownSkin_Group','HeartRateDifference',str(techid),'Resting1',UpSampled)
#         objFilterData.PlotFromSQLQuerySouthAsian( objSQLConfig, 'SouthAsian_BrownSkin_Group', 'HeartRateDifference', str(techid),hrstatus,  UpSampled, 1)

# objFilterData.PlotFromSQLQuerySouthAsian( objSQLConfig, 'SouthAsian_BrownSkin_Group', 'HeartRateDifference', str(12048),'Resting1',  UpSampled, 1)
# objFilterData.PlotFromSQLQuerySouthAsian( objSQLConfig, 'SouthAsian_BrownSkin_Group', 'HeartRateDifference', str(12048),'Resting2',  UpSampled, 1)

# for hrstatus in objFilterData.objConfig.hearratestatus:
#     objFilterData.PlotFromSQLQueryAll( objSQLConfig, 'SPODifference', str(9406),hrstatus,  UpSampled, 1)
#     objFilterData.PlotFromSQLQueryAll( objSQLConfig, 'SPODifference', str(9154),hrstatus,  UpSampled, 1)
#     objFilterData.PlotFromSQLQueryAll( objSQLConfig, 'SPODifference', str(10414),hrstatus,  UpSampled, 1)
#     objFilterData.PlotFromSQLQueryAll( objSQLConfig, 'SPODifference', str(9658),hrstatus,  UpSampled, 1)
#     objFilterData.PlotFromSQLQueryAll( objSQLConfig, 'SPODifference', str(9910),hrstatus,  UpSampled, 1)
# objFilterData.PlotFromSQLQueryGetUpSampledVSNotSampledDataSpecific(objSQLConfig, str(9406),'Resting1',  UpSampled, 1)
objFilterData.PlotFromSQLTimeAll(objSQLConfig, str(9406),'Resting2',  UpSampled, 1)#objSQLConfig,TechniqueId,HeartRateStatus,UpSampled,AttemptType
