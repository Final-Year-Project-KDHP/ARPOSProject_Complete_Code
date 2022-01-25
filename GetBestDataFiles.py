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

import CommonMethods
from Configurations import Configurations
from FileIO import FileIO
from SaveGraphs import Plots
from boxPlotMethodComparision import BoxPlot
import plotly.express as px


class GeneratedDataFiltering:
    objConfig = None
    objFile = None
    AcceptableDifference = 3

    # Constructor
    def __init__(self, skinGroup='None'):
        self.objConfig = Configurations(True, skinGroup)
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
        differenceValue = diffval# avgGroundTruth - generatedResult
        errorrate = (differenceValue / avgGroundTruth) * 100
        return errorrate

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

        CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-1_RS-2_SM-True')
        CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-1_RS-2_SM-False')
        CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-6_RS-2_SM-True')
        CustomCases.append('HRSPOwithLog_FastICA_PR-6_FFT-M1_FL-6_RS-2_SM-False')

        CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M1_FL-1_RS-2_SM-True')
        CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M1_FL-1_RS-2_SM-False')
        CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M1_FL-6_RS-2_SM-True')
        CustomCases.append('HRSPOwithLog_FastICA_PR-1_FFT-M1_FL-6_RS-2_SM-False')

        CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M1_FL-1_RS-2_SM-True')
        CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M1_FL-2_RS-2_SM-False')
        CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M1_FL-6_RS-2_SM-True')
        CustomCases.append('HRSPOwithLog_FastICA_PR-2_FFT-M1_FL-6_RS-2_SM-False')
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
        UpSampleData = True
        dfFinal = []
        for case in self.CaseList:
            for participant_number in self.objConfig.ParticipantNumbers:
                self.objConfig.setSavePath(participant_number, position, 'ProcessedData')
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
        dfFinal.to_csv(self.objConfig.DiskPath + "CaseWiseFromFolder_Results_" + position + "_CustomCasesNew_UpSampleData-" + str(UpSampleData) + ".csv")


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
        df = pd.read_csv("E:\\TestResultWindow.csv")  # read file
        # Draw a vertical boxplot grouped
        # by a categorical variable:
        df.head()
        ###MEthod2#####
        # fig, ax = plt.subplots(1, sharex=False, sharey=False, gridspec_kw={'hspace': 0}, figsize=(10, 5))
        # sns.boxplot(x="Window", y="Differences", hue="Techniques", data=df, palette="PRGn")
        # [ax.axvline(x, color='black', linestyle='--') for x in [0.5,1.5,2.5]]
        # plt.show()
        ###MEthod1#####
        sns.set_style('whitegrid')
        ax = sns.boxplot(x='Window', y='Differences', hue="Techniques", data=df)
        ax = sns.stripplot(x="Window", y="Differences", hue="Techniques", data=df,
                           palette="Set2")  # , size=6, marker="D",edgecolor="gray", alpha=.25
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        [ax.axvline(x, color='black', linestyle='--') for x in [0.5, 1.5, 2.5]]
        plt.show()

        #########method3
        # sns.stripplot(x="Window", y="Differences", hue="Techniques",
        #               data=df, jitter=True,
        #               palette="Set2", split=True, linewidth=1, edgecolor='gray')
        #
        # # Get the ax object to use later.
        # ax = sns.boxplot(x="Window", y="Differences", hue="Techniques",
        #                  data=df, palette="Set2", fliersize=0)
        #
        # # Get the handles and labels. For this example it'll be 2 tuples
        # # of length 4 each.
        # handles, labels = ax.get_legend_handles_labels()
        #
        # # When creating the legend, only use the first two elements
        # # to effectively remove the last two.
        # l = plt.legend(handles[0:2], labels[0:2], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        # plt.show()

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

# Execute method to get filenames which have good differnce
# AcceptableDifference = 3 # Max Limit of acceptable differnce
objFilterData = GeneratedDataFiltering('Europe_WhiteSkin_Group')#Europe_WhiteSkin_Group,OtherAsian_OtherSkin_Group,SouthAsian_BrownSkin_Group
objFilterData.AcceptableDifference = 10
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
objFilterData.RunCasewiseFromFolderCustomCases('Resting1')#TODO: RUN FOR OTHER SKIN TYPES
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
# objFilterData.PlotSignal() # to plot graph
