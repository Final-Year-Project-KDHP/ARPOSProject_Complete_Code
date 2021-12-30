import glob
import os

import numpy as np
import pandas as pd

import CommonMethods
from Configurations import Configurations
from FileIO import FileIO
from boxPlotMethodComparision import BoxPlot


class GeneratedDataFiltering:

    objConfig = None
    objFile = None
    AcceptableDifference =3
    #Constructor
    def __init__(self, skinGroup='None'):
        self.objConfig = Configurations(True, skinGroup)
        self.objFile = FileIO()

    def Getdata(self,fileName,AcceptableDifference,participant_number,position): #"None", loadpath,methodtype
        filepath = self.objConfig.DiskPath + 'Result\\' + participant_number + '\\' + position + '\\' + fileName + '\\' + 'HeartRate_' + fileName + '.txt' #HeartRate_FastICA_FFT-M1_FL-6_RS-1_PR-1_SM-False
                   # + algotype + '\\' + filename + "-"+ methodtype+ "_Fl_"+ str(filtertype) + "_Rs_" + str(resulttype)  + "_Pr_" + str(processtype)+ "_Sm_" + str(isSmooth)+ ".txt" #HRdata-M1_1_1
        # HRdata-M1_Fl_1_Rs_1_Pr_1_Sm_False

        Filedata = open(filepath, "r")
        data =Filedata.read().split("\n")
        generatedresult = []
        isAcceptableData =False
        diffValueList = []
        AcceptabledifferenceValue = -20
        for row in data:
            dataRow = row.split(",\t")
            # 0 index for windowCount, 1 index for GroundTruth, 2 index for generatedresult by arpos, 3 index for diference
            differenceValue = int(dataRow[2]) #change index here
            #  if 5 <= 7 and 5 >= -7:
            negativeAcceptableDifference = -1 * (AcceptableDifference)
            if( differenceValue <= AcceptableDifference and differenceValue >= negativeAcceptableDifference):#((differenceValue >= AcceptableDifference) ): #or (differenceValue <= negativeAcceptableDifference )
                isAcceptableData =True
                AcceptabledifferenceValue= differenceValue
            else:
                isAcceptableData = False
                break

            # isAcceptableData =True

        if(isAcceptableData):
            # filename = Filedata.name.split("HRdata-")
            generatedresult.append(fileName)
            diffValueList.append(AcceptabledifferenceValue)

        return generatedresult,isAcceptableData,diffValueList

    def CompareFiles(self,ListFilesP1,ListFilesP2):#participantnumber1, participantnumber2,
        # full_path_Save_P1 = "E:\\StudyData\\Result\\" + participantnumber1 + "\\Resting1\\BestDataFiles\\"
        # full_path_Save_P2 = "E:\\StudyData\\Result\\" + participantnumber2 + "\\Resting1\\BestDataFiles\\"

        # HRdata-M1_Fl_1_Rs_1_Pr_1_Sm_False
        generatedresult = []
        for row in ListFilesP1:
            Row1_P1_Value = row
            for row2 in ListFilesP2:
                if(Row1_P1_Value == row2):
                    generatedresult.append(row)

        # using list comprehension
        # to remove duplicated
        # from list
        result = []
        [result.append(x) for x in generatedresult if x not in result]

        for value in result:
            print(value)

    def processBestResults(self,filepath,filename):
        for position in self.objConfig.hearratestatus:
            # dataFile = filepath + filename
            dataFile = filepath+filename +"_"+ position+".txt"
            # SPOFile = filepath + "SPO.txt"

            # read data from files
            HrFiledata = open(dataFile, "r")
            # SPOFiledata = open(SPOFile, "r")

            HrFileNames = HrFiledata.read().split("\n")
            # SpoGr = SPOFiledata.read().split("\n")
            HrFiledata.close()

            _FastICA =[]
            _None = []
            _PCA =[]
            _ICAPCA =[]
            _Jade =[]

            for item in HrFileNames:
                if(item.__contains__("FastICA_")):
                    Filename = item.replace("FastICA_","")
                    _FastICA.append(Filename)            # FastICA_ # M1_Fl_1_Rs_1_Pr_3_Sm_True.txt
                elif(item.__contains__("None")):
                    Filename = item.replace("None_","")
                    _None.append(Filename)            # FastICA_ # M1_Fl_1_Rs_1_Pr_3_Sm_True.txt
                elif(item.__contains__("ICAPCA_")):
                    Filename = item.replace("ICAPCA_","")
                    _ICAPCA.append(Filename)            # FastICA_ # M1_Fl_1_Rs_1_Pr_3_Sm_True.txt
                elif(item.__contains__("PCA_")): #MATCHES WITH ICA PCA So check in end
                    Filename = item.replace("PCA_","")
                    _PCA.append(Filename)            # FastICA_ # M1_Fl_1_Rs_1_Pr_3_Sm_True.txt
                elif(item.__contains__("Jade_")):
                    Filename = item.replace("Jade_","")
                    _Jade.append(Filename)

            #Getcommon filenames only
            result = []
            [result.append(x) for x in _None if x not in result]
            [result.append(x) for x in _FastICA if x not in result]
            [result.append(x) for x in _ICAPCA if x not in result]
            [result.append(x) for x in _PCA if x not in result]

            result.sort()
            full_path_Save = self.objConfig.DiskPath + 'Result\\'
            # Wirte data
            RHr = open(full_path_Save + "BestCommonFiles" +  "_"+ position+".txt", "w+")

            for item in result:
                RHr.write(item + "\n")

            RHr.close()


    def Run(self,AcceptableDifference):
        fullistResting1=[]
        fullistResting2=[]
        fullistAfterExcersize=[]
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
                loadpath = self.objConfig.DiskPath + 'Result\\'+ participant_number + '\\' + position + '\\'
                datalist=[]
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
                                        generatedresult,isAcceptableData,diffValueList =self.Getdata(fileName,AcceptableDifference,participant_number,position)
                                        # generatedresult,isAcceptableData =self.Getdata(algoType, loadpath,fftype,"HRdata",filtertype,resulttype,AcceptableDifference,preprocesstype,isSmooth )
                                        # filepath = loadpath  + algotype + '\\' + filename + "-"+ methodtype+ "_"+ filtertype + "_" + resulttype+ ".txt" #HRdata-M1_1_1
                                        if(isAcceptableData):
                                            datalist.append(generatedresult)
                                            datalistValues.append(diffValueList)


                # datalist = np.array(datalist)
                if(position == "Resting1"):
                    Resting1List =datalist
                    Resting1ListValues = datalistValues
                elif(position == "Resting2"):
                    Resting2List =datalist
                    Resting2ListValues = datalistValues
                else:
                    AfterExcList =datalist
                    AfterExcListValues = datalistValues

            fullistResting1 =fullistResting1 + Resting1List
            fullistResting2 =fullistResting2 + Resting2List
            fullistAfterExcersize =fullistAfterExcersize + AfterExcList

            fullistResting1Values =fullistResting1Values + Resting1ListValues
            fullistResting2Values =fullistResting2Values + Resting2ListValues
            fullistAfterExcersizeValues =fullistAfterExcersizeValues + AfterExcListValues
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
        #Wirte data
        RHr = open(full_path_Save +'Result\\' + "BestDataFiles_Resting1.txt", "w+")
        RHr2 = open(full_path_Save+'Result\\' + "BestDataFiles_Resting2.txt", "w+")
        RHr3 = open(full_path_Save +'Result\\'+ "BestDataFiles_AfterExcersize.txt", "w+")

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

        #set(a).intersection(b, c)

    """
    Gemerate cases:
    """
    def GenerateCases(self):
        CaseList = []
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
                                CaseList.append(fileName)
        return  CaseList

    def getCase(self,fileName,participant_number,position):
        SavePath = self.objConfig.DiskPath + '\\Result\\' + participant_number + '\\' + position + '\\' + fileName + '\\'
        filepath = SavePath + 'HeartRate_' + fileName + '.txt' #HeartRate_FastICA_FFT-M1_FL-6_RS-1_PR-1_SM-False

        pathExsists = self.objFile.FileExits(SavePath + 'HeartRate_' + fileName + ".txt")
        data =None
        # already generated
        if (pathExsists):
            Filedata = open(filepath, "r")
            data = Filedata.read().split("\n")[0]
            Filedata.close()
        return data

    def IsAcceptableDifference(self,differenceValue): #"None", loadpath,methodtype
        #  if 5 <= 7 and 5 >= -7:
        negativeAcceptableDifference = -1 * (self.AcceptableDifference)
        if(differenceValue <= self.AcceptableDifference and differenceValue >= negativeAcceptableDifference):
            return True
        else:
            return False

    def splitDataRow(self,DataRow,participant_number,position): #"None", loadpath,methodtype
        dataRow = DataRow.split(",\t")
        # 0 index for GroundTruth, 1 index for generatedresult by arpos, 2 index for diference
        generatedResult = int(dataRow[1]) #change index here
        HrGr, SpoGr = CommonMethods.GetGroundTruth(participant_number, position,self.objConfig.DiskPath, int(60))
        # avgGroundTruthStored = int(dataRow[0])
        avgGroundTruth = np.mean(HrGr)
        # if(avgGroundTruthStored != round(avgGroundTruth)):
        #     differenceValue=int(dataRow[2])
        # else:
        differenceValue = avgGroundTruth - generatedResult
        return differenceValue

    def RunCasewise(self):
        CaseList = self.RunParticipantWiseAll() #GET Directories or generate below
        # CaseList = self.GenerateCases() # or generate

        df1 = pd.DataFrame({
            'CaseNames': CaseList
        })

        for participant_number in self.objConfig.ParticipantNumbers:
            df1[participant_number] = None
        # print(df1)
        for position in self.objConfig.hearratestatus:
            RowIndex = 0 #
            for case in CaseList:
                ColumnIndex = 1  # so doesntt replace case names
                for participant_number in self.objConfig.ParticipantNumbers:
                    #for position in self.objConfig.hearratestatus:
                    CaseData = self.getCase(case, participant_number, position)
                    if(CaseData != None):
                        differenceVal = self.splitDataRow(CaseData, participant_number, position)  # ROW,COlum
                        isAcceptable = self.IsAcceptableDifference(differenceVal)
                        if(isAcceptable):
                            df1.iloc[RowIndex, ColumnIndex] = differenceVal
                    else:
                        df1.iloc[RowIndex, ColumnIndex] = 'NotGenerated'
                    # else:
                    #     df1.iloc[RowIndex, ColumnIndex] = None
                    ColumnIndex = ColumnIndex +1
                    # print(df1)
                RowIndex = RowIndex +1

            # write dataFrame to SalesRecords CSV file
            df1.to_csv("E:\\ARPOS_Server_Data\\Server_Study_Data\\Europe_WhiteSkin_Group\\Result\\PIResults_" +position + ".csv")
        t=0

    """
       LoadFiles:
       Load file from path
       """

    def LoadFiles(self, filepath):
        CaseList=[]
        for path, subdirs, files in os.walk(filepath):
            for filename in subdirs:
                if(filename not in CaseList):
                    CaseList.append(filename)
                # a.write(str(f) + os.linesep)
        return CaseList

    def RunParticipantWiseAll(self):
        CaseList= []
        for participant_number in self.objConfig.ParticipantNumbers:
            for position in self.objConfig.hearratestatus:
                CaseSublist = []

                loadpath = self.objConfig.DiskPath + 'Result\\' + participant_number + '\\' + position + '\\'
                print(loadpath)
                CaseSublist = self.LoadFiles(loadpath)

                for name in CaseSublist:
                    if(name not in CaseList):
                        CaseList.append(name)


        finallist = 0
        return  CaseList



                    # file = open(loadpath + filename + ".txt", "r")
                    # Lines = file.readlines()
                    # file.close()


# Execute method to get filenames which have good differnce
# AcceptableDifference = 3 # Max Limit of acceptable differnce
objFilterData = GeneratedDataFiltering('Europe_WhiteSkin_Group')
objFilterData.AcceptableDifference = 8
# objFilterData.Run(AcceptableDifference)
objFilterData.RunCasewise()
# objFilterData.RunParticipantWiseAll()

# Only run after best files are generated
# objFilterData.processBestResults("E:\\ARPOS_Server_Data\\Server_Study_Data\\Europe_WhiteSkin_Group\\Result\\","BestDataFiles") #"E:\\StudyData\\Result\\BestDataFiles_Resting1.txt"
