import os

import numpy as np


def Getdata(algotype,loadpath,methodtype,filename,filtertype,resulttype,AcceptableDifference,processtype,isSmooth): #"None", loadpath,methodtype
    filepath = loadpath  + algotype + '\\' + filename + "-"+ methodtype+ "_Fl_"+ str(filtertype) + "_Rs_" + str(resulttype)  + "_Pr_" + str(processtype)+ "_Sm_" + str(isSmooth)+ ".txt" #HRdata-M1_1_1
    # HRdata-M1_Fl_1_Rs_1_Pr_1_Sm_False

    Filedata = open(filepath, "r")
    data =Filedata.read().split("\n")
    generatedresult = []
    isAcceptableData =False
    for row in data:
        dataRow = row.split(",\t")
        # 0 index for windowCount, 1 index for GroundTruth, 2 index for generatedresult by arpos, 3 index for diference
        differenceValue = int(dataRow[3]) #change index here
        #  if 5 <= 7 and 5 >= -7:
        negativeAcceptableDifference = -1 * (AcceptableDifference)
        if( differenceValue <= AcceptableDifference and differenceValue >= negativeAcceptableDifference):#((differenceValue >= AcceptableDifference) ): #or (differenceValue <= negativeAcceptableDifference )
            isAcceptableData =True
        else:
            isAcceptableData = False
            break

        # isAcceptableData =True

    if(isAcceptableData):
        filename = Filedata.name.split("HRdata-")
        generatedresult.append(algotype + "_" + filename[1])

    return generatedresult,isAcceptableData


DiskPath = "E:\\StudyData\\Result\\" ## change path here for uncompressed dataset

ticks = [ 'None','FastICA', 'ICAPCA', 'PCA', 'Gr']

fftTypeList = ["M1","M2", "M3", "M4","M5","M6","M7"] #, "M4","M5", "M6" with butter filter try
filtertypeList = [1,3,2,4,5,6,7] #[3] run for 3,
preprocesses =[1,2,3,4,5,6, 7]
resulttypeList = [1,2,3,4,5,6,7,8,9]
Smoothen = [False,True]
AlgoList = [ "FastICA","ICAPCA","None","PCA"] #,"Jade"], SVD"Jade",

#Only 30 fps
ParticipantNumbers=  ["PIS-2212","PIS-4497","PIS-8308"]#,"PIS-1118""PIS-2212","PIS-4497","PIS-8308","PIS-8343" ,"PIS-8343"#beard
positionLists =  ["AfterExcersize"]#,"Resting2","AfterExcersize"

def CompareFiles(ListFilesP1,ListFilesP2):#participantnumber1, participantnumber2,
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

def processBestResults(filepath,filename):
    for position in positionLists:
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

        for item in HrFileNames:
            if(item.__contains__("FastICA_")):
                ica_Filename = item.replace("FastICA_","")
                _FastICA.append(ica_Filename)            # FastICA_ # M1_Fl_1_Rs_1_Pr_3_Sm_True.txt
            elif(item.__contains__("None")):
                ica_Filename = item.replace("None_","")
                _None.append(ica_Filename)            # FastICA_ # M1_Fl_1_Rs_1_Pr_3_Sm_True.txt
            elif(item.__contains__("ICAPCA_")):
                ica_Filename = item.replace("ICAPCA_","")
                _ICAPCA.append(ica_Filename)            # FastICA_ # M1_Fl_1_Rs_1_Pr_3_Sm_True.txt
            elif(item.__contains__("PCA_")): #MATCHES WITH ICA PCA So check in end
                ica_Filename = item.replace("PCA_","")
                _PCA.append(ica_Filename)            # FastICA_ # M1_Fl_1_Rs_1_Pr_3_Sm_True.txt

        #Getcommon filenames only
        result = []
        [result.append(x) for x in _None if x not in result]
        [result.append(x) for x in _FastICA if x not in result]
        [result.append(x) for x in _ICAPCA if x not in result]
        [result.append(x) for x in _PCA if x not in result]

        result.sort()
        full_path_Save = DiskPath
        # Wirte data
        RHr = open(full_path_Save + "BestCommonFiles" +  "_"+ position+".txt", "w+")

        for item in result:
            RHr.write(item + "\n")

        RHr.close()


def Run(AcceptableDifference):
    fullistResting1=[]
    fullistResting2=[]
    fullistAfterExcersize=[]

    Resting1Store = {}
    Resting2Store = {}
    AfterExcersizeStore = {}

    for participant_number in ParticipantNumbers:
        Resting1List = []
        Resting2List = []
        AfterExcList = []
        for position in positionLists:
            loadpath = DiskPath + participant_number + '\\' + position + '\\'
            datalist=[]
            for algoType in AlgoList:

                for fftype in fftTypeList:

                    for resulttype in resulttypeList:

                        for filtertype in filtertypeList:

                            for preprocesstype in preprocesses:

                                for isSmooth in Smoothen:

                                    generatedresult,isAcceptableData =Getdata(algoType, loadpath,fftype,"HRdata",filtertype,resulttype,AcceptableDifference,preprocesstype,isSmooth )
                                    # filepath = loadpath  + algotype + '\\' + filename + "-"+ methodtype+ "_"+ filtertype + "_" + resulttype+ ".txt" #HRdata-M1_1_1
                                    if(isAcceptableData):
                                        datalist.append(generatedresult)


            # datalist = np.array(datalist)
            if(position == "Resting1"):
                Resting1List =datalist
            elif(position == "Resting2"):
                Resting2List =datalist
            else:
                AfterExcList =datalist

        fullistResting1 =fullistResting1 + Resting1List
        fullistResting2 =fullistResting2 + Resting2List
        fullistAfterExcersize =fullistAfterExcersize + AfterExcList
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
    [Resting1result.append(x) for x in fullistResting1 if x not in Resting1result]

    Resting2result = []
    [Resting2result.append(x) for x in fullistResting2 if x not in Resting2result]

    AfterExcersizeresult = []
    [AfterExcersizeresult.append(x) for x in fullistAfterExcersize if x not in AfterExcersizeresult]
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

    full_path_Save = DiskPath
    #Wirte data
    RHr = open(full_path_Save + "BestDataFiles_Resting1.txt", "w+")
    RHr2 = open(full_path_Save + "BestDataFiles_Resting2.txt", "w+")
    RHr3 = open(full_path_Save + "BestDataFiles_AfterExcersize.txt", "w+")

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

    # for x in fullist:
    #     print(x)
    # CompareFiles(P1List,P2List) Manual comparison between two participants data list containing filenames showing data within acceptablediffernce
    # finallist = set(np.array(P1List)) & set(np.array(P2List)) #& set(c)
    # P1List =np.array(P1List)
    # P2List =np.array(P2List)
    # fullist = set(P1List).intersection(P2List)
    # fullist = np.intersect1d(P1List, P2List)

    #set(a).intersection(b, c)

# Execute method to get filenames which have good differnce
AcceptableDifference = 16 # Max Limit of acceptable differnce
Run(AcceptableDifference)

# Only run after best files are generated
processBestResults("E:\\StudyData\\Result\\","BestDataFiles") #"E:\\StudyData\\Result\\BestDataFiles_Resting1.txt"
#"Resting1",
