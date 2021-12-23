import numpy as np
import pandas as pd

GrPath = "E:\\ARPOS_Server_Data\\Server_Study_Data\\Europe_WhiteSkin_Group\\GroundTruthData\\"#E:\\StudyData\\GroundTruthData\\
# data = pd.read_excel(file)  # reading file
# import xlrd
# xls = xlrd.open_workbook(file, on_demand=True)
# print xls.sheet_names()
# print(data)


def GetGroundTruth(directorypath,PiNo,position):
    filepath = directorypath + "\\" + PiNo + "\\" + position + "\\"
    HrFile =  filepath + "HR.txt"
    SPOFile =  filepath + "SPO.txt"

    #read data from files
    HrFiledata = open(HrFile, "r")
    SPOFiledata = open(SPOFile, "r")

    HrGr =HrFiledata.read().split("\n")
    SpoGr =SPOFiledata.read().split("\n")

    return HrGr, SpoGr


hrgrlist, spogrlist = GetGroundTruth(GrPath, "PIS-2212", 'Resting1')

hrgrlist = [int(value) for value in hrgrlist]
spogrlist = [int(value) for value in spogrlist]

hrgr = np.average(hrgrlist)
spogr = np.average(spogrlist)

print(round(hrgr))
print(round(spogr))