#ARPOS program files
import os

from LoadWriteImageData import LoadWriteROI
from LoadWriteImageDataIR import LoadWriteIRROI
import numpy as np

def PrintText(detail):
    print(f'Status: , {detail}')  # Press Ctrl+F8 to toggle the breakpoint.

##### ONLY RUN First Time for loading and cropping data ######
##### START ######
# filepath = r"PIS-001/Resting1/"
# Savefilepath = r"PIS-001/Resting1Cropped/"
#
# PrintText("Process started for, Loading and Cropping all Color files...")
# #Run for cropping Color
# objLoadCrop = LoadWriteROI()
# objLoadCrop.LoadFiles(filepath + 'Color',Savefilepath+ 'Color')
#
# PrintText("Process started for, Loading and Cropping all IR files...")
# # Run for cropping IR
# objLoadIRCrop = LoadWriteIRROI()
# objLoadIRCrop.LoadandCropFiles(filepath + 'IR',Savefilepath+ 'IR')
# ##### END ######

def CropandLoadData(position,ParticipantNumber):
    ##### ONLY RUN First Time for loading and cropping data ######
    ##### START ######
    # roiregions = ["lips", "forehead", "leftcheek", "rightcheek"]
    # roiReg = roiregions[3]
    #
    previouspath = "E:\\StudyData\\UnCompressed" #os.path.split(os.getcwd())[0]
    #os.listdir(a)
    filepath = previouspath + "\\" + ParticipantNumber + "\\"+ position + "\\"
    Savefilepath = previouspath + "\\" + ParticipantNumber + "\\"+ position + "Cropped\\"

    if not os.path.exists(Savefilepath):
        os.makedirs(Savefilepath)

    PrintText("Process started for, Loading and Cropping all Color files...")
    # #Run for cropping Color
    objLoadCrop = LoadWriteROI()
    objLoadCrop.LoadFiles(filepath + 'Color',Savefilepath+ 'Color')
    #
    PrintText("Process started for, Loading and Cropping all IR files...")
    # # Run for cropping IR
    objLoadIRCrop = LoadWriteIRROI()
    objLoadIRCrop.LoadandCropFiles(filepath + 'IR',Savefilepath+ 'IR')
    ##### END ######


#Either single
CropandLoadData("Resting1", "PIS-3186")
CropandLoadData("Resting2", "PIS-3186")
CropandLoadData("AfterExcersize", "PIS-3186")
CropandLoadData("Resting1", "PIS-6888")
CropandLoadData("Resting2", "PIS-6888")
CropandLoadData("AfterExcersize", "PIS-6888")
CropandLoadData("Resting1", "PIS-6729")
CropandLoadData("Resting2", "PIS-6729")
CropandLoadData("AfterExcersize", "PIS-6729")
# CropandLoadData("Resting1", "PIS-6888")
# CropandLoadData("Resting1", "PIS-6729")

# or ALL
# positions = ["Resting2","AfterExcersize"] #, "forehead", "leftcheek", "rightcheek"
# folder = "E:\\StudyData\\UnCompressed\\"
# subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
# for folder in subfolders:
#     foldername = str(folder)
#     foldernameparams = foldername.split("\\")
#     ParticipantNumber = foldernameparams[3]
#
#     #for each position
#     for pos in positions:
#         # if(ParticipantNumber == "PIS-4709" and pos == "AfterExcersize"):
#         #     skip = 1
#         # else:
#         PrintText("Processing for : " + ParticipantNumber + " for " + pos)
#         CropandLoadData(pos, ParticipantNumber)

#
#
# def CropandLoadData(position,ParticipantNumber):
#     ##### ONLY RUN First Time for loading and cropping data ######
#     ##### START ######
#     # roiregions = ["lips", "forehead", "leftcheek", "rightcheek"]
#     # roiReg = roiregions[3]
#     #
#     previouspath = "C:\\inetpub\\wwwroot\\ARPOSApi\\StudyData\\UnCompressed" #os.path.split(os.getcwd())[0]
#     #os.listdir(a)
#     filepath = previouspath + "\\" + ParticipantNumber + "\\"+ position + "\\"
#     Savefilepath = previouspath + "\\" + ParticipantNumber + "\\"+ position + "Cropped\\"
#
#     if not os.path.exists(Savefilepath):
#         os.makedirs(Savefilepath)
#
#     PrintText("Process started for, Loading and Cropping all Color files...")
#     # #Run for cropping Color
#     objLoadCrop = LoadWriteROI()
#     objLoadCrop.LoadFiles(filepath + 'Color',Savefilepath+ 'Color')
#     #
#     PrintText("Process started for, Loading and Cropping all IR files...")
#     # # Run for cropping IR
#     objLoadIRCrop = LoadWriteIRROI()
#     objLoadIRCrop.LoadandCropFiles(filepath + 'IR',Savefilepath+ 'IR')
#     ##### END ######
#
# CropandLoadData("AfterExcersize", "PIS-001")
#
# positions = ["Resting1","Resting2","AfterExcersize"] #, "forehead", "leftcheek", "rightcheek"
# folder = "C:\\inetpub\\wwwroot\\ARPOSApi\\StudyData\\UnCompressed\\"
# subfolders = [ f.path for f in os.scandir(folder) if f.is_dir() ]
# for folder in subfolders:
#     foldername = str(folder)
#     foldernameparams = foldername.split("\\")
#     ParticipantNumber = foldernameparams[6]
#
#     #for each position
#     for pos in positions:
#         PrintText("Processing for : " + ParticipantNumber + " for " + pos)
#         CropandLoadData(pos, ParticipantNumber)
#
