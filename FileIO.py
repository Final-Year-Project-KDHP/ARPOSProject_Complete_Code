import os

class FileIO:

    def WriteListDatatoFile(self,savePath,fileName, dataList):
        self.CreatePath(savePath)

        file = open(savePath + fileName + ".txt", "w+")

        i = 0
        for item in dataList:
            DataRow = item.replace("\t", "").split(" ,")
            # windowc = DataRow[0]
            # grtruth = DataRow[1]
            # bpmval = DataRow[2]

            if (i == len(dataList) - 1):
                file.write(str(item))
            else:
                file.write(str(item) + '\n')
            i = i + 1

        file.close()

    """
    WritedatatoFile: Write data content to file
    """
    def WritedatatoFile(self,savePath,fileName, content):
        file = open(savePath + fileName + ".txt", "w+")
        file.write(str(content))
        file.close()

    """
    ReaddatafromFile: Read data content from file
    """
    def ReaddatafromFile(self,filePath,fileName):
        file = open(filePath + fileName + ".txt", "r")
        Lines = file.readlines()
        file.close()
        return Lines #.split('\n')

    """
    FileExits: check if file exists
    """
    def FileExits(self,path):
        fileExists = True
        #check file exists
        if not os.path.exists(path):
            fileExists = False

        return fileExists

    """
    CreatePath: create path if does not exist
    """
    def CreatePath(self,path):
        if not self.FileExits(path):
            os.makedirs(path)

    """
    getROIPath:
    Store all the generated ROIS
    """
    def getROIPath(self,participantNumber,position, UnCompressed_dataPath):
        #Read Uncompressed data from path
        ROI_dataPath = UnCompressed_dataPath + participantNumber + '\\' + position
        #Save ROI to path
        ROIPath= ROI_dataPath + 'Cropped\\'

        #Create ROI path if does not exist
        self.CreatePath(ROIPath)

        return ROIPath, ROI_dataPath + '\\'