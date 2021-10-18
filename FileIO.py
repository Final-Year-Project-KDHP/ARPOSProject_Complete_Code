import os


class FileIO:

    def WriteListDatatoFile(self,savePath,fileName, dataList):
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

    def CreatePath(self,path):
        if not os.path.exists(path):
            os.makedirs(path)