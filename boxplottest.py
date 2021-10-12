# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
import sys
#
def Getdata(algotype,loadpath,methodtype,filename): #"None", loadpath,methodtype
    filepath = loadpath  + algotype + '\\' + filename + methodtype + ".txt"

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

DiskPath = "E:\\StudyData\\" ## change path here for uncompressed dataset

ticks = [ 'None','FastICA', 'ICAPCA', 'PCA', 'Gr']

fftTypeList = ["M1","M2", "M3", "M4"]

roiregions = ["lips", "forehead", "leftcheek", "rightcheek"]

ParticipantNumbers= ["PIS-2212","PIS-3807","PIS-4497","PIS-2169"]
#PIS-1118, PIS-6888, "PIS-9219"
hearratestatus = ["AfterExcersize","Resting1","Resting2"]


def Genrateboxplot(data,savepath,methodtype):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    # Creating axes instance
    bp = ax.boxplot(data, patch_artist=True,
                    notch='True', vert=0)

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
    ax.set_yticklabels(['none', 'ica',
                        'icapca', 'pca', 'gr'])

    # Adding title
    plt.title("Customized box plot")

    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # show plot
    path =savepath + "boxplot-" + methodtype + ".png"
    plt.savefig(path)  # show()

def RunBoxplotgeneration():
    for ParticipantNumber in ParticipantNumbers:
        print('Participant Id : ' + ParticipantNumber)

        for position in hearratestatus:

            for methodtype in fftTypeList:
                loadpath = DiskPath+  '\\Result\\' + ParticipantNumber + '\\' + position + '\\'

                data_d =Getdata("None", loadpath,methodtype,"GeneratedHRdata-" )
                data_a =Getdata("FastICA", loadpath,methodtype,"GeneratedHRdata-")
                data_b =Getdata("ICAPCA", loadpath,methodtype,"GeneratedHRdata-")
                data_c =Getdata("PCA", loadpath,methodtype,"GeneratedHRdata-")
                data_gr =Getdata("None", loadpath,methodtype,"GrHRdata-")

                data = [data_a, data_b, data_c,data_d,data_gr]

                Genrateboxplot(data,loadpath,methodtype)




RunBoxplotgeneration()
