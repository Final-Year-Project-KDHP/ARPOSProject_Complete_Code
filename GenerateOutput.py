#File to generate output
import os

import numpy as np
import pandas as pd

algos =[ "FastICA","ICAPCA","None","PCA"] #,"Jade"], SVD"Jade",
fftmethods = ["M1","M2", "M3", "M4"]
imagenames = ["1-InitialFrameData.png","2-AfterICA.png","3-AfterFFT.png","4-AfterABSfft.png"]
algorithmList =[ "FastICA","ICAPCA","None","PCA"] #,"Jade"], SVD"Jade",
HRValueList = []
SPOValueList = []

def GetImagePaths(region, htmlfilepath,ImageName,alg,method,position):
    ImagePaths = []
    #for alg in algos:
        #for method in fftmethods:
            #for imgname in imagenames:
    image_path = str(htmlfilepath)  + position  + '\\' + str(region) + '\\' + alg+ '\\' + method+ "\\" + ImageName  + ".png" ## Loading path for color data
    ImagePaths.append(image_path)

    return ImagePaths

def GetImagePath_Single(region, htmlfilepath,ImageName,alg,method,position):
    image_path = str(htmlfilepath)    + '\\' + str(region) + '\\' + alg+ '\\' + method+ "\\" + ImageName   ## Loading path for color data
    return image_path

# Tabledata =[]

def GenerateCSS(htmlfile):
    htmlfile.write("<style>\n")
    htmlfile.write("table, th, td {\n")
    htmlfile.write("  border: 1px solid black;  text-align: center; \n")
    htmlfile.write("}\n")

    htmlfile.write("\n")
    #For images side by side
    htmlfile.write("#banner {\n")
    htmlfile.write("  display: block;\n")
    htmlfile.write("}\n")

    htmlfile.write(".images {\n")
    htmlfile.write("  display: inline-block;\n")
    htmlfile.write("  max-width: 20%;\n")
    htmlfile.write("  margin: 0 2.5%;\n")
    htmlfile.write("}\n")
    htmlfile.write("</style>\n")


def definebreaks(htmlfile):
    htmlfile.write('</br>')

def defineParagh(htmlfile, text):
    htmlfile.write('<p>' + text + '</p>')

def defineTableHeader(htmlfile, type):
    htmlfile.write('<table >')
    htmlfile.write('                <tr>')
    htmlfile.write('                    <td rowspan="2"><b>PI No</b></td>')
    htmlfile.write('                    <td colspan="2"><b>Frame Rate</b></td>')
    htmlfile.write('                    <td colspan="'+ str(len(algorithmList) +1) +'"><b>' + type + ' Heart Rate</b></td>')
    htmlfile.write('                    <td colspan="'+ str(len(algorithmList)+1) +'"><b>' + type + ' Blood Oxygen</b></td>')
    htmlfile.write('                    <td colspan="'+ str(len(algorithmList)) +'"><b>' + type + ' HR Difference</b></td>')
    htmlfile.write('                    <td colspan="'+ str(len(algorithmList)) +'"><b>' + type + ' SPO Difference</b></td>')
    htmlfile.write('                    <td rowspan="2"><b>Other Details</b></td>')
    htmlfile.write('                </tr>')
    htmlfile.write('                <tr>')
    htmlfile.write('                    <td><b>Color</b></td>')
    htmlfile.write('                    <td><b>Ir</b></td>')
    for algotype in algorithmList:
        htmlfile.write('                    <td><b>' + algotype + '</b></td>')
    # htmlfile.write('                    <td><b>None</b></td>')
    # htmlfile.write('                    <td><b>FastICA</b></td>')
    # htmlfile.write('                    <td><b>FastICAM2</b></td>')
    # htmlfile.write('                    <td><b>PCA</b></td>')
    # htmlfile.write('                    <td><b>Jade</b></td>')
    # htmlfile.write('                    <td><b>Other</b></td>')
    htmlfile.write('                    <td  style="background:#81af91; color:white;"><b>GT</b></td>')

    for algotype in algorithmList:
        htmlfile.write('                    <td><b>' + algotype + '</b></td>')
    # htmlfile.write('                    <td><b>None</b></td>')
    # htmlfile.write('                    <td><b>FastICA</b></td>')
    # htmlfile.write('                    <td><b>FastICAM2</b></td>')
    # htmlfile.write('                    <td><b>PCA</b></td>')
    # htmlfile.write('                    <td><b>Jade</b></td>')
    # htmlfile.write('                    <td><b>Other</b></td>')

    htmlfile.write('                    <td  style="background:#81af91; color:white;"><b>GT</b></td>')

    ##diferences
    for algotype in algorithmList:
        htmlfile.write('                    <td><b>' + algotype + '</b></td>')

    for algotype in algorithmList:
        htmlfile.write('                    <td><b>' + algotype + '</b></td>')

    htmlfile.write('                </tr>')


def defineTableFooter(htmlfile, type):
    htmlfile.write('            </table>')


# def GenerateTableRows(htmlfile, PIno, colorfps, irfps, noneHR, ICAhrM2, ICAhr, PCAhr, Jadehr, otherHr, GRhr, noneSpo,
#                       ICASpoM2, ICASpo, PCASpo, JadeSpo, otherSpo, GRSpo, desc):


def GenerateTableRows(htmlfile, PIno, colorfps, irfps, GRhr, GRSpo, desc):
    htmlfile.write('                <tr>')
    htmlfile.write('                    <td>' + str(PIno) + '</td>')
    htmlfile.write('                    <td>' + str(colorfps) + '</td>')
    htmlfile.write('                    <td>' + str(irfps) + '</td>')

    for algoVal in HRValueList:
        htmlfile.write('                    <td>' + str(algoVal) + '</td>')

    # htmlfile.write('                    <td>' + str(noneHR) + '</td>')
    # htmlfile.write('                    <td>' + str(ICAhr) + '</td>')
    # htmlfile.write('                    <td>' + str(ICAhrM2) + '</td>')
    # htmlfile.write('                    <td>' + str(PCAhr) + '</td>')
    # htmlfile.write('                    <td>' + str(Jadehr) + '</td>')
    # htmlfile.write('                    <td>' + str(otherHr) + '</td>')
    htmlfile.write('                    <td  style="background:#81af91; color:white;">' + str(GRhr) + '</td>')


    for algoVal in SPOValueList:
        htmlfile.write('                    <td>' + str(algoVal) + '</td>')

    # htmlfile.write('                    <td>' + str(noneSpo) + '</td>')
    # htmlfile.write('                    <td>' + str(ICASpo) + '</td>')
    # htmlfile.write('                    <td>' + str(ICASpoM2) + '</td>')
    # htmlfile.write('                    <td>' + str(PCASpo) + '</td>')
    # htmlfile.write('                    <td>' + str(JadeSpo) + '</td>')
    # htmlfile.write('                    <td>' + str(otherSpo) + '</td>')
    htmlfile.write('                    <td  style="background:#81af91; color:white;">' + str(GRSpo) + '</td>')

    ##Diferences

    for algoVal in HRValueList:
        diffhr = int(GRhr) - int(algoVal)
        htmlfile.write('                    <td>' + str(diffhr) + '</td>')


    for algoVal in SPOValueList:
        diffspo = int(GRSpo) - int(algoVal)
        htmlfile.write('                    <td>' + str(diffspo) + '</td>')

    htmlfile.write('                    <td>' + str(desc) + '</td>')
    htmlfile.write('                </tr>')

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

ParticipantNumbers = ["PIS-2212",  "PIS-3807", "PIS-4497",  "PIS-8308", "PIS-8343"]

def participantsSummary(htmlfilepath,GrPath,participantResults,PINOSlist,AlgoList):
    algorithmList = AlgoList
    ParticipantNumbers =PINOSlist
    htmlfile = open(htmlfilepath + r"ParicipantSummaryReport.html", "w+")
    htmlfile.write("<!DOCTYPE html>\n")
    htmlfile.write("<html>\n")
    htmlfile.write("<body>\n")
    htmlfile.write("<h1> Summary Result of all participants</h1>\n\n")

    GenerateCSS(htmlfile)

    ICAResults = []
    NoneResults = []
    PCAResults = []
    JadeResults =[]
    otherResults =[]

    # Creating an empty Dataframe with column names only
    dfObjResting1 = pd.DataFrame(columns=['Participant_Id', 'position', 'Algorithm', 'HRresult','SPOresult', 'Desc'])
    dfObjResting2 = pd.DataFrame(columns=['Participant_Id', 'position', 'Algorithm', 'HRresult','SPOresult', 'Desc'])
    dfObjAfterExc = pd.DataFrame(columns=['Participant_Id', 'position', 'Algorithm', 'HRresult','SPOresult', 'Desc'])

    for k, v in participantResults.items():
        participantid = k
        for item in v:
            #PINumber = item.PINumber
            HRresult = item.HRresult.split('-')
            SPOresult = item.SPOresult.split('-')
            position = item.position
            algorithm = item.alogrithm

            bestHeartRateSnr =HRresult [1]
            bestBpm =HRresult [3]
            channel =HRresult [5]
            hrregion =HRresult [7]

            oxygenstd = SPOresult[2]
            sperror = SPOresult[4]
            spregionToUse = SPOresult[6]
            SPO = SPOresult[8]

            Colorfps = 0
            Irfps = 0

            Otherdesc = "Heart rate SNR: "+ bestBpm + ", for channel: " + channel + ", for region: " + hrregion + "</br>SPO std: " + oxygenstd + " with error : " + sperror + ", for region : " + spregionToUse

            # Append rows in Empty Dataframe
            if(position == "Resting1"):
                dfObjResting1 = dfObjResting1.append({'Participant_Id': participantid, 'position': position, 'Algorithm': algorithm, 'HRresult': bestBpm, 'SPOresult': SPO, 'Desc': Otherdesc}, ignore_index=True)

            if (position == "Resting2"):
                dfObjResting2 = dfObjResting2.append(
                    {'Participant_Id': participantid, 'position': position, 'Algorithm': algorithm, 'HRresult': bestBpm,
                     'SPOresult': SPO, 'Desc': Otherdesc}, ignore_index=True)
            if (position == "AfterExcersize"):
                dfObjAfterExc = dfObjAfterExc.append(
                    {'Participant_Id': participantid, 'position': position, 'Algorithm': algorithm, 'HRresult': bestBpm,
                     'SPOresult': SPO, 'Desc': Otherdesc}, ignore_index=True)

        # HRdetail = "bestHeartRateSnr-" + str(bestHeartRateSnr) + "-bestBpm-" + str(bestBpm) + "-channel-" + str(channeltype) + "-region-" + str(regiontype)
        #SPOdetail = "-STD-" + str(oxygenstd) + "-error-" + str(smallestOxygenError) + "-region-" + regionToUse + "-SPO-" + str(finaloxy)

        #find all algo data

    type = "Resting 1"
    # Define table header
    defineTableHeader(htmlfile, type)

    # PIno = ""
    # colorfps = ""
    # irfps = ""
    # noneHR = ""
    # ICAhr = ""
    # PCAhr = ""
    # Jadehr = ""
    # otherHr = ""
    # GRhr = ""
    # noneSpo = ""
    # ICASpo = ""
    # PCASpo = ""
    # JadeSpo = ""
    # otherSpo = ""
    # GRSpo = ""
    # desc = ""

    for PIno in ParticipantNumbers: #dfObjResting1.iterrows():

        #####FOR RESTING1 #############

        for algoType in algorithmList:
            AlgoVal = dfObjResting1.loc[(dfObjResting1['Algorithm'] == str(algoType)) & (dfObjResting1['Participant_Id'] == PIno)]
            HRvalue = AlgoVal['HRresult'].values[0]
            SPOvalue = AlgoVal['SPOresult'].values[0]
            HRValueList.append(HRvalue)
            SPOValueList.append(SPOvalue)

        #RowICAMethod2 = dfObjResting1.loc[(dfObjResting1['Algorithm'] == "ICAMethod2") & (dfObjResting1['Participant_Id'] == PIno)]
        #RowICA = dfObjResting1.loc[(dfObjResting1['Algorithm'] == "ICA") & (dfObjResting1['Participant_Id'] == PIno)]
        #RowPCA = dfObjResting1.loc[(dfObjResting1['Algorithm'] == "PCA") & (dfObjResting1['Participant_Id'] == PIno)]
        #RowWithoutAlgorithm = dfObjResting1.loc[(dfObjResting1['Algorithm'] == "WithoutAlgorithm")  & (dfObjResting1['Participant_Id'] == PIno)]  # & (dfObjResting1['Algorithm'] == B)

        hrgrlist, spogrlist = GetGroundTruth(GrPath, PIno, 'Resting1')

        hrgrlist = [int(value) for value in hrgrlist]
        spogrlist = [int(value) for value in spogrlist]

        hrgr = round(np.average(hrgrlist))
        spogr = round(np.average(spogrlist))

        # GenerateTableRows(htmlfile, PIno, 0, 0, RowWithoutAlgorithm['HRresult'].values[0], RowICAMethod2['HRresult'].values[0],RowICA['HRresult'].values[0],RowPCA['HRresult'].values[0],
        #                   0,0,hrgr,
        #                   RowWithoutAlgorithm['SPOresult'].values[0],RowICAMethod2['SPOresult'].values[0],RowICA['SPOresult'].values[0],RowPCA['SPOresult'].values[0],
        #                   0,0,spogr,
        #                   RowICA['Desc'].values[0])
        GenerateTableRows(htmlfile, PIno, 0, 0, hrgr, spogr,'')
        HRValueList.clear()
        SPOValueList.clear()


         #row['Algorithm'], row['HRresult'],row['SPOresult']
        #df.loc[(df['column_name'] >= A) & (df['column_name'] <= B)]


    #print("Dataframe Contens ", dfObj, sep='\n')


    #define table footer
    defineTableFooter(htmlfile, type)

    #define breaks in document
    definebreaks(htmlfile)
    definebreaks(htmlfile)

    ###### ------------ RESTING 2 ----------------########
    defineTableHeader(htmlfile, 'Resting2')

    for PIno in ParticipantNumbers:
        #####FOR RESTING2 #############
        for algoType in algorithmList:
            AlgoVal = dfObjResting2.loc[(dfObjResting2['Algorithm'] == str(algoType)) & (dfObjResting2['Participant_Id'] == PIno)]
            HRvalue = AlgoVal['HRresult'].values[0]
            SPOvalue = AlgoVal['SPOresult'].values[0]
            HRValueList.append(HRvalue)
            SPOValueList.append(SPOvalue)
        # RowICAMethod2 = dfObjResting1.loc[(dfObjResting2['Algorithm'] == "ICAMethod2") & (dfObjResting1['Participant_Id'] == PIno)]
        # RowICA = dfObjResting2.loc[(dfObjResting2['Algorithm'] == "ICA") & (dfObjResting2['Participant_Id'] == PIno)]
        # RowPCA = dfObjResting2.loc[(dfObjResting2['Algorithm'] == "PCA") & (dfObjResting2['Participant_Id'] == PIno)]
        # RowWithoutAlgorithm = dfObjResting2.loc[(dfObjResting2['Algorithm'] == "WithoutAlgorithm") & (
        #         dfObjResting2['Participant_Id'] == PIno)]  # & (dfObjResting2['Algorithm'] == B)

        hrgrlist, spogrlist = GetGroundTruth(GrPath, PIno, 'Resting2')

        hrgrlist = [int(value) for value in hrgrlist]
        spogrlist = [int(value) for value in spogrlist]

        hrgr = round(np.average(hrgrlist))
        spogr = round(np.average(spogrlist))

        # GenerateTableRows(htmlfile, PIno, 0, 0, RowWithoutAlgorithm['HRresult'].values[0], RowICA['HRresult'].values[0],
        #                   RowPCA['HRresult'].values[0],
        #                   0, 0, hrgr,
        #                   RowWithoutAlgorithm['SPOresult'].values[0], RowICA['SPOresult'].values[0],
        #                   RowPCA['SPOresult'].values[0],
        #                   0, 0, spogr,
        #                   RowICA['Desc'].values[0])

        # GenerateTableRows(htmlfile, PIno, 0, 0, RowWithoutAlgorithm['HRresult'].values[0], RowICAMethod2['HRresult'].values[0],RowICA['HRresult'].values[0],RowPCA['HRresult'].values[0],
        #                   0,0,hrgr,
        #                   RowWithoutAlgorithm['SPOresult'].values[0],RowICAMethod2['SPOresult'].values[0],RowICA['SPOresult'].values[0],RowPCA['SPOresult'].values[0],
        #                   0,0,spogr,
        #                   RowICA['Desc'].values[0])
        GenerateTableRows(htmlfile, PIno, 0, 0, hrgr, spogr, '')
        HRValueList.clear()
        SPOValueList.clear()

    defineTableFooter(htmlfile, 'Resting2')


    #define breaks in document
    definebreaks(htmlfile)
    definebreaks(htmlfile)

    ###### ------------ AfterExcersize ----------------########
    defineTableHeader(htmlfile, 'AfterExcersize')

    for PIno in ParticipantNumbers:
        #####FOR AfterExcersize #############
        for algoType in algorithmList:
            AlgoVal = dfObjAfterExc.loc[(dfObjAfterExc['Algorithm'] == str(algoType)) & (dfObjAfterExc['Participant_Id'] == PIno)]
            HRvalue = AlgoVal['HRresult'].values[0]
            SPOvalue = AlgoVal['SPOresult'].values[0]
            HRValueList.append(HRvalue)
            SPOValueList.append(SPOvalue)
        # RowICAMethod2 = dfObjResting1.loc[(dfObjAfterExc['Algorithm'] == "ICAMethod2") & (dfObjResting1['Participant_Id'] == PIno)]
        # RowICA = dfObjAfterExc.loc[(dfObjAfterExc['Algorithm'] == "ICA") & (dfObjAfterExc['Participant_Id'] == PIno)]
        # RowPCA = dfObjAfterExc.loc[(dfObjAfterExc['Algorithm'] == "PCA") & (dfObjAfterExc['Participant_Id'] == PIno)]
        # RowWithoutAlgorithm = dfObjAfterExc.loc[(dfObjAfterExc['Algorithm'] == "WithoutAlgorithm") & (
        #         dfObjAfterExc['Participant_Id'] == PIno)]  # & (dfObjAfterExc['Algorithm'] == B)

        hrgrlist, spogrlist = GetGroundTruth(GrPath, PIno, 'AfterExcersize')

        hrgrlist = [int(value) for value in hrgrlist]
        spogrlist = [int(value) for value in spogrlist]

        hrgr = round(np.average(hrgrlist))
        spogr = round(np.average(spogrlist))

        # GenerateTableRows(htmlfile, PIno, 0, 0, RowWithoutAlgorithm['HRresult'].values[0], RowICA['HRresult'].values[0],
        #                   RowPCA['HRresult'].values[0],
        #                   0, 0, hrgr,
        #                   RowWithoutAlgorithm['SPOresult'].values[0], RowICA['SPOresult'].values[0],
        #                   RowPCA['SPOresult'].values[0],
        #                   0, 0, spogr,
        #                   RowICA['Desc'].values[0])

        # GenerateTableRows(htmlfile, PIno, 0, 0, RowWithoutAlgorithm['HRresult'].values[0], RowICAMethod2['HRresult'].values[0],RowICA['HRresult'].values[0],RowPCA['HRresult'].values[0],
        #                   0,0,hrgr,
        #                   RowWithoutAlgorithm['SPOresult'].values[0],RowICAMethod2['SPOresult'].values[0],RowICA['SPOresult'].values[0],RowPCA['SPOresult'].values[0],
        #                   0,0,spogr,
        #                   RowICA['Desc'].values[0])
        GenerateTableRows(htmlfile, PIno, 0, 0, hrgr, spogr, '')
        HRValueList.clear()
        SPOValueList.clear()

    defineTableFooter(htmlfile, 'AfterExcersize')


    htmlfile.write("</body>\n")
    # Document end
    htmlfile.write("</html>\n")
    htmlfile.close()
    print("Generated!")

def GenerateWindowResult(htmlfile,Lips_IR_bpm,Lips_Red_bpm,Lips_Green_bpm,Lips_Blue_bpm, Lips_Grey_bpm,
                         Forehead_IR_bpm,Forehead_Red_bpm,Forehead_Green_bpm,Forehead_Blue_bpm, Forehead_Grey_bpm,
                         RighCheek_IR_bpm,RighCheek_Red_bpm,RighCheek_Green_bpm,RighCheek_Blue_bpm, RighCheek_Grey_bpm,
                         LeftCheek_IR_bpm,LeftCheek_Red_bpm,LeftCheek_Green_bpm,LeftCheek_Blue_bpm, LeftCheek_Grey_bpm,
                         Lips_IR_SNR,Lips_Red_SNR,Lips_Green_SNR,Lips_Blue_SNR, Lips_Grey_SNR,
                         Forehead_IR_SNR,Forehead_Red_SNR,Forehead_Green_SNR,Forehead_Blue_SNR, Forehead_Grey_SNR,
                         RighCheek_IR_SNR,RighCheek_Red_SNR,RighCheek_Green_SNR,RighCheek_Blue_SNR, RighCheek_Grey_SNR,
                         LeftCheek_IR_SNR,LeftCheek_Red_SNR,LeftCheek_Green_SNR,LeftCheek_Blue_SNR, LeftCheek_Grey_SNR):
    htmlfile.write("<table >\n")
    htmlfile.write("                <tr>")
    htmlfile.write("                    <td colspan='5'><strong>Lips</strong></td>\n")
    htmlfile.write("                    <td colspan='5'><strong>Forehead</strong></td>\n")
    htmlfile.write("                    <td colspan='5'><strong>RightCheek</strong></td>\n")
    htmlfile.write("                    <td colspan='5'><strong>LeftCheek</strong></td>\n")
    htmlfile.write("                </tr>\n")
    htmlfile.write("                <tr>\n")
    htmlfile.write("                    <td>IR</td>\n")
    htmlfile.write("                    <td>Red</td>\n")
    htmlfile.write("                    <td>Green</td>\n")
    htmlfile.write("                    <td>Blue</td>\n")
    htmlfile.write("                    <td>Grey</td>\n")
    htmlfile.write("                    <td>IR</td>\n")
    htmlfile.write("                    <td>Red</td>\n")
    htmlfile.write("                    <td>Green</td>\n")
    htmlfile.write("                    <td>Blue</td>\n")
    htmlfile.write("                    <td>Grey</td>\n")
    htmlfile.write("                    <td>IR</td>\n")
    htmlfile.write("                    <td>Red</td>\n")
    htmlfile.write("                    <td>Green</td>\n")
    htmlfile.write("                    <td>Blue</td>\n")
    htmlfile.write("                    <td>Grey</td>\n")
    htmlfile.write("                    <td>IR</td>\n")
    htmlfile.write("                    <td>Red</td>\n")
    htmlfile.write("                    <td>Green</td>\n")
    htmlfile.write("                    <td>Blue</td>\n")
    htmlfile.write("                    <td>Grey</td>\n")
    htmlfile.write("                </tr>\n")
    htmlfile.write("                <tr>\n")
    htmlfile.write("                    <td>"+ str(Lips_IR_bpm) + "</td>\n")
    htmlfile.write("                    <td>"+ str(Lips_Red_bpm) + "</td>\n")
    htmlfile.write("                    <td>"+ str(Lips_Green_bpm) + "</td>\n")
    htmlfile.write("                    <td>"+ str(Lips_Blue_bpm) + "</td>\n")
    htmlfile.write("                    <td>"+ str(Lips_Grey_bpm) + "</td>\n")
    htmlfile.write(" <td>"+ str(Forehead_IR_bpm) + "</td>\n")
    htmlfile.write("         <td>"+ str(Forehead_Red_bpm) + "</td>\n")
    htmlfile.write("                      <td>"+ str(Forehead_Green_bpm) + "</td>\n")
    htmlfile.write("                   <td>"+ str(Forehead_Blue_bpm) + "</td>\n")
    htmlfile.write("                     <td>"+ str(Forehead_Grey_bpm) + "</td>\n")
    htmlfile.write(" <td>"+ str(RighCheek_IR_bpm) + "</td>\n")
    htmlfile.write("         <td>"+ str(RighCheek_Red_bpm) + "</td>\n")
    htmlfile.write("                      <td>"+ str(RighCheek_Green_bpm) + "</td>\n")
    htmlfile.write("                   <td>"+ str(RighCheek_Blue_bpm) + "</td>\n")
    htmlfile.write("                     <td>"+ str(RighCheek_Grey_bpm) + "</td>\n")
    htmlfile.write(" <td>"+ str(LeftCheek_IR_bpm) + "</td>\n")
    htmlfile.write("         <td>"+ str(LeftCheek_Red_bpm) + "</td>\n")
    htmlfile.write("                      <td>"+ str(LeftCheek_Green_bpm) + "</td>\n")
    htmlfile.write("                   <td>"+ str(LeftCheek_Blue_bpm) + "</td>\n")
    htmlfile.write("                     <td>"+ str(LeftCheek_Grey_bpm) + "</td>\n")
    htmlfile.write("                </tr>\n")
    htmlfile.write("                <tr>\n")
    htmlfile.write("                    <td colspan='5'><strong>SNR</strong></td>\n")
    htmlfile.write(" <td colspan='5'><strong>SNR</strong></td>\n")
    htmlfile.write(" <td colspan='5'><strong>SNR</strong></td>\n")
    htmlfile.write(" <td colspan='5'><strong>SNR</strong></td>\n")
    htmlfile.write("                </tr>\n")
    htmlfile.write("                <tr>\n")
    htmlfile.write("                    <td>IR</td>\n")
    htmlfile.write("                    <td>Red</td>\n")
    htmlfile.write("                    <td>Green</td>\n")
    htmlfile.write("                    <td>Blue</td>\n")
    htmlfile.write("                    <td>Grey</td>\n")
    htmlfile.write("                    <td>IR</td>\n")
    htmlfile.write("                    <td>Red</td>\n")
    htmlfile.write("                    <td>Green</td>\n")
    htmlfile.write("                    <td>Blue</td>\n")
    htmlfile.write("                    <td>Grey</td>\n")
    htmlfile.write("                    <td>IR</td>\n")
    htmlfile.write("                    <td>Red</td>\n")
    htmlfile.write("                    <td>Green</td>\n")
    htmlfile.write("                    <td>Blue</td>\n")
    htmlfile.write("                    <td>Grey</td>\n")
    htmlfile.write("                    <td>IR</td>\n")
    htmlfile.write("                    <td>Red</td>\n")
    htmlfile.write("                    <td>Green</td>\n")
    htmlfile.write("                    <td>Blue</td>\n")
    htmlfile.write("                    <td>Grey</td>\n")
    htmlfile.write("                </tr>\n")
    htmlfile.write("                <tr>\n")
    htmlfile.write("                    <td>" + str(Lips_IR_SNR) + "</td>\n")
    htmlfile.write("                    <td>" + str(Lips_Red_SNR) + "</td>\n")
    htmlfile.write("                    <td>" + str(Lips_Green_SNR) + "</td>\n")
    htmlfile.write("                    <td>" + str(Lips_Blue_SNR) + "</td>\n")
    htmlfile.write("                    <td>" + str(Lips_Grey_SNR) + "</td>\n")
    htmlfile.write(" <td>" + str(Forehead_IR_SNR) + "</td>\n")
    htmlfile.write("         <td>" + str(Forehead_Red_SNR) + "</td>\n")
    htmlfile.write("                      <td>" + str(Forehead_Green_SNR) + "</td>\n")
    htmlfile.write("                   <td>" + str(Forehead_Blue_SNR) + "</td>\n")
    htmlfile.write("                     <td>" + str(Forehead_Grey_SNR) + "</td>\n")
    htmlfile.write(" <td>" + str(RighCheek_IR_SNR) + "</td>\n")
    htmlfile.write("         <td>" + str(RighCheek_Red_SNR) + "</td>\n")
    htmlfile.write("                      <td>" + str(RighCheek_Green_SNR)+ "</td>\n")
    htmlfile.write("                   <td>" + str(RighCheek_Blue_SNR) + "</td>\n")
    htmlfile.write("                     <td>" + str(RighCheek_Grey_SNR) + "</td>\n")
    htmlfile.write(" <td>" + str(LeftCheek_IR_SNR) + "</td>\n")
    htmlfile.write("         <td>" + str(LeftCheek_Red_SNR) + "</td>\n")
    htmlfile.write("                      <td>" + str(LeftCheek_Green_SNR) + "</td>\n")
    htmlfile.write("                   <td>" + str(LeftCheek_Blue_SNR) + "</td>\n")
    htmlfile.write("                     <td>" + str(LeftCheek_Grey_SNR) + "</td>\n")
    htmlfile.write("                       </tr>\n")
    htmlfile.write("                </table>\n")

def html(htmlfilepath,participantnumber,hearratestatus,regions):
    #image_path= r'E:\StudyData\Result\PIS-2212\lips\ICA\3-AfterFFT.png'
    #Begin end
    htmlfile = open(htmlfilepath+ r"FullReport.html", "w+")
    htmlfile.write("<!DOCTYPE html>\n")
    htmlfile.write("<html>\n")
    htmlfile.write("<body>\n")
    htmlfile.write("<h1> Results for " + participantnumber + "</h1>\n\n")

    GenerateCSS(htmlfile)

    for position in hearratestatus:
        htmlfile.write("<h2> For " + position + "</h2>\n")
        GenerateTable(htmlfile, position)

        for region in regions:
            ImagePaths = GetImagePaths(region, htmlfilepath)
            for image_path in ImagePaths:
                image_desc = str(image_path).replace(htmlfilepath,"")
                htmlfile.write("<h3> Region/algorithm/ImageType is : " + image_desc + "</h3>\n")
                htmlfile.write('<img width="50%" src = "' + image_path + '" alt ="cfg">\n')


    htmlfile.write("</body>\n")
    #Document end
    htmlfile.write("</html>\n")
    htmlfile.close()

###region wise
def htmlFullReportwithWindow(htmlfilepath,participantnumber,hearratestatus,TotalWindows):
    #image_path= r'E:\StudyData\Result\PIS-2212\lips\ICA\3-AfterFFT.png'
    #Begin end
    htmlfile = open(htmlfilepath+ r"FullReport_Windows_Region.html", "w+")
    htmlfile.write("<!DOCTYPE html>\n")
    htmlfile.write("<html>\n")
    htmlfile.write("<body>\n")
    htmlfile.write("<h1> Results for " + participantnumber + "</h1>\n\n")

    GenerateCSS(htmlfile)

    for position in hearratestatus:
        htmlfile.write("<h2> For " + position + " </h2>\n")
        #GenerateTable(htmlfile, position)

        htmlfile.write(" <div id='banner'>\n")

        #for region in regions:
        for x in range(1, TotalWindows):
            region="lips"
            ImagePaths = GetImagePaths(region, htmlfilepath + "Result\\" + participantnumber + "\\",  "Before-" + region + "-" + str(x),"None", "M1",position)
            for image_path in ImagePaths:
                image_desc = str(image_path).replace(htmlfilepath,"")
                # htmlfile.write("<h3> Region/None/M1/ImageType is : " + image_desc + "</h3>\n")
                # htmlfile.write('<img width="50%" src = "' + image_path + '" alt ="cfg">\n')

                htmlfile.write("        <div class='images'>\n")
                htmlfile.write("            <img src ='" + image_path + "'>\n")
                htmlfile.write("        </div>\n")
                htmlfile.write("\n")

        htmlfile.write("   </div>\n")



    htmlfile.write("</body>\n")
    #Document end
    htmlfile.write("</html>\n")
    htmlfile.close()

roiregions = ["lips", "forehead", "leftcheek", "rightcheek"]
###Window wise
def htmlFullReportwithWindow(htmlfilepath,participantnumber,position,windowNumber,algo, method,lipsResult,foreheadResult,rightcheekResult,leftcheekResult,GR, bestbpm,bestbpm2):
    #image_path= r'E:\StudyData\Result\PIS-2212\lips\ICA\3-AfterFFT.png'
    #Begin end
    htmlfilepathSave = htmlfilepath + "HTMLresults_Windows\\" + algo + "\\" + method+ "\\"
    if not os.path.exists(htmlfilepathSave):
        os.makedirs(htmlfilepathSave)

    htmlfile = open(htmlfilepathSave+ r"FullReport_Windows_" + str(windowNumber) + ".html", "w+")
    htmlfile.write("<!DOCTYPE html>\n")
    htmlfile.write("<html>\n")
    htmlfile.write("<body>\n")
    htmlfile.write("<h1> Results for " + participantnumber + "</h1>\n\n")

    GenerateCSS(htmlfile)

    htmlfile.write("<h2> For " + position + " </h2>\n")
    #GenerateTable(htmlfile, position)

    htmlfile.write(" <div id='banner'>\n")

    for region in roiregions:
        image_path = GetImagePath_Single(region, htmlfilepath ,  "Before-" + region + "-" + str(windowNumber),algo, method,position)
        #image_desc = str(image_path).replace(htmlfilepath,"")
        #htmlfile.write("<h3> "+ region + "/"+ algo+ "/"+ method + "/ImageType is : " + image_desc + "</h3>\n")
        #htmlfile.write('<img width="50%" src = "' + image_path + '" alt ="cfg">\n')

        htmlfile.write("        <div class='images'>\n")
        htmlfile.write("            <img width='100%' src ='" + image_path + ".png" + "'>\n")
        htmlfile.write("        </div>\n")

    htmlfile.write("   </div>\n")

    htmlfile.write("<h3> After Algorithm </h3>\n")

    htmlfile.write(" <div id='banner'>\n")
    for region in roiregions:
        image_path = GetImagePath_Single(region, htmlfilepath ,  "After-" + region + "-" + str(windowNumber),algo, method,position)

        htmlfile.write("        <div class='images'>\n")
        htmlfile.write("            <img width='100%' src ='" + image_path + ".png" + "'>\n")
        htmlfile.write("        </div>\n")

    htmlfile.write("   </div>\n")

    # htmlfile.write("<h3> After FFT </h3>\n")
    #
    # htmlfile.write(" <div id='banner'>\n")
    # for region in roiregions:
    #     image_path = GetImagePath_Single(region, htmlfilepath ,
    #                                      "FFT-" + region + "-" + str(windowNumber), algo, method, position)
    #
    #     htmlfile.write("        <div class='images'>\n")
    #     htmlfile.write("            <img width='100%' src ='" + image_path + ".png" + "'>\n")
    #     htmlfile.write("        </div>\n")
    #
    # htmlfile.write("   </div>\n")

    htmlfile.write("<h3> After fft and Filtering fft result </h3>\n")

    htmlfile.write(" <div id='banner'>\n")
    for region in roiregions:
        image_path = GetImagePath_Single(region, htmlfilepath ,
                                         "FFt-" + region + "-" + str(windowNumber), algo, method, position)

        htmlfile.write("        <div class='images'>\n")
        htmlfile.write("            <img width='100%' src ='" + image_path + ".png" + "'>\n")
        htmlfile.write("        </div>\n")

    htmlfile.write("   </div>\n")


    htmlfile.write("<h3> BPMs and SNRs Lips</h3>\n")
    htmlfile.write("<p> All region results fo bpms and SNR and choosign best one </p>\n")
    GenerateWindowResult(htmlfile,
                         lipsResult.IrBpm,lipsResult.RedBpm, lipsResult.GreenBpm,lipsResult.BlueBpm,lipsResult.GreyBpm,
                         foreheadResult.IrBpm,foreheadResult.RedBpm, foreheadResult.GreenBpm,foreheadResult.BlueBpm,foreheadResult.GreyBpm,
                         rightcheekResult.IrBpm,rightcheekResult.RedBpm, rightcheekResult.GreenBpm,rightcheekResult.BlueBpm,rightcheekResult.GreyBpm,
                         leftcheekResult.IrBpm,leftcheekResult.RedBpm, leftcheekResult.GreenBpm,leftcheekResult.BlueBpm,leftcheekResult.GreyBpm,
                         lipsResult.IrSnr, lipsResult.RedSnr, lipsResult.GreenSnr, lipsResult.BlueSnr,lipsResult.GreySnr,
                         foreheadResult.IrSnr, foreheadResult.RedSnr, foreheadResult.GreenSnr, foreheadResult.BlueSnr,foreheadResult.GreySnr,
                         rightcheekResult.IrSnr, rightcheekResult.RedSnr, rightcheekResult.GreenSnr,rightcheekResult.BlueSnr, rightcheekResult.GreySnr,
                         leftcheekResult.IrSnr, leftcheekResult.RedSnr, leftcheekResult.GreenSnr,leftcheekResult.BlueSnr, leftcheekResult.GreySnr)

    difference = round(float(GR) - float(bestbpm))
    htmlfile.write("<p> BEST BPM: " + str(bestbpm) + ", Ground Truth: " + str(GR) + " and Differnce is " + str(
        difference) + "  </p>\n")

    htmlfile.write("<p> All region results for bpms and SNR and choosign best one but for SNR calculation 2 </p>\n")
    GenerateWindowResult(htmlfile,
                         lipsResult.IrBpm, lipsResult.RedBpm, lipsResult.GreenBpm, lipsResult.BlueBpm,
                         lipsResult.GreyBpm,
                         foreheadResult.IrBpm, foreheadResult.RedBpm, foreheadResult.GreenBpm, foreheadResult.BlueBpm,
                         foreheadResult.GreyBpm,
                         rightcheekResult.IrBpm, rightcheekResult.RedBpm, rightcheekResult.GreenBpm,
                         rightcheekResult.BlueBpm, rightcheekResult.GreyBpm,
                         leftcheekResult.IrBpm, leftcheekResult.RedBpm, leftcheekResult.GreenBpm,
                         leftcheekResult.BlueBpm, leftcheekResult.GreyBpm,
                         lipsResult.IrSnr2, lipsResult.RedSnr2, lipsResult.GreenSnr2, lipsResult.BlueSnr2,
                         lipsResult.GreySnr2,
                         foreheadResult.IrSnr2, foreheadResult.RedSnr2, foreheadResult.GreenSnr2, foreheadResult.BlueSnr2,
                         foreheadResult.GreySnr2,
                         rightcheekResult.IrSnr2, rightcheekResult.RedSnr2, rightcheekResult.GreenSnr2,
                         rightcheekResult.BlueSnr2, rightcheekResult.GreySnr2,
                         leftcheekResult.IrSnr2, leftcheekResult.RedSnr2, leftcheekResult.GreenSnr2,
                         leftcheekResult.BlueSnr2, leftcheekResult.GreySnr2)

    difference = round(float(GR) - float(bestbpm2))
    htmlfile.write("<p> BEST BPM: " + str(bestbpm2) + ", Ground Truth: " + str(GR) + " and Differnce is " + str(
        difference) + "  </p>\n")
    #Document end

    htmlfile.write("</body>\n")
    htmlfile.write("</html>\n")
    htmlfile.close()


def GenerateWindowSummaryTableHeader(htmlfile):
    htmlfile.write("<table >\n")
    htmlfile.write("                <tr>\n")
    htmlfile.write("                    <td>Window_Count</td>\n")
    htmlfile.write("                    <td  style='background:#5F90AC;color:white;'><strong>Ground Truth</strong></td>\n")

    #for all methods
    for item in fftmethods:
        htmlfile.write("                    <td>HR_"+ item + "</td>\n")
        htmlfile.write("                    <td  style='background:#FFA48E;'>Diff</td>\n")

    htmlfile.write("                </tr>\n")

def GenerateWindowSummaryTableBody(htmlfile,windowcount, GeneratedmethodValues,groundtruth):
    htmlfile.write("                <tr>\n")
    htmlfile.write("                    <td>" + str(windowcount) + "</td>\n")
    htmlfile.write("                    <td  style='background:#5F90AC;color:white;'><strong>"+ str(groundtruth) + "</strong></td>\n")
    ##methods values
    #for index, row in GeneratedmethodValues.iterrows():
        #print(row['c1'], row['c2'])
    for value in GeneratedmethodValues:
        hr = value[0]
        difference = value[1]
        htmlfile.write("                    <td >" + str(hr) + "</td>\n")
        htmlfile.write("                    <td  style='background:#FFA48E;' >" + str(difference) + "</td>\n")
    htmlfile.write("                </tr>\n")


def GetGeneratedData(PiNo,position,htmlfilepath,algotype,fftype):
    filepath = htmlfilepath + algotype + '\\' #+ fftype + '\\'
    HrFile =  filepath + "HRdata-" + fftype + ".txt"
    #SPOFile =  filepath + "SPOdata-" + fftype + ".txt""

    #read data from files
    HrFiledata = open(HrFile, "r")
    #SPOFiledata = open(SPOFile, "r")

    HrDATA =HrFiledata.read().replace("\t","").split("\n") #remove \t
    #SpoGr =SPOFiledata.read().split("\n")

    return HrDATA #, SpoGr

def WindowWise_participantsSummary(htmlfilepath,ParticipantNumber, GeneratedmethodValues,Algorithm,position):
    htmlfile = open(htmlfilepath + r"Window_" + Algorithm + "_ParicipantSummaryReport.html", "w+")
    htmlfile.write("<!DOCTYPE html>\n")
    htmlfile.write("<html>\n")
    GenerateCSS(htmlfile)
    htmlfile.write("<body>\n")
    htmlfile.write("<h1> Summary for "+ ParticipantNumber + " against all Techniques</h1>\n\n")


    htmlfile.write("<h2> Result for "+ Algorithm + " with position : "+ position+"</h2>\n")

    GenerateWindowSummaryTableHeader(htmlfile)
    #
    # dataResultsheetW1= []
    # dataResultsheetW2= []

    # for ffttype in fftmethods:
    #     HrDATA =GetGeneratedData(ParticipantNumber,position,htmlfilepath,Algorithm,ffttype)
    #
    #     for item in HrDATA:
    #         allvalues = item.split(",")
    #         GenerateWindowSummaryTableBodyandFooter(htmlfile,allvalues[0], dfMethodsValuesAll)


    HrDATA1 =GetGeneratedData(ParticipantNumber,position,htmlfilepath,Algorithm,"M1")
    HrDATA2 =GetGeneratedData(ParticipantNumber,position,htmlfilepath,Algorithm,"M2")
    HrDATA3 =GetGeneratedData(ParticipantNumber,position,htmlfilepath,Algorithm,"M3")
    HrDATA4 =GetGeneratedData(ParticipantNumber,position,htmlfilepath,Algorithm,"M4")
    # HrDATA5 =GetGeneratedData(ParticipantNumber,position,htmlfilepath,Algorithm,"M5")
    # HrDATA6 =GetGeneratedData(ParticipantNumber,position,htmlfilepath,Algorithm,"M6")

    N = len(HrDATA1) ## all should have same lenght

    MethodsData = []

    for i in range(0,N): #windows count
        currentwindow = i
        allvalues =HrDATA1[i].split(",") #WindowCount , GroundTruth, BestBpm, Difference
        groundTruth = allvalues[1] # GroundTruth same over all methods for a specific window

        MethodsData.append([allvalues[2] , allvalues[3]])

        allvalues =HrDATA2[i].split(",")
        MethodsData.append([allvalues[2] , allvalues[3]])

        allvalues =HrDATA3[i].split(",")
        MethodsData.append([allvalues[2] , allvalues[3]])

        allvalues =HrDATA4[i].split(",")
        MethodsData.append([allvalues[2] , allvalues[3]])
        #
        # allvalues =HrDATA5[i].split(",")
        # MethodsData.append([allvalues[2] , allvalues[3]])
        #
        # allvalues =HrDATA6[i].split(",")
        # MethodsData.append([allvalues[2] , allvalues[3]])

        GenerateWindowSummaryTableBody(htmlfile,currentwindow,MethodsData,groundTruth)
        MethodsData=[]

    htmlfile.write("            </table>\n")

    #define breaks in document
    definebreaks(htmlfile)
    definebreaks(htmlfile)

    htmlfile.write("</body>\n")
    # Document end
    htmlfile.write("</html>\n")
    htmlfile.close()
    print("Generated!")

def GenerateTable(htmlfile, type):
    htmlfile.write('<table >')
    htmlfile.write('                <tr>')
    htmlfile.write('                    <td style="width:10%" rowspan="2">PI No</td>')
    htmlfile.write('                    <td style="width:10%"  colspan="2">Frame Rate</td>')
    htmlfile.write('                    <td style="width:30%"  colspan="6">' + type + ' Heart Rate</td>')
    htmlfile.write('                    <td style="width:30%"  colspan="6">' + type + ' Blood Oxygen</td>')
    htmlfile.write('                    <td style="width:20%"  rowspan="2">Other Details</td>')
    htmlfile.write('                </tr>')
    htmlfile.write('                <tr>')
    htmlfile.write('                    <td>Color</td>')
    htmlfile.write('                    <td>Ir</td>')
    htmlfile.write('                    <td>None</td>')
    htmlfile.write('                    <td>FastICA</td>')
    htmlfile.write('                    <td>PCA</td>')
    htmlfile.write('                    <td>Jade</td>')
    htmlfile.write('                    <td>Other</td>')
    htmlfile.write('                    <td>GroundTruth</td>')
    htmlfile.write('                    <td>None</td>')
    htmlfile.write('                    <td>FastICA</td>')
    htmlfile.write('                    <td>PCA</td>')
    htmlfile.write('                    <td>Jade</td>')
    htmlfile.write('                    <td>Other</td>')
    htmlfile.write('                    <td>GroundTruth</td>')
    htmlfile.write('                </tr>')
    htmlfile.write('                <tr>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                </tr>')
    htmlfile.write('                <tr>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                    <td>&nbsp;</td>')
    htmlfile.write('                </tr>')
    htmlfile.write('            </table>')

