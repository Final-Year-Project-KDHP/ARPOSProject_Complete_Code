import os
import sys
import numpy as np

import CommonMethods
from CheckReliability import CheckReliability
from Configurations import Configurations
from ProcessData import ProcessFaceData

# Global Objects
objConfig = Configurations()

def Process_Participants_Data_Windows(ROIStore, savepath, DiskPath, ParticipantNumber, position, algotype, fftype, HrGr,
                                      filtertype, resulttype, preprocess, isSmoothen,EstimatedFPS):
    WindowRegionList = {}

    previousComputedHeartRate = 0.0
    numberOfAnalysisFailuresSinceCorrectHeartRate = 0.0

    # Create global data object and use dictionary (ROI Store) to uniquely store a regions data
    lipsData = ROIStore.get(objConfig.roiregions[0]) #Lips
    # lipsData = v.getAllData()
    # timecolorLips = v.timecolor
    # timecolorcountLips = v.timecolorcount
    # timeirLips = v.timeir
    # timeircountLips = v.timeircount
    foreheadData = ROIStore.get(objConfig.roiregions[1]) #forehead
    leftcheekData = ROIStore.get(objConfig.roiregions[2]) #leftcheek
    rightcheekData = ROIStore.get(objConfig.roiregions[3]) #rightcheek

    # dfMethodsValues = pd.DataFrame(columns=['Window_Id', 'GroundTruth', 'HrValue', 'Difference'])
    savepath = savepath + algotype + '/'  # + fftype + '/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    ListHrdata = []
    ListSPOdata = []

    ##Windows for lips
    LengthofAllFrames = len(lipsData)  # all have same lenghts
    TimeinSeconds = LengthofAllFrames / EstimatedFPS
    step = 30  # slide window for 1 second or 30 frames
    timeinSeconds = 5
    WindowSlider = step * timeinSeconds  # step * 5 second,  window can hold  150 frames or 5 second data

    # TotalWindows in this sample
    TotalWindows = (TimeinSeconds - 5) + 1  # 5 second window gorup
    HrAvgList = CommonMethods.splitGroundTruth(HrGr, TotalWindows)

    bestHeartRateSnr = 0.0
    bestBpm = 0.0
    freqencySamplingError = 0.0

    smallestOxygenError = sys.float_info.max
    ignoregray = False
    IsDetrend = False
    GenerateGraphs = True
    WindowCount=0
    objProcessData = ProcessFaceData()

    foreheadResult = objProcessData.Process_EntireSignalData(foreheadData, foreheadData.distanceM, 30,
                                                      algotype, fftype, 'forehead',
                                                      savepath + "forehead/" + algotype + "/" + fftype + "/",
                                                      foreheadData.timecolorForehead,
                                                      foreheadData.timecolorcountForehead,
                                                      foreheadData.timeirForehead,
                                                      foreheadData.timeircountForehead, WindowCount,
                                                      filtertype, resulttype,
                                                      ignoregray, isSmoothen, IsDetrend, preprocess, savepath,
                                                      GenerateGraphs, timeinSeconds)

    lipsResult = objProcessData.Process_EntireSignalData(lipsData, lipsData.distanceM, 30,
                                                  algotype, fftype, 'lips',
                                                  savepath + "lips/" + algotype + "/" + fftype + "/",
                                                  lipsData.timecolorLips,
                                                  lipsData.timecolorcountLips, lipsData.timeirLips,
                                                  lipsData.timeircountLips, WindowCount, filtertype,
                                                  resulttype,
                                                  ignoregray, isSmoothen, IsDetrend, preprocess, savepath,
                                                  GenerateGraphs, timeinSeconds)

    leftcheekResult = objProcessData.Process_EntireSignalData(leftcheekData, leftcheekData.distanceM, 30,
                                                        algotype, fftype, 'rightcheek',
                                                        savepath + "rightcheek/" + algotype + "/" + fftype + "/",
                                                        leftcheekData.timecolorRcheek,
                                                        leftcheekData.timecolorcountRcheek,
                                                        leftcheekData.timeirRcheek,
                                                        leftcheekData.timeircountRcheek, WindowCount,
                                                        filtertype, resulttype,
                                                        ignoregray, isSmoothen, IsDetrend, preprocess, savepath,
                                                        GenerateGraphs, timeinSeconds)

    rightcheekResult= objProcessData.Process_EntireSignalData(rightcheekData, rightcheekData.distanceM, 30,
                                                       algotype, fftype, 'leftcheek',
                                                       savepath + "leftcheek/" + algotype + "/" + fftype + "/",
                                                       rightcheekData.timecolorLcheek,
                                                       rightcheekData.timecolorcountLcheek,
                                                       rightcheekData.timeirLcheek,
                                                       rightcheekData.timeircountLcheek, WindowCount,
                                                       filtertype, resulttype,
                                                       ignoregray, isSmoothen, IsDetrend, preprocess, savepath,
                                                       GenerateGraphs, timeinSeconds)

    WindowRegionList['lips'] = lipsResult
    WindowRegionList['forehead'] = foreheadResult
    WindowRegionList['rightcheek'] = rightcheekResult
    WindowRegionList['leftcheek'] = leftcheekResult

    for k, v in WindowRegionList.items():
        if (v.IrSnr > bestHeartRateSnr):
            bestHeartRateSnr = v.IrSnr
            bestBpm = v.IrBpm
            channeltype = 'Ir'
            regiontype = k
            freqencySamplingError = v.IrFreqencySamplingError

        if (v.GreySnr > bestHeartRateSnr):
            bestHeartRateSnr = v.GreySnr
            bestBpm = v.GreyBpm
            channeltype = 'Grey'
            regiontype = k
            freqencySamplingError = v.GreyFreqencySamplingError

        if (v.RedSnr > bestHeartRateSnr):
            bestHeartRateSnr = v.RedSnr
            bestBpm = v.RedBpm
            channeltype = 'Red'
            regiontype = k
            freqencySamplingError = v.RedFreqencySamplingError

        if (v.GreenSnr > bestHeartRateSnr):
            bestHeartRateSnr = v.GreenSnr
            bestBpm = v.GreenBpm
            channeltype = 'Green'
            regiontype = k
            freqencySamplingError = v.GreenFreqencySamplingError

        if (v.BlueSnr > bestHeartRateSnr):
            bestHeartRateSnr = v.BlueSnr
            bestBpm = v.BlueBpm
            channeltype = 'Blue'
            regiontype = k
            freqencySamplingError = v.BlueFreqencySamplingError

        if (v.oxygenSaturationValueError < smallestOxygenError):
            smallestOxygenError = v.oxygenSaturationValueError
            finaloxy = v.oxygenSaturationValueValue

    objReliability = CheckReliability()

    previousHR, numberOfAnalysisFailureshr, HRValue, HRError = objReliability.AddHeartRate(bestHeartRateSnr, bestBpm,
                                                                            previousComputedHeartRate,
                                                                            numberOfAnalysisFailuresSinceCorrectHeartRate,
                                                                            freqencySamplingError)
    previousComputedHeartRate = previousHR
    numberOfAnalysisFailuresSinceCorrectHeartRate = numberOfAnalysisFailureshr
    bestBpm = HRValue

    difference = round(float(HrAvgList[WindowCount]) - float(bestBpm))
    ListHrdata.append(str(WindowCount) + " ,\t" + str(round(HrAvgList[WindowCount])) + " ,\t" + str(
        round(bestBpm)) + " ,\t" + str(difference))



    fHr = open(savepath + "HRdata-" + fftype + "_Fl_" + str(filtertype) + "_Rs_" + str(resulttype) + "_Pr_" + str(
        preprocess) + "_Sm_" + str(isSmoothen) + ".txt", "w+")
    i = 0
    for item in ListHrdata:
        DataRow = item.replace("\t", "").split(" ,")
        windowc = DataRow[0]
        grtruth = DataRow[1]
        bpmval = DataRow[2]

        if (i == len(ListHrdata) - 1):
            fHr.write(str(item))
        else:
            fHr.write(str(item) + '\n')
        i = i + 1

    fHr.close()