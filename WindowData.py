import enum
from datetime import datetime

class Window_Data:
  #Current Window Caluculations
  BestBPM =0.0
  BestSnR =0.0
  IrSnr = 0.0
  GreySnr = 0.0
  RedSnr = 0.0
  GreenSnr = 0.0
  BlueSnr = 0.0
  IrBpm = 0.0
  GreyBpm = 0.0
  RedBpm = 0.0
  GreenBpm = 0.0
  BlueBpm = 0.0
  WindowNo = 0
  regiontype = ''
  isSmooth = False

  BestSnR2 =0.0
  IrSnr2 = 0.0
  GreySnr2 = 0.0
  RedSnr2 = 0.0
  GreenSnr2 = 0.0
  BlueSnr2 = 0.0
  BestBPM2 = 0.0
  BestSnR2 = 0.0

  IrFreqencySamplingError = 0.0
  GreyFreqencySamplingError = 0.0
  RedFreqencySamplingError = 0.0
  GreenFreqencySamplingError = 0.0
  BlueFreqencySamplingError = 0.0

  oxygenSaturationValueError = 0.0
  oxygenSaturationValueValue = 0.0
  oxygenSaturationSTD=0.0

  #Storage on everystep details
  SignalWindowOriginal = None
  SignalWindowPreProcessed = None
  SignalWindowAfterAlgorithm = None
  SignalWindowSmoothed = None
  SignalWindowFFT = None
  SignalWindowFiltered = None
  SignalWindowHeartRateCalculation = None
  SignalWindowBestBPM = None
  SignalWindowBestSPO = None

  def getFileNameForWindow(self,windowCount, resultType):
    return 'test_' + str(windowCount) + str(resultType)

  #Log Details
  diffTime = 0.0
  TimeLog = {}
  diffTimeLog = {}
  LogItems = []

  def LogTime(self, LogItem):
    logTime = datetime(datetime.now().year, datetime.now().month, datetime.now().day,
                         datetime.now().time().hour, datetime.now().time().minute,
                         datetime.now().time().second, datetime.now().time().microsecond)
    self.TimeLog[LogItem] = logTime
    return logTime

  def timeDifferences(self):
    startTime = self.TimeLog[LogItems.Start_Total]
    endTime = self.TimeLog[LogItems.End_Total]
    self.diffTime = (endTime - startTime)
    self.diffTimeLog[LogItems.End_Total] = self.diffTime
    # self.diffTime = self.diffTime.total_seconds()

    startTime = self.TimeLog[LogItems.Start_PreProcess]
    endTime = self.TimeLog[LogItems.End_PreProcess]
    difference = endTime - startTime
    self.diffTimeLog[LogItems.End_PreProcess] = difference

    startTime = self.TimeLog[LogItems.Start_Algorithm]
    endTime = self.TimeLog[LogItems.End_Algorithm]
    difference = endTime - startTime
    self.diffTimeLog[LogItems.End_Algorithm] = difference

    startTime = self.TimeLog[LogItems.Start_FFT]
    endTime = self.TimeLog[LogItems.End_FFT]
    difference = endTime - startTime
    self.diffTimeLog[LogItems.End_FFT] = difference

    if(self.isSmooth):
      startTime = self.TimeLog[LogItems.Start_Smooth]
      endTime = self.TimeLog[LogItems.End_Smooth]
      difference = endTime - startTime
      self.diffTimeLog[LogItems.End_Smooth] = difference

    startTime = self.TimeLog[LogItems.Start_Filter]
    endTime = self.TimeLog[LogItems.End_Filter]
    difference = endTime - startTime
    self.diffTimeLog[LogItems.End_Filter] = difference

    startTime = self.TimeLog[LogItems.Start_ComputerHRSNR]
    endTime = self.TimeLog[LogItems.End_ComputerHRSNR]
    difference = endTime - startTime
    self.diffTimeLog[LogItems.End_ComputerHRSNR] = difference

    startTime = self.TimeLog[LogItems.Start_SPO]
    endTime = self.TimeLog[LogItems.End_SPO]
    difference = endTime - startTime
    self.diffTimeLog[LogItems.End_SPO] = difference

class LogItems(enum.Enum):
  Start_Total = 1
  End_Total = 2
  Start_PreProcess =3
  End_PreProcess=4
  Start_Algorithm=5
  End_Algorithm=6
  Start_FFT=7
  End_FFT=8
  Start_Smooth=9
  End_Smooth=10
  Start_Filter=11
  End_Filter=12
  Start_ComputerHRSNR=13
  End_ComputerHRSNR=14
  Start_SPO=15
  End_SPO=16