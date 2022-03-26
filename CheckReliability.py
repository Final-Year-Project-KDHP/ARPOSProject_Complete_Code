import numpy as np

class CheckReliability:

    # Constructor
    def __init__(self):
        self.previousAcceptedHeartRates = [60.0, 60.0]
        self.previousAcceptedOxygenSaturation = [ 98.0, 98.0 ]

    #Heart rate
    heartRateValueValue = 0.0
    heartRateValueError = 0.0
    SignalToNoiseAcceptanceThreshold = 5.0#3.0
    HeartRateSnrAcceptanceThreshold = SignalToNoiseAcceptanceThreshold
    numberOfAnalysisFailuresSinceCorrectHeartRate = 0.0
    previousComputedHeartRate= 0.0 # previous computed values for this face
    HeartRateDeviationAcceptanceFactor = 0.18
    NumberOfAnalysisFailuresBeforeTrackingFailureInHeartRate = 20
    HeartRateErrorScalingFactor = 0.2
    previousAcceptedHeartRates = []
    freqencySamplingError = 0.0

    #Blood oxygen
    OxygenSaturationErrorAcceptanceThreshold = 5.0
    previousComputedOxygenSaturation = 0.0
    previousAcceptedOxygenSaturation = []
    numberOfAnalysisFailuresSinceCorrectOxygenSaturation = 0
    OxygenSaturationAcceptanceFactor = 0.14
    NumberOfAnalysisFailuresBeforeTrackingFailureInOxygenSaturation=20
    OxygenSaturationErrorScalingFactor = 0.2

    #region Heart Reliability
    def AddPreviousHeartRate(self,heartRate):
        self.previousAcceptedHeartRates.append(heartRate)
        if (len(self.previousAcceptedHeartRates) > 2):
            self.previousAcceptedHeartRates.pop(0) # Remove at zero index

    def IsHeartRateReliable(self):
        return self.numberOfAnalysisFailuresSinceCorrectHeartRate < self.NumberOfAnalysisFailuresBeforeTrackingFailureInHeartRate

    '''-----------------------------------------
    Accept or reject the best heart rate reading:
    accept or reject region based on signal to noise or deviation from last computed value
    --------------------------------------------'''
    def AcceptorRejectHR(self,bestHeartRateSnr, bestBpm, freqencySamplingError):

        condition1 = (bestHeartRateSnr > self.HeartRateSnrAcceptanceThreshold)
        if(self.previousComputedHeartRate == 0.0):
            condition2 = False
        else:
            condition2 = (np.abs(bestBpm - self.previousComputedHeartRate) > (self.HeartRateDeviationAcceptanceFactor * self.previousComputedHeartRate))

        if (not condition1):
            self.numberOfAnalysisFailuresSinceCorrectHeartRate = self.numberOfAnalysisFailuresSinceCorrectHeartRate + 1

            # compute error
            if not (self.IsHeartRateReliable()):
                # the number of analysis failures saturates at NumberOfAnalysisFailuresBeforeTrackingFailure
                # this stop the error value from making the time series chart unreadable.
                self.numberOfAnalysisFailuresSinceCorrectHeartRate = self.NumberOfAnalysisFailuresBeforeTrackingFailureInHeartRate

            freq_error_fudge_factor = self.numberOfAnalysisFailuresSinceCorrectHeartRate * self.HeartRateErrorScalingFactor
            subOld = (np.abs(self.previousAcceptedHeartRates[0] - self.previousAcceptedHeartRates[1]))
            mulFreq =freq_error_fudge_factor * subOld

            bad_bpm_freqerror = mulFreq + freqencySamplingError

            if (self.previousComputedHeartRate > 0):
                heartRateValue = self.previousComputedHeartRate
            else:
                if (self.previousAcceptedHeartRates[0] > self.previousAcceptedHeartRates[1]):
                    heartRateValue = self.previousAcceptedHeartRates[0]
                else:
                    heartRateValue = self.previousAcceptedHeartRates[1]

            heartRateError = bad_bpm_freqerror

            bestDifference = (bestBpm - self.previousComputedHeartRate)
            factorpercentage = (self.HeartRateDeviationAcceptanceFactor * self.previousComputedHeartRate)

            # if (self.previousComputedHeartRate > 0):
            #     if (bestDifference > 40):
            #         self.previousComputedHeartRate = heartRateValue
            #     else:
            #         self.previousComputedHeartRate = bestBpm
            #         heartRateValue = bestBpm
            # else:
            #     self.previousComputedHeartRate = bestBpm
            #     heartRateValue = bestBpm
            # if(heartRateError>=30):
            #     self.previousComputedHeartRate = heartRateValue
            # else:
            # self.previousComputedHeartRate = bestBpm

        else:
            if (condition2):
                self.numberOfAnalysisFailuresSinceCorrectHeartRate = self.numberOfAnalysisFailuresSinceCorrectHeartRate + 1

                # compute error
                if not (self.IsHeartRateReliable()):
                    # the number of analysis failures saturates at NumberOfAnalysisFailuresBeforeTrackingFailure
                    # this stop the error value from making the time series chart unreadable.
                    self.numberOfAnalysisFailuresSinceCorrectHeartRate = self.NumberOfAnalysisFailuresBeforeTrackingFailureInHeartRate

                freq_error_fudge_factor = self.numberOfAnalysisFailuresSinceCorrectHeartRate * self.HeartRateErrorScalingFactor
                subOld = (np.abs(self.previousAcceptedHeartRates[0] - self.previousAcceptedHeartRates[1]))
                mulFreq = freq_error_fudge_factor * subOld

                bad_bpm_freqerror = mulFreq + freqencySamplingError

                if(self.previousComputedHeartRate >0):
                    heartRateValue = self.previousComputedHeartRate
                else:
                    if(self.previousAcceptedHeartRates[0] > self.previousAcceptedHeartRates[1]):
                        heartRateValue = self.previousAcceptedHeartRates[0]
                    else:
                        heartRateValue = self.previousAcceptedHeartRates[1]

                heartRateError = bad_bpm_freqerror

                bestDifference = (bestBpm - self.previousComputedHeartRate)
                factorpercentage = (self.HeartRateDeviationAcceptanceFactor * self.previousComputedHeartRate)
            else:
                # accept the heart rate
                self.numberOfAnalysisFailuresSinceCorrectHeartRate = 0
                self.AddPreviousHeartRate(bestBpm)

                heartRateValue = bestBpm#self.previousAcceptedHeartRates[0]
                heartRateError = self.freqencySamplingError
                self.previousComputedHeartRate = bestBpm

        return heartRateValue, heartRateError
    #endregion

    #region blood oxygen Reliability
    '''-----------------------------------------
    Accept or reject the best oxygen saturation reading:
    accept or reject region based on signal to noise or deviation from last computed value
    --------------------------------------------'''

    def IsOxygenSaturationReliable(self):
        return self.numberOfAnalysisFailuresSinceCorrectOxygenSaturation < self.NumberOfAnalysisFailuresBeforeTrackingFailureInOxygenSaturation

    def AddPreviousOxygenSaturation(self,bloodOxygen):
        self.previousAcceptedOxygenSaturation.append(bloodOxygen)
        if (len(self.previousAcceptedOxygenSaturation) > 2):
            self.previousAcceptedOxygenSaturation.pop(0)

    def AcceptorRejectSPO(self,oxygenSaturationValueError,oxygenSaturationValue):
        condition1=(oxygenSaturationValueError > self.OxygenSaturationErrorAcceptanceThreshold)

        if (self.previousComputedOxygenSaturation == 0.0):
            condition2 = False
        else:
            condition2 =(np.abs(oxygenSaturationValue - self.previousComputedOxygenSaturation) > self.OxygenSaturationAcceptanceFactor * self.previousComputedOxygenSaturation)

        # accept or reject region based on signal to noise or deviation from last computed value
        if ( condition1 or condition2):
            # the value has been rejected
            # set the oxygen saturation value the last
            oxygenSaturationValue = self.previousAcceptedOxygenSaturation[1]
            self.numberOfAnalysisFailuresSinceCorrectOxygenSaturation= self.numberOfAnalysisFailuresSinceCorrectOxygenSaturation+1

            # compute error
            if not (self.IsOxygenSaturationReliable()):
                # the number of analysis failures saturates at NumberOfAnalysisFailuresBeforeTrackingFailure
                # this stop the error value from making the time series chart unreadable.
                self.numberOfAnalysisFailuresSinceCorrectOxygenSaturation = self.NumberOfAnalysisFailuresBeforeTrackingFailureInOxygenSaturation

            freq_error_fudge_factor = self.numberOfAnalysisFailuresSinceCorrectOxygenSaturation * self.OxygenSaturationErrorScalingFactor
            oxygenSaturationValueError = freq_error_fudge_factor * (np.abs(
            self.previousAcceptedOxygenSaturation[0] - self.previousAcceptedOxygenSaturation[1]) + oxygenSaturationValueError)


            self.previousComputedOxygenSaturation = oxygenSaturationValue



        else:
            # accept the bloodoxygen
            self.numberOfAnalysisFailuresSinceCorrectOxygenSaturation = 0
            self.AddPreviousOxygenSaturation(oxygenSaturationValue)
            oxygenSaturationValue = oxygenSaturationValue#self.previousAcceptedOxygenSaturation[0]
            self.previousComputedOxygenSaturation = oxygenSaturationValue

        return oxygenSaturationValue, oxygenSaturationValueError

    #endregion
