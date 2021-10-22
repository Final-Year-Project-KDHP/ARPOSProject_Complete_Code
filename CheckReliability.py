import numpy as np

class CheckReliability:

    #Heart rate
    heartRateValueValue = 0.0
    heartRateValueError = 0.0
    SignalToNoiseAcceptanceThreshold = 2.5
    HeartRateSnrAcceptanceThreshold = SignalToNoiseAcceptanceThreshold
    numberOfAnalysisFailuresSinceCorrectHeartRate = 0.0
    previousComputedHeartRate= 0.0 # previous computed values for this face
    HeartRateDeviationAcceptanceFactor = 0.2
    NumberOfAnalysisFailuresBeforeTrackingFailureInHeartRate = 20
    HeartRateErrorScalingFactor = 0.2
    previousAcceptedHeartRates = [60.0, 60.0]
    freqencySamplingError = 0.0

    #Blood oxygen
    OxygenSaturationErrorAcceptanceThreshold = 5.0
    previousComputedOxygenSaturation = 0.0
    previousAcceptedOxygenSaturation = [ 98.0, 98.0 ]
    numberOfAnalysisFailuresSinceCorrectOxygenSaturation = 0
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

        condition1 = (bestHeartRateSnr < self.HeartRateSnrAcceptanceThreshold)
        condition2 = (np.abs(bestBpm - self.previousComputedHeartRate) > (self.HeartRateDeviationAcceptanceFactor * self.previousComputedHeartRate))

        if (condition1 or condition2):
            self.numberOfAnalysisFailuresSinceCorrectHeartRate = self.numberOfAnalysisFailuresSinceCorrectHeartRate + 1

            # compute error
            if not (self.IsHeartRateReliable()):
                # the number of analysis failures saturates at NumberOfAnalysisFailuresBeforeTrackingFailure
                # this stop the error value from making the time series chart unreadable.
                self.numberOfAnalysisFailuresSinceCorrectHeartRate = self.NumberOfAnalysisFailuresBeforeTrackingFailureInHeartRate

            freq_error_fudge_factor = self.numberOfAnalysisFailuresSinceCorrectHeartRate * self.HeartRateErrorScalingFactor
            bad_bpm_freqerror = freq_error_fudge_factor * (np.abs(self.previousAcceptedHeartRates[0] - self.previousAcceptedHeartRates[1]) + freqencySamplingError)

            heartRateValue = self.previousAcceptedHeartRates[0]
            heartRateError = bad_bpm_freqerror

            bestDifference = (bestBpm - self.previousComputedHeartRate)
            factorpercentage = (self.HeartRateDeviationAcceptanceFactor * self.previousComputedHeartRate)

            if (self.previousComputedHeartRate > 0):
                if (bestDifference > 50):
                    self.previousComputedHeartRate = heartRateValue
                else:
                    self.previousComputedHeartRate = bestBpm
                    heartRateValue = bestBpm
            else:
                self.previousComputedHeartRate = bestBpm
                heartRateValue = bestBpm

            # self.previousComputedHeartRate = bestBpm

        else:
            # accept the heart rate
            self.numberOfAnalysisFailuresSinceCorrectHeartRate = 0
            self.AddPreviousHeartRate(bestBpm)

            heartRateValue = self.previousAcceptedHeartRates[0]
            heartRateError = self.freqencySamplingError

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
        condition2 =(np.abs(oxygenSaturationValue - self.previousComputedOxygenSaturation) > self.OxygenSaturationErrorAcceptanceThreshold * self.previousComputedOxygenSaturation)
        # accept or reject region based on signal to noise or deviation from last computed value
        if ( condition1 or condition2):  # TODO: IS THIS CORRECT!
            # the value has been rejected
            # set the oxygen saturation value the last
            oxygenSaturationValue = self.previousAcceptedOxygenSaturation[0]
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

            oxygenSaturationValue = self.previousAcceptedOxygenSaturation[0]

        return oxygenSaturationValue, oxygenSaturationValueError

    #endregion
