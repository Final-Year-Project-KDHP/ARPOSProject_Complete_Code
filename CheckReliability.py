import numpy as np

##############################################
# Accept or reject the best heart rate reading
##############################################
# accept or reject region based on signal to noise or deviation from last computed value
SignalToNoiseAcceptanceThreshold = 5.0
HeartRateSnrAcceptanceThreshold = SignalToNoiseAcceptanceThreshold
# previous computed values for this face
previousComputedHeartRate = 0.0
HeartRateDeviationAcceptanceFactor = 0.3
NumberOfAnalysisFailuresBeforeTrackingFailureInHeartRate = 20
HeartRateErrorScalingFactor = 0.2
previousAcceptedHeartRates = [60.0, 60.0]

class CheckReliability:
    # ##############################################
    # # Accept or reject the best heart rate reading
    # ##############################################
    # # accept or reject region based on signal to noise or deviation from last computed value
    heartRateValueValue = 0.0
    heartRateValueError = 0.0
    SignalToNoiseAcceptanceThreshold = 2.5  # 3.5
    HeartRateSnrAcceptanceThreshold = SignalToNoiseAcceptanceThreshold
    # #previous computed values for this face
    HeartRateDeviationAcceptanceFactor = 0.3
    NumberOfAnalysisFailuresBeforeTrackingFailureInHeartRate = 20
    HeartRateErrorScalingFactor = 0.2
    previousAcceptedHeartRates = [60.0, 60.0]
    freqencySamplingError = 0.0

    def AddPreviousHeartRate(self,heartRate):
        previousAcceptedHeartRates.append(heartRate)
        if (len(previousAcceptedHeartRates) > 2):
            previousAcceptedHeartRates.pop(0)

    def IsHeartRateReliable(self,numberOfAnalysisFailuresSinceCorrectHeartRate):
        return numberOfAnalysisFailuresSinceCorrectHeartRate < NumberOfAnalysisFailuresBeforeTrackingFailureInHeartRate

    def AddHeartRate(self,bestHeartRateSnr, bestBpm, previousComputedHeartRate,
                     numberOfAnalysisFailuresSinceCorrectHeartRate, freqencySamplingError):

        condition1 = (bestHeartRateSnr < HeartRateSnrAcceptanceThreshold)
        condition2 = (np.abs(bestBpm - previousComputedHeartRate) > (
                    HeartRateDeviationAcceptanceFactor * previousComputedHeartRate))

        if (condition1 or condition2):
            numberOfAnalysisFailuresSinceCorrectHeartRate = numberOfAnalysisFailuresSinceCorrectHeartRate + 1

            # compute error
            if not (self.IsHeartRateReliable(numberOfAnalysisFailuresSinceCorrectHeartRate)):
                # the number of analysis failures saturates at NumberOfAnalysisFailuresBeforeTrackingFailure
                # this stop the error value from making the time series chart unreadable.
                numberOfAnalysisFailuresSinceCorrectHeartRate = NumberOfAnalysisFailuresBeforeTrackingFailureInHeartRate

            freq_error_fudge_factor = numberOfAnalysisFailuresSinceCorrectHeartRate * HeartRateErrorScalingFactor
            bad_bpm_freqerror = freq_error_fudge_factor * (
                        np.abs(previousAcceptedHeartRates[0] - previousAcceptedHeartRates[1]) + freqencySamplingError)

            heartRateValueValue = previousAcceptedHeartRates[0]
            heartRateValueError = bad_bpm_freqerror

            bestDifference = (bestBpm - previousComputedHeartRate)
            factorpercentage = (HeartRateDeviationAcceptanceFactor * previousComputedHeartRate)

            if (previousComputedHeartRate > 0):
                if (bestDifference > 50):
                    previousComputedHeartRate = heartRateValueValue
                else:
                    previousComputedHeartRate = bestBpm
            else:
                previousComputedHeartRate = bestBpm
        else:
            # accept the heart rate
            numberOfAnalysisFailuresSinceCorrectHeartRate = 0
            # AddPreviousHeartRate(bestBpm)

            heartRateValueValue = previousAcceptedHeartRates[0]
            heartRateValueError = freqencySamplingError

        return previousComputedHeartRate, numberOfAnalysisFailuresSinceCorrectHeartRate, heartRateValueValue, heartRateValueError

