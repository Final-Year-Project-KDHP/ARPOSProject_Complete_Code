# O(n) solution for finding
# maximum sum of a subarray of size k


def maxSum(arr, k,RoiAllData,fps):
    OriginalData = RoiAllData
    LengthofAllFrames = len(OriginalData)
    TimeinSeconds = LengthofAllFrames / fps
    step = 30  # slide window for 1 second or 30 frames
    WindowSlider = step * 5  # step * 5 second,  window can hold  150 frames or 5 second data

    for j in range(0, int(TimeinSeconds)):
        if len(OriginalData) >= WindowSlider:  # has atleast enoguth data to process
            CroppedImages = OriginalData[:WindowSlider]  # process only window slider content at a time

            bluechannel = []
            #Load color and ir data
            greenchannel = []
            redchannel = []

            for imgCapturedCropped in CroppedImages:
                # Calculate the mean of each channel
                channels = cv2.mean(imgCapturedCropped)
                # print('---')
                # print(channels)

                # Swap blue and red values (making it RGB, not BGR)
                observation = np.array([(channels[2], channels[1], channels[0])])

                # append in arrays
                bluechannel.append(channels[0])
                greenchannel.append(channels[1])
                redchannel.append(channels[2])
                bluechannel = np.asarray(bluechannel)
                greenchannel = np.asarray(greenchannel)
                redchannel = np.asarray(redchannel)

                countTime = 0
                TimeData = []
                for f1 in CroppedImages:
                    totaltime = countTime / fps
                    TimeData.append(totaltime)
                    countTime += 1

                Ts = TimeData[2] - TimeData[1]

                P = np.c_[redchannel, greenchannel, bluechannel]
                P -= P.mean(axis=0)  # SUBTRACT mean
                P /= P.std(axis=0)  # Standardize data
                P = signal.detrend(P)

                # ICA
                # Mixing matrix Assumed
                A = np.array([[0.5, 1, 0.2],
                              [1, 0.5, 0.4],
                              [0.5, 0.8, 1]])
                X = np.dot(P, A.T)  # Generate observations

                # Compute ICA
                ica = FastICA(n_components=3)
                S_ = ica.fit_transform(X)  # Reconstruct signals
                A_ = ica.mixing_  # Get estimated mixing matrix

                N = len(S_[:, 0])
                time_step = 1 / fps
                psR = np.abs(np.fft.fft(S_[:, 0]))  # **2
                freqsR = np.fft.fftfreq(N, time_step)
                idxR = np.argsort(freqsR)

                psG = np.abs(np.fft.fft(S_[:, 1]))  # **2
                freqsG = np.fft.fftfreq(N, time_step)
                idxG = np.argsort(freqsG)

                psB = np.abs(np.fft.fft(S_[:, 2]))  # **2
                freqsB = np.fft.fftfreq(N, time_step)
                idxB = np.argsort(freqsB)

                # Butter filter # change for various results
                fftGbutter = butter_bandpass_filter(psG, 0.8, 4, fps, order=3)
                fftRbutter = butter_bandpass_filter(psR, 0.8, 4, fps, order=3)
                fftBbutter = butter_bandpass_filter(psB, 0.8, 4, fps, order=3)

                valr = max(fftRbutter)
                valg = max(fftGbutter)
                valb = max(fftBbutter)

                RedHR.append('W' + str(j) + '-' + str(int(60 * valr)))
                GreenHR.append('W' + str(j) + '-' + str(int(60 * valg)))
                BlueHR.append('W' + str(j) + '-' + str(int(60 * valb)))

                # get next window by getting all data and then getting next data for that window while removing rest
                # CroppedImages = OriginalData

                del OriginalData[:step]
                # go back to step 1

                ## get and print data in end
            else:
                break

    # length of the array
    n = len(arr)

    # n must be greater than k
    if n < k:
        print("Invalid")
        return -1

    # Compute sum of first window of size k
    window_sum = ProcessArrrayHR(arr[:k])

    # first sum available
    max_sum = window_sum

    # Compute the sums of remaining windows by
    # removing first element of previous
    # window and adding last element of
    # the current window.
    for i in range(n - k):
        window_sum = window_sum - arr[i] + arr[i + k]
        max_sum = max(window_sum, max_sum)

    return max_sum


# Driver code
arr = [1, 4, 2, 10, 2, 3, 1, 0, 20]
k = 4
print(maxSum(arr, k))

# This code is contributed by Kyle McClay