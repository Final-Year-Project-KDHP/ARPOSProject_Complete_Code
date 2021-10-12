import numpy as np
from matplotlib import pyplot as plt
from scipy import signal


def FourierTransform():

    def get_rfft_hr(signal, ino, vallbl, roiReg):
        fft_spec = []

        signal_size = len(signal)
        signal = signal.flatten()
        fft_data = np.fft.rfft(signal)  # FFT
        fft_data = np.abs(fft_data)

        freq = np.fft.rfftfreq(signal_size, 1. / framerate)  # Frequency data

        inds = np.where((freq < minFreq) | (freq > maxFreq))[0]
        fft_data[inds] = 0
        bps_freq = 60.0 * freq
        max_index = np.argmax(fft_data)
        fft_data[max_index] = fft_data[max_index] ** 2

        fft_spec.append(fft_data)
        HR = bps_freq[max_index]

        samplelingErr = bps_freq[max_index + 1] - bps_freq[max_index - 1];
        # print('sampling err : ' + str(samplelingErr))

        figpath = r'C:\Users\pp62\Documents\Python Data\Julia\HRtestColor\pyprg\StudyData\Result\\' + roiReg

        plt.figure(ino)
        plt.plot(fft_data, vallbl)
        plt.title('FFT')
        plt.xlabel('hz')
        plt.ylabel('mag')
        nn = str(ino)
        plt.savefig(figpath + '\\hrfft' + vallbl + nn + '.png')

        return HR

    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = signal.lfilter(b, a, data)
        return y

    def signaltonoise(a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        # return np.where(sd == 0, 0, m / sd)
        return 20 * np.log10(abs(np.where(sd == 0, 0, m / sd)))  # convertin got decibels


    # The FFT of the signal
    sig_fft4 = fft4

    # And the power3 (sig_fft3 is of complex dtype)
    power4 = np.abs(sig_fft4) ** 2

    # The corresponding frequencies
    n = len(fft4)
    sample_freq4 = np.fft.fftfreq(n, d=time_step)

    # Plot the FFT power3
    plt.figure(11)
    plt.plot(sample_freq4, power4, 'black')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('power')

    plt.savefig(figpath + '\\fftIR.png')

    ##################  BPM   ##################
    mY = np.abs(power4)  # Find magnitude
    peakY = np.max(mY)  # Find max peak
    locY = np.argmax(mY)  # Find its location
    frqY = sample_freq4[locY]  # Get the actual frequency value

    print('Method 1, BPM IR')
    print(peakY)
    print('frq val : ' + str(frqY))
    print('BPM : ' + str(frqY * 60))
    BPM = (locY) / N * 30 * 60
    print('BPM 2  : ' + str(BPM))

    ################## Grey

    # The FFT of the signal
    sig_fft3 = fft3

    # And the power3 (sig_fft3 is of complex dtype)
    power3 = np.abs(sig_fft3) ** 2

    # The corresponding frequencies
    n = len(fft3)
    sample_freq3 = np.fft.fftfreq(n, d=time_step)

    # Plot the FFT power3
    plt.figure(12)
    plt.plot(sample_freq3, power3, 'grey')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('power')

    plt.savefig(figpath + '\\fftGrey.png')

    ##################  BPM   ##################
    mY = np.abs(power3)  # Find magnitude
    peakY = np.max(mY)  # Find max peak
    locY = np.argmax(mY)  # Find its location
    frqY = sample_freq3[locY]  # Get the actual frequency value

    print('Method 1, BPM Grey')
    print(peakY)
    print('frq val : ' + str(frqY))
    print('BPM : ' + str(frqY * 60))
    BPM = (locY) / N * 30 * 60
    print('BPM 2  : ' + str(BPM))

    ################## Red

    # The FFT of the signal
    sig_fft2 = fft2

    # And the power2 (sig_fft2 is of complex dtype)
    power2 = np.abs(sig_fft2) ** 2

    # The corresponding frequencies
    n = len(fft2)
    sample_freq2 = np.fft.fftfreq(n, d=time_step)

    # Plot the FFT power2
    plt.figure(13)
    plt.plot(sample_freq2, power2, 'red')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('power')

    plt.savefig(figpath + '\\fftRed.png')

    ##################  BPM   ##################
    mY = np.abs(power2)  # Find magnitude
    peakY = np.max(mY)  # Find max peak
    locY = np.argmax(mY)  # Find its location
    frqY = sample_freq2[locY]  # Get the actual frequency value

    print('Method 1, BPM Red')
    print(peakY)
    print('frq val : ' + str(frqY))
    print('BPM : ' + str(frqY * 60))
    BPM = (locY) / N * 30 * 60
    print('BPM 2  : ' + str(BPM))

    ################## Green

    # The FFT of the signal
    sig_fft1 = fft1

    # And the power1 (sig_fft1 is of complex dtype)
    power1 = np.abs(sig_fft1) ** 2

    # The corresponding frequencies
    n = len(fft1)
    sample_freq1 = np.fft.fftfreq(n, d=time_step)

    # Plot the FFT power1
    plt.figure(14)
    plt.plot(sample_freq1, power1, 'green')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('power')

    plt.savefig(figpath + '\\fftGreen.png')
    ##################  BPM   ##################
    mY = np.abs(power1)  # Find magnitude
    peakY = np.max(mY)  # Find max peak
    locY = np.argmax(mY)  # Find its location
    frqY = sample_freq1[locY]  # Get the actual frequency value

    print('####### Method 1, BPM Green ####### ')
    #print(peakY)
    print('frq val : ' + str(frqY))
    print('BPM : ' + str(frqY * 60))
    BPM = (locY) / N * 30 * 60
    print('BPM 2  : ' + str(BPM))
    ################## Blue

    # The FFT of the signal
    sig_fft = fft

    # And the power (sig_fft is of complex dtype)
    power = np.abs(sig_fft) ** 2

    # The corresponding frequencies
    n = len(fft)
    sample_freq = np.fft.fftfreq(n, d=time_step)

    # Plot the FFT power
    plt.figure(15)
    plt.plot(sample_freq, power, 'blue')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('power')

    plt.savefig(figpath + '\\fftBlue.png')

    ##################  BPM   ##################
    mY = np.abs(power)  # Find magnitude
    peakY = np.max(mY)  # Find max peak
    locY = np.argmax(mY)  # Find its location
    frqY = sample_freq[locY]  # Get the actual frequency value

    print(' ####### Method 1, BPM blue ####### ')
    #print(peakY)
    print('frq val : ' + str(frqY))
    print('BPM : ' + str(frqY * 60))
    BPM = (locY) / N * 30 * 60
    print('BPM 2  : ' + str(BPM))

    ################## FFT method 2  ##################
    ##################  BLUE  ##################
    N = len(S_[:, 0])
    T = time_step
    x = time_list
    y = S_[:, 0]
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(N, T)[:N // 2]
    plt.figure(16)
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]), 'blue')
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('power')
    plt.savefig(figpath + '\\fftBlueM2.png')
    # pos_mask = np.where(xf > 0)
    # freqs = xf[pos_mask]
    # peak_freq = freqs[power[pos_mask].argmax()]
    # print('Peak frequency : ' + str(peak_freq))
    # print('bmp frequency : ' + str(peak_freq * 60))
    # # snr = signaltonoise(yf)
    # print(f'blue SNR, {name}', snr)


    ##################  BPM   ##################
    # mY = np.abs(yf)  # Find magnitude
    # peakY = np.max(mY)  # Find max peak
    # locY = np.argmax(mY)  # Find its location
    # frqY = xf[locY]  # Get the actual frequency value
    #
    # print(' ####### Method 2, Blue IR #######  ')
    # #print(peakY)
    # print('frq val : ' + str(frqY))
    # print('BPM : ' + str(frqY*60))
    BPM = (locY) / N * 30 * 60
    print('BPM 2  : ' + str(BPM))

    ##################  GREEN  ##################
    N = len(S_[:, 1])
    T = time_step
    x = time_list
    y = S_[:, 1]
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(N, T)[:N // 2]
    plt.figure(17)
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]), 'green')
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('power')
    plt.savefig(figpath + '\\fftGreenM2.png')
    # pos_mask = np.where(xf > 0)
    # freqs = xf[pos_mask]
    # peak_freq = freqs[power[pos_mask].argmax()]
    # print('Peak frequency : ' + str(peak_freq))
    # print('bmp frequency : ' + str(peak_freq * 60))
    # snr = signaltonoise(yf)
    # print(f'Green SNR, {name}', snr)

    ##################  BPM   ##################
    mY = np.abs(yf)  # Find magnitude
    peakY = np.max(mY)  # Find max peak
    locY = np.argmax(mY)  # Find its location
    # frqY = xf[locY]  # Get the actual frequency value
    #
    # print(' ####### Method 2, Green IR  #######  ')
    # #print(peakY)
    # print('frq val : ' + str(frqY))
    # print('BPM : ' + str(frqY*60))
    BPM = (locY) / N * 30 * 60
    print('BPM 2  : ' + str(BPM))


    ##################  RED  ##################
    N = len(S_[:, 2])
    T = time_step
    x = time_list
    y = S_[:, 2]
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(N, T)[:N // 2]
    plt.figure(18)
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]), 'red')
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('power')
    plt.savefig(figpath + '\\fftRedM2.png')
    # pos_mask = np.where(xf > 0)
    # freqs = xf[pos_mask]
    # peak_freq = freqs[power[pos_mask].argmax()]
    # print('Peak frequency : ' + str(peak_freq))
    # print('bmp frequency : ' + str(peak_freq * 60))
    # snr = signaltonoise(yf)
    # print(f'Red SNR, {name}', snr)

    ##################  BPM   ##################
    mY = np.abs(yf)  # Find magnitude
    peakY = np.max(mY)  # Find max peak
    locY = np.argmax(mY)  # Find its location
    frqY = xf[locY]  # Get the actual frequency value

    print(' ####### Method 2, Red IR  ####### ')
    #print(peakY)
    print('frq val : ' + str(frqY))
    print('BPM : ' + str(frqY * 60))
    BPM = (locY) / N * 30 * 60
    print('BPM 2  : ' + str(BPM))

    ##################  Grey  ##################
    N = len(S_[:, 3])
    T = time_step
    x = time_list
    y = S_[:, 3]
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(N, T)[:N // 2]
    plt.figure(19)
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]), 'grey')
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('power')
    plt.savefig(figpath + '\\fftGreyM2.png')
    # pos_mask = np.where(xf > 0)
    # freqs = xf[pos_mask]
    # peak_freq = freqs[power[pos_mask].argmax()]
    # print('Peak frequency : ' + str(peak_freq))
    # print('bmp frequency : ' + str(peak_freq * 60))

    ##################  BPM   ##################
    mY = np.abs(yf)  # Find magnitude
    peakY = np.max(mY)  # Find max peak
    locY = np.argmax(mY)  # Find its location
    frqY = xf[locY]  # Get the actual frequency value

    print(' ####### Method 2, Grey IR #######  ')
    #print(peakY)
    print('frq val : ' + str(frqY))
    print('BPM : ' + str(frqY*60))
    BPM = (locY) / N * 30 * 60
    print('BPM 2  : ' + str(BPM))

    ##################  IR  ##################
    N = len(S_[:, 4])
    T = time_step
    x = time_list
    y = S_[:, 4]
    yf = np.fft.fft(y)
    xf = np.fft.fftfreq(N, T)[:N // 2]
    plt.figure(20)
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]), 'black')
    plt.grid()
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('power')
    plt.savefig(figpath + '\\fftIrM2.png')

    #BPM = (largest_index - 1)/N * Fs * 60

    ##################  BPM   ##################
    mY = np.abs(yf)  # Find magnitude
    peakY = np.max(mY)  # Find max peak
    locY = np.argmax(mY)  # Find its location
    frqY = xf[locY]  # Get the actual frequency value


    print(' ####### Method 2, BPM IR  ####### ')
    #print(peakY)
    print('frq val : ' + str(frqY))
    print('BPM : ' + str(frqY*60))
    BPM = (locY) / N * 30 * 60
    print('BPM 2  : ' + str(BPM))
