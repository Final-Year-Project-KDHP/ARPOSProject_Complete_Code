import numpy as np
import scipy.io

class SNR:
    def signaltonoiseDB(self,a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        # return np.where(sd == 0, 0, m / sd)
        return 20 * np.log10(abs(np.where(sd == 0, 0, m / sd)))  # convertin got decibels

    def signaltonoise(self,a, axis=0, ddof=0):
        a = np.asanyarray(a)
        m = a.mean(axis)
        sd = a.std(axis=axis, ddof=ddof)
        return np.where(sd == 0, 0, m / sd)

    def CalculateSNR(self, savepath,fftblue,fftGreen,fftRed,fftGrey,fftIR):
        snr = self.signaltonoise(fftblue)
        bluesnr = f'BLUE SNR ' + str(snr)
        snr = self.signaltonoise(fftGreen)
        greensnr = f'GREEN SNR ' + str(snr)
        snr = self.signaltonoise(fftRed)
        redsnr = f'RED SNR ' + str(snr)
        snr = self.signaltonoise(fftGrey)
        greysnr = f'GREY SNR ' + str(snr)
        snr = self.signaltonoise(fftIR)
        IRsnr = f'IR SNR ' + str(snr)

        f = open(savepath + "\\snr.txt", "w")
        f.writelines(bluesnr + "\n\r")
        f.writelines(greensnr + "\n\r")
        f.writelines(redsnr + "\n\r")
        f.writelines(greysnr + "\n\r")
        f.writelines(IRsnr + "\n\r")
        f.close()