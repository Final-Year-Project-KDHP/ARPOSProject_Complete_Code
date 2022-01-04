import numpy as np
from sklearn.decomposition import FastICA, PCA
from scipy import signal
from scipy.fftpack import fft, ifft
from scipy.fft import fft, fftfreq, fftshift
import scipy.fftpack as fftpack
import jade

from SaveGraphs import Plots

from numpy import abs, append, arange, arctan2, argsort, array, concatenate, \
    cos, diag, dot, eye, float32, float64, loadtxt, matrix, multiply, ndarray, \
    newaxis, savetxt, sign, sin, sqrt, zeros
from numpy.linalg import eig, pinv

class AlgorithmCollection:
    #constatns
    S_ =[]
    X = []

    objPlots = Plots()
    #FfT
    fftblue = [] # blue
    fftGreen = [] # green
    fftRed = [] # red
    fftGrey = [] # gr
    fftIR = [] # ir

    SavefigPath = ''

    fft_MaxIndex=0
    fft_MaxVal=0
    fft_frq=0

    Colorfreq = []
    IRfreq = []
    # EstimatedFPS = 0
    ColorEstimatedFPS = 0
    IREstimatedFPS = 0
    NumberofSample = 0

    powerfftblue = []
    powerfftGreen = []
    powerfftRed = []
    powerfftGrey = []
    powerfftIR = []


    #########------FFT------#############
    '''
    '''
    def Apply_FFT_WithoutPower_M4_eachsignal(self,Blue, Green, Red, Grey, IR,ignoregray): #rfft, abs, get power and its freq

        self.apply_rfft(Blue, Green, Red, Grey, IR,ignoregray)
        self.apply_Abs(ignoregray)
        self.Colorfreq = np.fft.rfftfreq(len(Grey),1/self.ColorEstimatedFPS)
        self.IRfreq = np.fft.rfftfreq(len(IR),1/self.IREstimatedFPS)

        return self.fftblue, self.fftGreen, self.fftRed, self.fftGrey, self.fftIR, self.Colorfreq, self.IRfreq

    def Apply_FFT_M2_eachsignal(self,Blue, Green, Red, Grey, IR, ignoregray): #rfft, abs, get power and its freq

        N = len(Grey)  # Number of data points
        '''perform fourier transform'''
        self.fftblue = fftpack.fft(Blue)  # blue
        self.fftblue = 2.0 / N * np.abs(self.fftblue[0:N // 2])

        self.fftGreen = fftpack.fft(Green)  # green
        self.fftGreen = 2.0 / N * np.abs(self.fftGreen[0:N // 2])

        self.fftRed = fftpack.fft(Red)  # red
        self.fftRed = 2.0 / N * np.abs(self.fftRed[0:N // 2])

        index = 3
        if (not ignoregray):
            self.fftGrey = fftpack.fft(Grey)  # gr
            self.fftGrey = 2.0 / N * np.abs(self.fftGrey[0:N // 2])
            index = 4  # ir

        self.fftIR = fftpack.fft(IR)  # ir
        self.fftIR = 2.0 / N * np.abs(self.fftIR[0:N // 2])

        # self.freq = np.linspace(0.0, 1 / (T * 2), N // 2)  # replot complex data over freq domain
        T = 1. / self.ColorEstimatedFPS  # delta between frames (s)
        self.Colorfreq = np.linspace(0.0, 1 / (T * 2), N // 2)  # replot complex data over freq domain
        T = 1. / self.IREstimatedFPS  # delta between frames (s)
        self.IRfreq = np.linspace(0.0, 1 / (T * 2), N // 2)  # replot complex data over freq domain

        return self.fftblue, self.fftGreen, self.fftRed, self.fftGrey, self.fftIR, self.Colorfreq, self.IRfreq

    def Apply_FFT_M1_byeachsignal(self,Blue, Green, Red, Grey, IR,ignoregray): #rfft, abs, get power and its freq

        self.apply_rfft(Blue, Green, Red, Grey, IR,ignoregray)
        self.apply_Abs(ignoregray)
        self.getPower(ignoregray)
        # self.freq = np.fft.rfftfreq(len(yf),1/self.EstimatedFPS)
        self.Colorfreq = np.fft.rfftfreq(len(Grey),1/self.ColorEstimatedFPS)
        self.IRfreq = np.fft.rfftfreq(len(Grey),1/self.IREstimatedFPS)

        return self.powerfftblue, self.powerfftGreen, self.powerfftRed, self.powerfftGrey, self.powerfftIR, self.Colorfreq, self.IRfreq

    def ApplyFFT9(self,Blue, Green, Red, Grey, IR,ignoregray):
        # obtain the frequencies using scipy function
        self.get_Freq(len(Grey),len(IR))

        #### blue
        blue_x=Blue
        # FFT the signal
        blue_sig_fft = fft(blue_x)

        #### GREEN
        green_x=Green
        # FFT the signal
        green_sig_fft = fft(green_x)

        #### RED
        red_x=Red
        # FFT the signal
        red_sig_fft = fft(red_x)

        #### grey
        grey_x=Grey
        # FFT the signal
        grey_sig_fft = fft(grey_x)

        #### Ir
        ir_x=IR
        # FFT the signal
        ir_sig_fft = fft(ir_x)
        #
        # else:
        #     #### Ir
        #     ir_x=S[:, 3]
        #     # FFT the signal
        #     ir_sig_fft = fft(ir_x)
        #     grey_sig_fft = []

        return blue_sig_fft,green_sig_fft,red_sig_fft,grey_sig_fft,ir_sig_fft,self.Colorfreq, self.IRfreq

    def Apply_FFT_M6_Individual(self, Blue, Green, Red, Grey, IR,ignoregray): #original 5
        # aply fft
        self.fftblue = np.fft.fft(Blue)  # blue
        self.fftGreen = np.fft.fft(Green)  # green
        self.fftRed = np.fft.fft(Red)  # red
        index = 3
        if(not ignoregray):
            self.fftGrey = np.fft.fft(Grey)  # gr
            index = 4
        self.fftIR = np.fft.fft(IR)  # ir

        self.fftblue = self.fftblue / np.sqrt(len(self.fftblue))
        self.fftGreen = self.fftGreen / np.sqrt(len(self.fftGreen))
        self.fftRed = self.fftRed / np.sqrt(len(self.fftRed))
        if(not ignoregray):
            self.fftGrey = self.fftGrey / np.sqrt(len(self.fftGrey))
        self.fftIR = self.fftIR / np.sqrt(len(self.fftIR))

        xf, xf2 = self.get_Freq(len(Grey),len(IR))
        yplotIR = abs(self.fftIR) #with and without abs
        yplotRed = abs(self.fftRed)
        yplotGreen = abs(self.fftGreen)
        yplotBlue = abs(self.fftblue)
        if(not ignoregray):
            yplotGrey = abs(self.fftGrey)
        else:
            yplotGrey= []

        return yplotBlue,yplotGreen,yplotRed,yplotGrey,yplotIR,xf,xf2

    def Apply_FFT_M5_Individual(self,  Blue, Green, Red, Grey, IR,ignoregray): #original 4
        #aply fft
        self.fftblue = np.fft.fft(Blue)  # blue
        self.fftGreen = np.fft.fft(Green)  # green
        self.fftRed = np.fft.fft(Red)  # red
        if(not ignoregray):
            self.fftGrey = np.fft.fft(Grey)  # gr  # gr

        self.fftIR = np.fft.fft(IR)  # ir
        self.fftblue = self.fftblue / np.sqrt(len(self.fftblue))
        self.fftGreen = self.fftGreen / np.sqrt(len(self.fftGreen))
        self.fftRed = self.fftRed / np.sqrt(len(self.fftRed))

        if(not ignoregray):
            self.fftGrey = self.fftGrey / np.sqrt(len(self.fftGrey))

        self.fftIR = self.fftIR / np.sqrt(len(self.fftIR))

        xf,xf2 = self.get_Freq(len(Grey),len(IR))
        xf = np.fft.fftshift(xf)
        xf2 = np.fft.fftshift(xf2)

        yplotIR = np.fft.fftshift(abs(self.fftIR))
        yplotRed = np.fft.fftshift(abs(self.fftRed))
        yplotGreen = np.fft.fftshift(abs(self.fftGreen))
        yplotBlue = np.fft.fftshift(abs(self.fftblue))
        if(not ignoregray):
            yplotGrey = np.fft.fftshift(abs(self.fftGrey))
        else:
            yplotGrey = []

        return yplotBlue,yplotGreen,yplotRed,yplotGrey,yplotIR,xf,xf2

    def Apply_FFT_forpreprocessed(self,Blue, Green, Red, Grey, IR,ignoregray): #rfft, abs, get power and its freq
        '''Apply Fast Fourier Transform'''
        # self.freq = float(self.EstimatedFPS) / L * np.arange(L / 2 + 1)

        # self.freq = np.fft.rfftfreq(len(yf),1/self.EstimatedFPS)
        self.Colorfreq = np.fft.rfftfreq(len(Grey),1/self.ColorEstimatedFPS)
        self.IRfreq = np.fft.rfftfreq(len(IR),1/self.IREstimatedFPS)
        # self.fftblue =S_[:, 0]
        raw_fft = np.fft.rfft(Blue * self.ColorEstimatedFPS)
        self.fftblue = np.abs(raw_fft) ** 2

        raw_fft = np.fft.rfft( Green * self.ColorEstimatedFPS)
        self.fftGreen = np.abs(raw_fft) ** 2

        raw_fft = np.fft.rfft(Red * self.ColorEstimatedFPS)
        self.fftRed = np.abs(raw_fft) ** 2

        index=3
        if(not ignoregray):
            raw_fft = np.fft.rfft(Grey * self.ColorEstimatedFPS)
            self.fftGrey = np.abs(raw_fft) ** 2
            index=4
        else:
            self.fftGrey = []

        raw_fft = np.fft.rfft(IR * self.IREstimatedFPS)
        self.fftIR = np.abs(raw_fft) ** 2

        return self.fftblue, self.fftGreen, self.fftRed, self.fftGrey, self.fftIR, self.Colorfreq, self.IRfreq

    #########------FFT------#############

    def apply_rfft_All(self, S_):
        y = S_
        yf = np.fft.rfft(y)
        return yf

    def apply_rfft(self, Blue, Green, Red, Grey, IR,ignoregray):
        self.fftblue = np.fft.rfft(Blue)  # blue
        self.fftGreen = np.fft.rfft(Green)  # green
        self.fftRed = np.fft.rfft(Red)  # red
        index =3
        if(not ignoregray):
            self.fftGrey = np.fft.rfft(Grey)  # gr
            index =4  # ir
        self.fftIR = np.fft.rfft(IR)  # ir

    def apply_fft(self, S_,ignoregray):
        self.fftblue = np.fft.fft(S_[:, 0])  # blue
        self.fftGreen = np.fft.fft(S_[:, 1])  # green
        self.fftRed = np.fft.fft(S_[:, 2])  # red
        index = 3
        if(not ignoregray):
            self.fftGrey = np.fft.fft(S_[:, 3])  # gr
            index =4  # ir
        self.fftIR = np.fft.fft(S_[:, index])  # ir

    def apply_sqrt_AfterFFt(self,ignoregray):
        self.fftblue = self.fftblue / np.sqrt(len(self.fftblue))
        self.fftGreen = self.fftGreen / np.sqrt(len(self.fftGreen))
        self.fftRed = self.fftRed / np.sqrt(len(self.fftRed))
        if(not ignoregray):
            self.fftGrey = self.fftGrey / np.sqrt(len(self.fftGrey))
        self.fftIR = self.fftIR / np.sqrt(len(self.fftIR))

    def apply_fft_All(self, S_):
        y = S_
        yf = np.fft.fft(y)
        return yf

    def apply_Abs_All(self,S):
        S = np.abs(S)
        return S

    def apply_Abs(self,ignoregray):
        self.fftblue = np.abs(self.fftblue)  # blue
        self.fftGreen = np.abs(self.fftGreen)  # green
        self.fftRed = np.abs(self.fftRed )  # red
        if(not ignoregray):
            self.fftGrey = np.abs(self.fftGrey)  # gr
        self.fftIR = np.abs(self.fftIR)  # ir

    def getPower_All(self,S):
        S = np.abs(S) ** 2
        return S

    def getPower(self,ignoregray):
        self.powerfftblue = np.abs(self.fftblue) ** 2  # blue
        self.powerfftGreen = np.abs(self.fftGreen) ** 2  # green
        self.powerfftRed = np.abs(self.fftRed) ** 2  # red
        if(not ignoregray):
            self.powerfftGrey = np.abs(self.fftGrey) ** 2  # gr
        self.powerfftIR = np.abs(self.fftIR) ** 2  # ir

    def get_Freq_bySamplesize(self,N,T):
        xf = np.fft.fftfreq(N, T)[:N // 2]
        return xf

    def get_Freq(self,ColorL, IRL):
        time_step = 1 / self.ColorEstimatedFPS
        self.Colorfreq = np.fft.fftfreq(ColorL, d=time_step)
        time_step = 1 / self.IREstimatedFPS
        self.IRfreq = np.fft.fftfreq(IRL, d=time_step)
        return self.Colorfreq, self.IRfreq

    #########------Algorithms------#############

    def ApplyICA(self,S,compts):
        ica = FastICA(n_components=compts, max_iter=1000)#n_components=3,max_iter=10000 whiten=True, max_iter=1000
        np.random.seed(0)
        S /= S.std(axis=0)  # Standardize data
        self.X = S
        # X = np.dot(self.S, A.T)
        self.S_ = ica.fit(self.X).transform(self.X)  # Get the estimated sources
        A_ = ica.mixing_  # Get estimated mixing matrix , setting seed , numpy

        return self.S_

    def ApplyPCA(self,X,compts):
        X /= X.std(axis=0)  # Standardize data
        pca = PCA(n_components=compts)
        pca.fit(X)
        X_new = pca.fit_transform(X)
        #v=pca.explained_variance_ratio_
        return X_new

    def jadeOriginal(self,X,m):

        ICA = jade.main(X,m)
        return ICA

    def jadeOptimised(self,X, m=None, verbose=False):

        a=0
        assert isinstance(X, ndarray), \
            "X (input data matrix) is of the wrong type (%s)" % type(X)
        origtype = X.dtype  # remember to return matrix B of the same type
        X = matrix(X.astype(float64))
        assert X.ndim == 2, "X has %d dimensions, should be 2" % X.ndim
        assert (verbose == True) or (verbose == False), \
            "verbose parameter should be either True or False"

        # GB: n is number of input signals, T is number of samples
        [T,n] = X.shape
        assert n < T, "number of sensors must be smaller than number of samples"

        # Number of sources defaults to number of sensors
        if m == None:
            m = n
        assert m <= n, \
            "number of sources (%d) is larger than number of sensors (%d )" % (m, n)

        if verbose:
            print("jade -> Looking for %d sources" % m)
            print("jade -> Removing the mean value")
        X -= X.mean(1)

        # whitening & projection onto signal subspace
        # ===========================================

        if verbose: print("jade -> Whitening the data")
        # An eigen basis for the sample covariance matrix
        [D, U] = eig((X * X.T) / float(T))
        # Sort by increasing variances
        k = D.argsort()
        Ds = D[k]
        # The m most significant princip. comp. by decreasing variance
        PCs = arange(n - 1, n - m - 1, -1)

        # --- PCA  ----------------------------------------------------------
        # At this stage, B does the PCA on m components
        B = U[:, k[PCs]].T

        # --- Scaling  ------------------------------------------------------
        # The scales of the principal components
        scales = sqrt(Ds[PCs])
        # Now, B does PCA followed by a rescaling = sphering
        B = diag(1. / scales) * B
        # --- Sphering ------------------------------------------------------
        X = B * X

        # We have done the easy part: B is a whitening matrix and X is white.

        del U, D, Ds, k, PCs, scales

        # NOTE: At this stage, X is a PCA analysis in m components of the real
        # data, except that all its entries now have unit variance. Any further
        # rotation of X will preserve the property that X is a vector of
        # uncorrelated components. It remains to find the rotation matrix such
        # that the entries of X are not only uncorrelated but also `as independent
        # as possible". This independence is measured by correlations of order
        # higher than 2. We have defined such a measure of independence which 1)
        # is a reasonable approximation of the mutual information 2) can be
        # optimized by a `fast algorithm" This measure of independence also
        # corresponds to the `diagonality" of a set of cumulant matrices. The code
        # below finds the `missing rotation " as the matrix which best
        # diagonalizes a particular set of cumulant matrices.

        # Estimation of the cumulant matrices
        # ===================================

        if verbose: print("jade -> Estimating cumulant matrices")

        # Reshaping of the data, hoping to speed up things a little bit...
        X = X.T
        # Dim. of the space of real symm matrices
        dimsymm = (m * (m + 1)) / 2
        # number of cumulant matrices
        nbcm = int(dimsymm)
        # Storage for cumulant matrices
        #CM = matrix(zeros([m, m * nbcm], dtype=float64))
        CM = matrix(zeros([int(m),int(m * nbcm) ], dtype=float64))
        R = matrix(np.eye(m, dtype=float64))
        # Temp for a cum. matrix
        Qij = matrix(np.zeros([m, m], dtype=float64))
        # Temp
        Xim = np.zeros(m, dtype=float64)
        # Temp
        Xijm = np.zeros(m, dtype=float64)
        Uns		= np.ones(m, dtype=float64)   # for convenience

        # I am using a symmetry trick to save storage. I should write a short note
        # one of these days explaining what is going on here.
        # will index the columns of CM where to store the cum. mats.
        Range = arange(m)

        for im in range(m):
            Xim = X[:, im]
            Xijm = multiply(Xim, Xim)
            # Note to myself: the -R on next line can be removed: it does not affect
            # the joint diagonalization criterion
            Qij = multiply(Xijm, X).T * X / float(T) - R - 2 * (R[:, im] * R[:, im].T)
            CM[:, Range] = Qij
            Range = Range + m
            for jm in range(im):
                Xijm = multiply(Xim, X[:, jm])
                Qij = sqrt(2) * (multiply(Xijm, X).T * X / float(T)
                                 - R[:, im] * R[:, jm].T - R[:, jm] * R[:, im].T)
                CM[:, Range] = Qij
                Range = Range + m

        # Now we have nbcm = m(m+1)/2 cumulants matrices stored in a big
        # m x m*nbcm array.

        # Joint diagonalization of the cumulant matrices
        # ==============================================

        V = matrix(np.eye(m, dtype=float64))
        if 0:
            if verbose: print("jade -> Total of %d Givens rotations")
            [V, D] = eig(CM[:, 1: m])

            for u in range(1, m, m * nbcm):
                CM[:, u: u + m - 1] = CM[:, u: u + m - 1] * V
            CM = V.T * CM
        else:
            V = np.eye(m)
        #
        Diag = zeros(m, dtype=float64)
        On = 0.0
        Range = arange(m)
        for im in range(int(nbcm)):
            Diag = diag(CM[:, Range])
            On = On + (Diag * Diag).sum(axis=0)
            Range = Range + m
        Off = (multiply(CM, CM).sum(axis=0)).sum(axis=1) - On
        # A statistically scaled threshold on `small" angles
        seuil = 1.0e-6 / sqrt(T)
        # sweep number
        encore = True
        sweep = 0
        # Total number of rotations
        updates = 0
        # Number of rotations in a given seep
        upds = 0
        g = zeros([2, nbcm], dtype=float64)
        gg = zeros([2, 2], dtype=float64)
        G = zeros([2, 2], dtype=float64)
        c = 0
        s = 0
        ton = 0
        toff = 0
        theta = 0
        Gain = 0

        # Joint diagonalization proper
        # ============================
        if verbose: print("jade -> Contrast optimization by joint diagonalization")
        updates = updates + upds
        # while encore:
        #     encore = False
        #     if verbose: print("jade -> Sweep #%3d" % sweep)
        #     sweep = sweep + 1
        #     upds = 0
        #     Vkeep = V
        #
        #     for p in range(m - 1):
        #         for q in range(p + 1, m):
        #
        #             Ip = arange(p, m * nbcm, m)
        #             Iq = arange(q, m * nbcm, m)
        #
        #             # computation of Givens angle
        #             g = concatenate([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
        #             gg = dot(g, g.T)
        #             ton = gg[0, 0] - gg[1, 1]
        #             toff = gg[0, 1] + gg[1, 0]
        #             theta = 0.5 * arctan2(toff, ton + sqrt(ton * ton + toff * toff))
        #             Gain = (sqrt(ton * ton + toff * toff) - ton) / 4.0
        #
        #             # Givens update
        #             if abs(theta) > seuil:
        #                 encore = True
        #                 upds = upds + 1
        #                 c = cos(theta)
        #                 s = sin(theta)
        #                 G = matrix([[c, -s], [s, c]])
        #                 pair = array([p, q])
        #                 V[:, pair] = V[:, pair] * G
        #                 CM[pair, :] = G.T * CM[pair, :]
        #                 CM[:, concatenate([Ip, Iq])] = \
        #                     append(c * CM[:, Ip] + s * CM[:, Iq], -s * CM[:, Ip] + c * CM[:, Iq], \
        #                            axis=1)
        #                 On = On + Gain
        #                 Off = Off - Gain
        #
        #     if verbose: print("completed in %d rotations" % upds)
        #     updates = updates + upds



        # A separating matrix
        # ===================
        B = V.T * B

        # Permute the rows of the separating matrix B to get the most energetic
        # components first. Here the **signals** are normalized to unit variance.
        # Therefore, the sort is according to the norm of the columns of
        # A = pinv(B)

        if verbose: print("jade -> Sorting the components")

        A = pinv(B)

        keys = array(argsort(multiply(A, A).sum(axis=0)[0]))[0]
        B = B[keys, :]
        # % Is this smart ?
        B = B[::-1, :]

        if verbose: print("jade -> Fixing the signs")
        b = B[:, 0]
        # just a trick to deal with sign == 0
        signs = array(sign(sign(b) + 0.1).T)[0]
        B = diag(signs) * B

        return B.astype(origtype)














