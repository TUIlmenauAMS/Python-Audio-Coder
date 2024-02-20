# Programs to implement a psycho-acoustic model
# Using a matrix for the spreading function (faster)
# Gerald Schuller, Nov. 2016
# torch version from Renato Profeta, Feb 2024

import torch


def f_SP_dB_torch(maxfreq, nfilts):
    # usage: spreadingfunctionmatdB=f_SP_dB(maxfreq,nfilts)
    # computes the spreading function protoype, in the Bark scale.
    # Arguments: maxfreq: half the sampling freqency
    # nfilts: Number of subbands in the Bark domain, for instance 64
    # upper end of our Bark scale:22 Bark at 16 kHz
    maxbark = hz2bark_torch(maxfreq)
    # Number of our Bark scale bands over this range: nfilts=64
    spreadingfunctionBarkdB = torch.zeros(2*nfilts)
    # Spreading function prototype, "nfilts" bands for lower slope
    spreadingfunctionBarkdB[0:nfilts] = torch.linspace(
        -maxbark*27, -8, nfilts)-23.5
    # "nfilts" bands for upper slope:
    spreadingfunctionBarkdB[nfilts:2 *
                            nfilts] = torch.linspace(0, -maxbark*12.0, nfilts)-23.5
    return spreadingfunctionBarkdB


def spreadingfunctionmat_torch(spreadingfunctionBarkdB, alpha, nfilts):
    # Turns the spreading prototype function into a matrix of shifted versions.
    # Convert from dB to "voltage" and include alpha exponent
    # nfilts: Number of subbands in the Bark domain, for instance 64
    spreadingfunctionBarkVoltage = 10.0**(
        spreadingfunctionBarkdB/20.0*alpha)
    # Spreading functions for all bark scale bands in a matrix:
    spreadingfuncmatrix = torch.zeros((nfilts, nfilts))
    for k in range(nfilts):
        spreadingfuncmatrix[k, :] = spreadingfunctionBarkVoltage[(
            nfilts-k):(2*nfilts-k)]
    return spreadingfuncmatrix


def maskingThresholdBark_torch(mXbark, spreadingfuncmatrix, alpha, fs, nfilts):
    # Computes the masking threshold on the Bark scale with non-linear superposition
    # usage: mTbark=maskingThresholdBark(mXbark,spreadingfuncmatrix,alpha)
    # Arg: mXbark: magnitude of FFT spectrum, on the Bark scale
    # spreadingfuncmatrix: spreading function matrix from function spreadingfunctionmat
    # alpha: exponent for non-linear superposition (eg. 0.6),
    # fs: sampling freq., nfilts: number of Bark subbands
    # nfilts: Number of subbands in the Bark domain, for instance 64
    # Returns: mTbark: the resulting Masking Threshold on the Bark scale

    # Compute the non-linear superposition:
    mTbark = torch.matmul(mXbark**alpha, spreadingfuncmatrix**alpha)
    # apply the inverse exponent to the result:
    mTbark = mTbark**(1.0/alpha)
    # Threshold in quiet:
    maxfreq = fs/2.0
    maxbark = hz2bark_torch(maxfreq)
    step_bark = maxbark/(nfilts-1)
    barks = torch.arange(0, nfilts)*step_bark
    # convert the bark subband frequencies to Hz:
    f = bark2hz_torch(barks)+1e-6
    # Threshold of quiet in the Bark subbands in dB:
    LTQ = torch.clip((3.64*(f/1000.)**-0.8 - 6.5*torch.exp(-0.6*(f/1000.-3.3)**2.)
                      + 1e-3*((f/1000.)**4.)), -20, 120)
    # Maximum of spreading functions and hearing threshold in quiet:
    a = mTbark
    b = 10.0**((LTQ-60)/20)
    mTbark = torch.max(a, b)
    return mTbark


def hz2bark_torch(f):
    """ Usage: Bark=hz2bark(f)
          f    : (ndarray)    Array containing frequencies in Hz.
      Returns  :
          Brk  : (ndarray)    Array containing Bark scaled values.
      """
    if not torch.is_tensor(f):
        f = torch.tensor(f)

    Brk = 6. * torch.arcsinh(f/600.)
    return Brk


def bark2hz_torch(Brk):
    """ Usage:
      Hz=bark2hs(Brk)
      Args     :
          Brk  : (ndarray)    Array containing Bark scaled values.
      Returns  :
          Fhz  : (ndarray)    Array containing frequencies in Hz.
      """
    if not torch.is_tensor(Brk):
        Brk = torch.tensor(Brk)
    Fhz = 600. * torch.sinh(Brk/6.)
    return Fhz


def mapping2barkmat_torch(fs, nfilts, nfft):
    # Constructing mapping matrix W which has 1's for each Bark subband, and 0's else
    # usage: W=mapping2barkmat(fs, nfilts,nfft)
    # arguments: fs: sampling frequency
    # nfilts: number of subbands in Bark domain
    # nfft: number of subbands in fft
    # upper end of our Bark scale:22 Bark at 16 kHz
    maxbark = hz2bark_torch(fs/2)
    nfreqs = nfft/2
    step_bark = maxbark/(nfilts-1)
    binbark = hz2bark_torch(
        torch.linspace(0, (nfft/2), (nfft//2)+1)*fs/nfft)
    W = torch.zeros((nfilts, nfft))
    for i in range(nfilts):
        W[i, 0:int(nfft/2)+1] = (torch.round(binbark/step_bark) == i)
    return W


def mapping2bark_torch(mX, W, nfft):
    # Maps (warps) magnitude spectrum vector mX from DFT to the Bark scale
    # arguments: mX: magnitude spectrum from fft
    # W: mapping matrix from function mapping2barkmat
    # nfft: : number of subbands in fft
    # returns: mXbark, magnitude mapped to the Bark scale
    nfreqs = int(nfft/2)
    # Here is the actual mapping, suming up powers and conv. back to Voltages:
    mXbark = (torch.matmul(
        torch.abs(mX[:nfreqs])**2.0, W[:, :nfreqs].T))**(0.5)
    return mXbark


def mappingfrombarkmat_torch(W, nfft):
    # Constructing inverse mapping matrix W_inv from matrix W for mapping back from bark scale
    # usuage: W_inv=mappingfrombarkmat(Wnfft)
    # argument: W: mapping matrix from function mapping2barkmat
    # nfft: : number of subbands in fft
    nfreqs = int(nfft/2)
    W_inv = torch.matmul(torch.diag(
        (1.0/(torch.sum(W, 1)+1e-6))**0.5), W[:, 0:nfreqs + 1]).T
    return W_inv

# -------------------


def mappingfrombark_torch(mTbark, W_inv, nfft):
    # usage: mT=mappingfrombark(mTbark,W_inv,nfft)
    # Maps (warps) magnitude spectrum vector mTbark in the Bark scale
    # back to the linear scale
    # arguments:
    # mTbark: masking threshold in the Bark domain
    # W_inv : inverse mapping matrix W_inv from matrix W for mapping back from bark scale
    # nfft: : number of subbands in fft
    # returns: mT, masking threshold in the linear scale
    nfreqs = int(nfft/2)
    mT = torch.matmul(mTbark, W_inv[:, :nfreqs].T.float())
    return mT


if __name__ == '__main__':
    # testing:
    import matplotlib.pyplot as plt

    fs = 32000  # sampling frequency of audio signal
    maxfreq = fs/2
    alpha = 0.8  # Exponent for non-linear superposition of spreading functions
    nfilts = 64  # number of subbands in the bark domain
    nfft = 2048  # number of fft subbands

    W = mapping2barkmat_torch(fs, nfilts, nfft)
    plt.imshow(W[:, :256], cmap='Blues')
    plt.title('Matrix W for Uniform to Bark Mapping as Image')
    plt.xlabel('Uniform Subbands')
    plt.ylabel('Bark Subbands')
    plt.show()

    W_inv = mappingfrombarkmat_torch(W, nfft)
    plt.imshow(W_inv[:256, :], cmap='Blues')
    plt.title('Matrix W_inv for Bark to Uniform Mapping as Image')
    plt.xlabel('Bark Subbands')
    plt.ylabel('Uniform Subbands')
    plt.show()

    spreadingfunctionBarkdB = f_SP_dB_torch(maxfreq, nfilts)
    # x-axis: maxbark Bark in nfilts steps:
    maxbark = hz2bark_torch(maxfreq)
    print("maxfreq=", maxfreq, "maxbark=", maxbark)
    bark = torch.linspace(0, maxbark, nfilts)
    # The prototype over "nfilt" bands or 22 Bark, its center
    # shifted down to 22-26/nfilts*22=13 Bark:
    plt.plot(bark, spreadingfunctionBarkdB[26:(26+nfilts)])
    plt.axis([6, 23, -100, 0])
    plt.xlabel('Bark')
    plt.ylabel('dB')
    plt.title('Spreading Function')
    plt.show()

    spreadingfuncmatrix = spreadingfunctionmat_torch(
        spreadingfunctionBarkdB, alpha, nfilts)
    plt.imshow(spreadingfuncmatrix)
    plt.title('Matrix spreadingfuncmatrix as Image')
    plt.xlabel('Bark Domain Subbands')
    plt.ylabel('Bark Domain Subbands')
    plt.show()

    # -Testing-----------------------------------------
    # A test magnitude spectrum:
    # White noise:
    x = torch.randn(32000)*1000

    mX = torch.abs(torch.fft.fft(x[0:2048], norm='ortho'))[0:1025]
    mXbark = mapping2bark_torch(mX, W, nfft)
    # Compute the masking threshold in the Bark domain:
    mTbark = maskingThresholdBark_torch(
        mXbark, spreadingfuncmatrix, alpha, fs, nfilts)
    # Massking threshold in the original frequency domain
    mT = mappingfrombark_torch(mTbark, W_inv, nfft)
    plt.plot(20*torch.log10(mX+1e-3))
    plt.plot(20*torch.log10(mT+1e-3))
    plt.title('Masking Theshold for White Noise')
    plt.legend(('Magnitude Spectrum White Noise', 'Masking Threshold'))
    plt.xlabel('FFT subband')
    plt.ylabel("Magnitude ('dB')")
    plt.show()
    # ----------------------------------------------
    # A test magnitude spectrum, an idealized tone in one subband:
    # tone at FFT band 200:
    x = torch.sin(2*torch.pi/nfft*200*torch.arange(32000))*1000

    mX = torch.abs(torch.fft.fft(x[0:2048], norm='ortho'))[0:1025]
    # Compute the masking threshold in the Bark domain:
    mXbark = mapping2bark_torch(mX, W, nfft)
    mTbark = maskingThresholdBark_torch(
        mXbark, spreadingfuncmatrix, alpha, fs, nfilts)
    mT = mappingfrombark_torch(mTbark, W_inv, nfft)
    plt.plot(20*torch.log10(mT+1e-3))
    plt.title('Masking Theshold for a Tone')
    plt.plot(20*torch.log10(mX+1e-3))
    plt.legend(('Masking Trheshold', 'Magnitude Spectrum Tone'))
    plt.xlabel('FFT subband')
    plt.ylabel("dB")
    plt.show()

    # stft, norm='ortho':
    # import scipy.signal
    # f,t,y=scipy.signal.stft(x,fs=32000,nperseg=2048)
    # make it orthonormal for Parsevals Theorem:
    # Hann window power per sample: 0.375
    # y=y*sqrt(2048/2)/2/0.375
    # plot(y[:,1])
    # plot(mX)

    """
  y=zeros((1025,3))
  y[0,0]=1
  t,x=scipy.signal.istft(y,window='boxcar')
  plot(x)
  #yields rectangle with amplitude 1/2, for orthonormality it would be sqrt(2/N) with overlap, 
  #hence we need a factor sqrt(2/N)*2 for the synthesis, and sqrt(N/2)/2 for the analysis
  #for othogonality.
  #Hence it needs factor sqrt(N/2)/2/windowpowerpersample, hence for Hann Window:
  #y=y*sqrt(2048/2)/2/0.375
  """
