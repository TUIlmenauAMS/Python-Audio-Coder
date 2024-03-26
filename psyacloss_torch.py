# Psycho-acoustic threshold function
# Gerald Schuller, September 2023
# torch version from Renato Profeta and Gerald Schuller, Feb 2024

import sys
currentpath=sys.path[0]
sys.path.append(currentpath+'/PythonPsychoacoustics')
from psyacmodel_torch import *
import torch


def psyacthresh_torch(ys, fs):
    # input: ys: 2d array of sound STFT (from a mono signal, shape N+1,M)
    # fs: sampling frequency in samples per second
    # returns: mT, the masking threshold in N+1 subbands for the M blocks (shape N+1,M)

    maxfreq = fs/2
    alpha = 0.8  # Exponent for non-linear superposition of spreading functions
    nfilts = 64  # number of subbands in the bark domain
    # M=len(snd)//nfft
    M = ys.shape[1]
    # N=nfft//2
    N = ys.shape[0]-1
    nfft = 2*N

    W = mapping2barkmat_torch(fs, nfilts, nfft)
    W_inv = mappingfrombarkmat_torch(W, nfft)
    spreadingfunctionBarkdB = f_SP_dB_torch(maxfreq, nfilts)
    # maxbark=hz2bark(maxfreq)
    # bark=np.linspace(0,maxbark,nfilts)
    spreadingfuncmatrix = spreadingfunctionmat_torch(
        spreadingfunctionBarkdB, alpha, nfilts)
    # Computing the masking threshold in each block of nfft samples:
    mT = torch.zeros((N+1, M))
    for m in range(M):  # M: number of blocks
        # mX=np.abs(np.fft.fft(snd[m*nfft+np.arange(2048)],norm='ortho'))[0:1025]
        mX = torch.abs(ys[:, m])
        mXbark = mapping2bark_torch(mX, W, nfft)
        # Compute the masking threshold in the Bark domain:
        mTbark = maskingThresholdBark_torch(
            mXbark, spreadingfuncmatrix, alpha, fs, nfilts)
        # Massking threshold in the original frequency domain
        mT[:, m] = mappingfrombark_torch(mTbark, W_inv, nfft)

    return mT  # the masking threshold in N+1 subbands for the M blocks


def percloss(orig, modified, fs):
    # computes the perceptually weighted distance between the original (orig) and modified audio signals,
    # with sampling rate fs. The psycho-acoustic threshold is computed from orig, hence it is not commutative.
    # returns: ploss, the perceptual loss value, the mean squarred difference of the two spectra, normalized to the masking threshold of the orig.
    # Gerald Schuller, September 2023

    nfft = 2048  # number of fft subbands
    N = nfft//2

    # print("orig.shape=", orig.shape)
    
    # origsys.shape= freq.bin, channel, block
    if len(orig.shape) == 2:  # multichannel
        chan = orig.shape[1]
        for c in range(chan):
            origys = torch.stft(orig[:,c], n_fft=2*N, hop_length=2 *
                        N//2, return_complex=True, normalized=True, window=torch.hann_window(2*N))
            if c == 0:  # initialize masking threshold tensor mT
                mT0 = psyacthresh_torch(origys[:, :], fs)
                rows, cols = mT0.shape
                mT = torch.zeros((rows, chan, cols))
                mT[:, 0, :] = mT0
            else:
                mT[:, c, :] = psyacthresh_torch(origys[:, :], fs)
    else:
        chan = 1
        origys = torch.stft(orig, n_fft=2*N, hop_length=2 *
                        N//2, return_complex=True, normalized=True, window=torch.hann_window(2*N))
        mT = psyacthresh_torch(origys, fs)
    """
    plt.plot(20*np.log10(np.abs(origys[:,0,400])+1e-6))
    plt.plot(20*np.log10(mT[:,0,400]+1e-6))
    plt.legend(('Original spectrum','Masking threshold'))
    plt.title("Spectrum over bins")
    """
    # print("origys.shape=",origys.shape, "mT.shape=",mT.shape)

    modifiedys = torch.stft(
        modified, n_fft=2*N, hop_length=2*N//2, return_complex=True, normalized=True, window=torch.hann_window(2*N))

    # normalized diff. spectrum:
    normdiffspec = torch.abs((origys-modifiedys)/mT)
    # Plot difference spectrum, normalized to masking threshold:
    """
    plt.plot(20*np.log10(normdiffspec[:,0,400])+1e-6)
    plt.title("normalized diff. spectrum")
    plt.show()
    """
    ploss = torch.mean(normdiffspec**2)
    return ploss


if __name__ == '__main__':  # testing
    import scipy.io.wavfile as wav
    import scipy.signal
    import numpy as np
    import matplotlib.pyplot as plt
    import sound
    import os

    fs, snd = wav.read(r'./fantasy-orchestra.wav')
    plt.plot(snd[:, 0])
    plt.title("The original sound")
    plt.show()
    
    print("\nThe original signal:")
    sound.sound(snd,fs)

    nfft = 2048  # number of fft subbands
    N = nfft//2

    print("snd.shape=", snd.shape)
    f, t, ys = scipy.signal.stft(snd[:, 0], fs=2*np.pi, nperseg=2*N)
    # scaling for the application of the
    # resulting masking threshold to MDCT subbands:
    ys *= np.sqrt(2*N/2)/2/0.375

    print("fs=", fs)
    ys = torch.from_numpy(ys)
    mT = psyacthresh_torch(ys, fs)

    print("mT.shape=", mT.shape)
    plt.plot(20*np.log10(np.abs(ys[:, 400])+1e-6))
    plt.plot(20*np.log10(mT[:, 400]+1e-6))
    plt.legend(('Original spectrum', 'Masking threshold'))
    plt.title("Spectrum over bins")

    plt.figure()
    plt.imshow(20*np.log10(np.abs(ys)+1e-6))
    plt.title("Spectrogram of Original")
    plt.show()

    # Audio signal with uniform quantization and de-quantization
    snd = torch.from_numpy(snd[:, 0]).float()
    snd_quant = (torch.round(snd/10000))*10000
    
    print("\nThe quantized signal:")
    sound.sound(np.array(snd_quant),fs)

    ploss = percloss(snd, snd_quant, fs)
    
    #version AAC encoded and decoded:
    os.system("ffmpeg -y -i fantasy-orchestra.wav -b:a 64k fantasy-orchestra64k.aac")
    os.system("ffmpeg -y -i fantasy-orchestra64k.aac fantasy-orchestradec_aac.wav")
    fs, snd_aac = wav.read(r'./fantasy-orchestradec_aac.wav')
    
    print("\nThe AAC encoded/Decoded Signal:")
    sound.sound(np.array(snd_aac),fs)
    
    print("\n\npsyco-acoustic loss to quantized signal=", ploss)
    
    snd_aac = torch.from_numpy(snd_aac[:, 0]).float()
    
    minlength=min(snd.shape[0],snd_aac.shape[0])
    #print("\n\nminlength=", minlength)
    delay=120 #aac delay in samples
    
    ploss_aac = percloss(snd[:minlength], snd_aac[delay:minlength+delay], fs)
    print("\n\npsyco-acoustic loss to aac enc/dec signal=", ploss_aac)
    
