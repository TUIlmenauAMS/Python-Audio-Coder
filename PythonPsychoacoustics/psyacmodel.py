#Programs to implement a psycho-acoustic model
#Using a matrix for the spreading function (faster)
#Gerald Schuller, Nov. 2016

import numpy as np



def f_SP_dB(maxfreq,nfilts):
   #usage: spreadingfunctionmatdB=f_SP_dB(maxfreq,nfilts)
   #computes the spreading function protoype, in the Bark scale.
   #Arguments: maxfreq: half the sampling freqency
   #nfilts: Number of subbands in the Bark domain, for instance 64   
   maxbark=hz2bark(maxfreq) #upper end of our Bark scale:22 Bark at 16 kHz
   #Number of our Bark scale bands over this range: nfilts=64
   spreadingfunctionBarkdB=np.zeros(2*nfilts)
   #Spreading function prototype, "nfilts" bands for lower slope 
   spreadingfunctionBarkdB[0:nfilts]=np.linspace(-maxbark*27,-8,nfilts)-23.5
   #"nfilts" bands for upper slope:
   spreadingfunctionBarkdB[nfilts:2*nfilts]=np.linspace(0,-maxbark*12.0,nfilts)-23.5
   return spreadingfunctionBarkdB


def spreadingfunctionmat(spreadingfunctionBarkdB,alpha,nfilts):
   #Turns the spreading prototype function into a matrix of shifted versions.
   #Convert from dB to "voltage" and include alpha exponent
   #nfilts: Number of subbands in the Bark domain, for instance 64  
   spreadingfunctionBarkVoltage=10.0**(spreadingfunctionBarkdB/20.0*alpha)
   #Spreading functions for all bark scale bands in a matrix:
   spreadingfuncmatrix=np.zeros((nfilts,nfilts))
   for k in range(nfilts):
      spreadingfuncmatrix[k,:]=spreadingfunctionBarkVoltage[(nfilts-k):(2*nfilts-k)]
   return spreadingfuncmatrix



def maskingThresholdBark(mXbark,spreadingfuncmatrix,alpha,fs,nfilts): 
  #Computes the masking threshold on the Bark scale with non-linear superposition
  #usage: mTbark=maskingThresholdBark(mXbark,spreadingfuncmatrix,alpha)
  #Arg: mXbark: magnitude of FFT spectrum, on the Bark scale
  #spreadingfuncmatrix: spreading function matrix from function spreadingfunctionmat
  #alpha: exponent for non-linear superposition (eg. 0.6), 
  #fs: sampling freq., nfilts: number of Bark subbands
  #nfilts: Number of subbands in the Bark domain, for instance 64  
  #Returns: mTbark: the resulting Masking Threshold on the Bark scale 
  
  #Compute the non-linear superposition:
  mTbark=np.dot(mXbark**alpha, spreadingfuncmatrix**alpha)
  #apply the inverse exponent to the result:
  mTbark=mTbark**(1.0/alpha)
  #Threshold in quiet:
  maxfreq=fs/2.0
  maxbark=hz2bark(maxfreq)
  step_bark = maxbark/(nfilts-1)
  barks=np.arange(0,nfilts)*step_bark
  #convert the bark subband frequencies to Hz:
  f=bark2hz(barks)+1e-6
  #Threshold of quiet in the Bark subbands in dB:
  LTQ=np.clip((3.64*(f/1000.)**-0.8 -6.5*np.exp(-0.6*(f/1000.-3.3)**2.)+1e-3*((f/1000.)**4.)),-20,160)
  #Maximum of spreading functions and hearing threshold in quiet:
  mTbark=np.max((mTbark, 10.0**((LTQ-60)/20)),0)
  return mTbark





def hz2bark(f):
        """ Usage: Bark=hz2bark(f)
            f    : (ndarray)    Array containing frequencies in Hz.
        Returns  :
            Brk  : (ndarray)    Array containing Bark scaled values.
        """
        Brk = 6. * np.arcsinh(f/600.)                                                 
        return Brk

def bark2hz(Brk):
        """ Usage:
        Hz=bark2hs(Brk)
        Args     :
            Brk  : (ndarray)    Array containing Bark scaled values.
        Returns  :
            Fhz  : (ndarray)    Array containing frequencies in Hz.
        """
        Fhz = 600. * np.sinh(Brk/6.)
        return Fhz

def mapping2barkmat(fs, nfilts,nfft):
  #Constructing mapping matrix W which has 1's for each Bark subband, and 0's else
  #usage: W=mapping2barkmat(fs, nfilts,nfft)  
  #arguments: fs: sampling frequency
  #nfilts: number of subbands in Bark domain
  #nfft: number of subbands in fft
  maxbark=hz2bark(fs/2) #upper end of our Bark scale:22 Bark at 16 kHz
  nfreqs=nfft/2; step_bark = maxbark/(nfilts-1)
  binbark = hz2bark(np.linspace(0,(nfft/2),(nfft/2)+1)*fs/nfft)
  W = np.zeros((nfilts, nfft))
  for i in range(nfilts):
     W[i,0:int(nfft/2)+1] = (np.round(binbark/step_bark)== i)
  return W

def mapping2bark(mX,W,nfft):
  #Maps (warps) magnitude spectrum vector mX from DFT to the Bark scale
  #arguments: mX: magnitude spectrum from fft
  #W: mapping matrix from function mapping2barkmat
  #nfft: : number of subbands in fft
  #returns: mXbark, magnitude mapped to the Bark scale
  nfreqs=int(nfft/2)
  #Here is the actual mapping, suming up powers and conv. back to Voltages:
  mXbark = (np.dot( np.abs(mX[:nfreqs])**2.0, W[:, :nfreqs].T))**(0.5)
  return mXbark

def mappingfrombarkmat(W,nfft):
  #Constructing inverse mapping matrix W_inv from matrix W for mapping back from bark scale
  #usuage: W_inv=mappingfrombarkmat(Wnfft)
  #argument: W: mapping matrix from function mapping2barkmat
  #nfft: : number of subbands in fft
  nfreqs=int(nfft/2)
  W_inv= np.dot(np.diag((1.0/(np.sum(W,1)+1e-6))**0.5), W[:,0:nfreqs + 1]).T
  return W_inv

#-------------------
def mappingfrombark(mTbark,W_inv,nfft):
  #usage: mT=mappingfrombark(mTbark,W_inv,nfft)
  #Maps (warps) magnitude spectrum vector mTbark in the Bark scale
  # back to the linear scale
  #arguments:
  #mTbark: masking threshold in the Bark domain
  #W_inv : inverse mapping matrix W_inv from matrix W for mapping back from bark scale
  #nfft: : number of subbands in fft
  #returns: mT, masking threshold in the linear scale
  nfreqs=int(nfft/2)
  mT = np.dot(mTbark, W_inv[:, :nfreqs].T)
  return mT


if __name__ == '__main__':
  #testing:
  import matplotlib.pyplot as plt
  import sound 

  fs=32000  # sampling frequency of audio signal
  maxfreq=fs/2
  alpha=0.8  #Exponent for non-linear superposition of spreading functions
  nfilts=64  #number of subbands in the bark domain
  nfft=2048  #number of fft subbands

  W=mapping2barkmat(fs,nfilts,nfft)
  plt.imshow(W[:,:256],cmap='Blues')
  plt.title('Matrix W for Uniform to Bark Mapping as Image')
  plt.xlabel('Uniform Subbands')
  plt.ylabel('Bark Subbands')
  plt.show()
  
  W_inv=mappingfrombarkmat(W,nfft)
  plt.imshow(W_inv[:256,:],cmap='Blues')
  plt.title('Matrix W_inv for Bark to Uniform Mapping as Image')
  plt.xlabel('Bark Subbands')
  plt.ylabel('Uniform Subbands')
  plt.show()

  spreadingfunctionBarkdB=f_SP_dB(maxfreq,nfilts)
  #x-axis: maxbark Bark in nfilts steps:
  maxbark=hz2bark(maxfreq)
  print("maxfreq=", maxfreq, "maxbark=", maxbark)
  bark=np.linspace(0,maxbark,nfilts)
  #The prototype over "nfilt" bands or 22 Bark, its center 
  #shifted down to 22-26/nfilts*22=13 Bark:
  plt.plot(bark,spreadingfunctionBarkdB[26:(26+nfilts)])
  plt.axis([6,23,-100,0])
  plt.xlabel('Bark')
  plt.ylabel('dB')
  plt.title('Spreading Function')
  plt.show()
  
  spreadingfuncmatrix=spreadingfunctionmat(spreadingfunctionBarkdB,alpha, nfilts)
  plt.imshow(spreadingfuncmatrix)
  plt.title('Matrix spreadingfuncmatrix as Image')
  plt.xlabel('Bark Domain Subbands')
  plt.ylabel('Bark Domain Subbands')
  plt.show()
  
  #-Testing-----------------------------------------
  #A test magnitude spectrum:
  # White noise:
  x=np.random.randn(32000)*1000
  sound.sound(x,fs)
  
  mX=np.abs(np.fft.fft(x[0:2048],norm='ortho'))[0:1025]
  mXbark=mapping2bark(mX,W,nfft)
  #Compute the masking threshold in the Bark domain:  
  mTbark=maskingThresholdBark(mXbark,spreadingfuncmatrix,alpha,fs,nfilts)
  #Massking threshold in the original frequency domain
  mT=mappingfrombark(mTbark,W_inv,nfft)
  plt.plot(20*np.log10(mX+1e-3))
  plt.plot(20*np.log10(mT+1e-3))
  plt.title('Masking Theshold for White Noise')
  plt.legend(('Magnitude Spectrum White Noise','Masking Threshold'))
  plt.xlabel('FFT subband')
  plt.ylabel("Magnitude ('dB')")
  plt.show()
  #----------------------------------------------
  #A test magnitude spectrum, an idealized tone in one subband:
  #tone at FFT band 200:
  x=np.sin(2*np.pi/nfft*200*np.arange(32000))*1000
  sound.sound(x,fs)
  
  mX=np.abs(np.fft.fft(x[0:2048],norm='ortho'))[0:1025]
  #Compute the masking threshold in the Bark domain:  
  mXbark=mapping2bark(mX,W,nfft)
  mTbark=maskingThresholdBark(mXbark,spreadingfuncmatrix,alpha,fs,nfilts)
  mT=mappingfrombark(mTbark,W_inv,nfft)
  plt.plot(20*np.log10(mT+1e-3))
  plt.title('Masking Theshold for a Tone')
  plt.plot(20*np.log10(mX+1e-3))
  plt.legend(('Masking Trheshold', 'Magnitude Spectrum Tone'))
  plt.xlabel('FFT subband')
  plt.ylabel("dB")
  plt.show()
  
  #stft, norm='ortho':
  #import scipy.signal
  #f,t,y=scipy.signal.stft(x,fs=32000,nperseg=2048)
  #make it orthonormal for Parsevals Theorem:
  #Hann window power per sample: 0.375
  #y=y*sqrt(2048/2)/2/0.375
  #plot(y[:,1])
  #plot(mX)
  
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

