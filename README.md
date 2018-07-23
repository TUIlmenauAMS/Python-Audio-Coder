# Python Audio Coder 

This is a Python implementation of an audio coder, for teaching purposes. 
The audio coder includes Huffman coding and writing a binary compressed file
as encoded version.
Copy the directory to a local directory with:
git clone https://github.com/TUIlmenauAMS/Python-Audio-Coder

* For the encoder use: 

python3 audio_encoder.py audiofile.wav [quality]

where quality is an optional quality argument.
Default is quality=100 as 100%. For higher values, the masking threshold is lowered accordingly, 
to reduce the quantization error and increase the quality, but also increase bit rate.
For multi-channel audio signals, like stereo, it encodes the channels separately. 
It writes the compressed signal to file audiofile.acod.
The audiofile could be the included test48khz.wav for testing.
The resulting bit rate is around 1.5 bits/sample for quality around 100%.
Observe that there is no inherent limit on the sampling rate and quality setting, for experimenting.

It uses an MDCT filter bank with 1024 subbands (can be set in file audio_encoder.py on line 48 and on line 27 in audio_decoder.py) with a sine window, a psycho-acoustic model with non-linear superposition, and Huffman coding. It computes new Huffman tables for each audio signal and stores them in the compressed binary file as side information.

For experimentation, different numbers of subbands can be tried for different signals, and the resulting audio quality and bit rate can be compared. For instance for percussive signals like castanets, 1024 subbands should lead to audible pre-echo artifacts, and a lower mumber of subbands should lead to a higher audio quality. For more tonal signals high subbands should be better, speech should be in between.

The number of bark subbands used in the psycho-acoustic subbands is 64, but it can also be changed, on line 49 in file audio_encoder.py and line 28 in audio_decoder.py. This can be used to experiment with the Bark resolution of the psycho-acoustic model.


* To decode the encoded audio signal in *.acod
use: 

python3 audio_decoder.py audiofile.acod

It writes the decoded audio signal in .wav format in file audiofilerek.wav

* A demo and test of the psycho-acoustic model is obtained be running

python3 psyacmodel.py

It includes audio output, for which pyaudio is needed.

A demo and test of quantization a signal according the the psycho-acoustic masking threshold is
obtained with

python3 psyac_quantization.py

* The audio library pyaudio can be installed with:

sudo apt install python3-pyaudio 

or
sudo pip3 install PyAudio

For the binary file, the library "pickle" is needed, it can be installed with

sudo pip3 install pickle 

** Filter Bank Optimization

The folder also contains a few programs which show how to optimize different types of filter banks, with regard to their filter characteristics.

The following runs an optimization for an MDCT, in this example for N=4 subbands (and filter length 2N=8),

python3 optimfuncMDCT.py

This runs an optimization for a Low Delay Filter Bank, also for N=4, but filter length 3N=12, and system delay of 7 (including blocking delay of N-1=3, which doesn't show up in file based examples),

python3 optimfuncLDFB.py

The last runs an optimization for a PQMF filter bank, for N=4 subbands and filter length 8N=32, and system delay of 31 (including the blocking delay of N-1=3),

python3 optimfuncQMF.py

Gerald Schuller, gerald.schuller@tu-ilmenau.de, July 2018.

