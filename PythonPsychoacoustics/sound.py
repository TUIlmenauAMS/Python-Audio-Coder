#Module for sound playback functions for pylab
#Gerald Schuller, October 9, 2013

import pyaudio
import struct
from numpy import clip
import numpy as np

opened=0;
stream=[]

def sound(audio,  samplingRate):
  #funtion to play back an audio signal, in array "audio"
    import pyaudio
    if len(audio.shape)==2:
       channels=audio.shape[1]
       print("Stereo")
    else:
       channels=1
       print("Mono")
    p = pyaudio.PyAudio()
    # open audio stream

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=samplingRate,
                    output=True)
                    
    #Clipping to avoid overloading the sound device:
    audio=np.clip(audio,-2**15,2**15-1)
    sound = (audio.astype(np.int16).tostring())
    stream.write(sound)

    # close stream and terminate audio object
    stream.stop_stream()
    stream.close()
    p.terminate()
    return  



def wavread(sndfile):
   "This function implements the wavread function of Octave or Matlab to read a wav sound file into a vector s and sampling rate info at its return, with: import sound; [s,rate]=sound.wavread('sound.wav'); or s,rate=sound.wavread('sound.wav');"

   import wave
   wf=wave.open(sndfile,'rb');
   nchan=wf.getnchannels();
   bytes=wf.getsampwidth();
   rate=wf.getframerate();
   length=wf.getnframes();
   print("Number of channels: ", nchan);
   print("Number of bytes per sample:", bytes);
   print("Sampling rate: ", rate);
   print("Number of samples:", length);
   length=length*nchan
   data=wf.readframes(length);
   if bytes==2: #2 bytes per sample:
      shorts = (struct.unpack( 'h' * length, data ));
   else:  #1 byte per sample:
      shorts = (struct.unpack( 'B' * length, data ));
   wf.close;
   shorts=np.array(shorts)
   if nchan> 1:
      shorts=np.reshape(shorts,(-1,nchan))
   return shorts, rate;


def wavwrite(snd,Fs,sndfile):
   """This function implements the wawwritefunction of Octave or Matlab to write a wav sound file from a vector snd with sampling rate Fs, with: 
import sound; 
sound.wavwrite(snd,Fs,'sound.wav');"""

   import wave
   import pylab
 
   WIDTH = 2 #2 bytes per sample
   #FORMAT = pyaudio.paInt16
   CHANNELS = 1
   #RATE = 22050
   RATE = Fs #32000

   length=pylab.size(snd);
   
   wf = wave.open(sndfile, 'wb')
   wf.setnchannels(CHANNELS)
   wf.setsampwidth(WIDTH)
   wf.setframerate(RATE)
   data=struct.pack( 'h' * length, *snd )
   wf.writeframes(data)
   wf.close()



def record(time, Fs):
   "Records sound from a microphone to a vector s, for instance for 5 seconds and with sampling rate of 32000 samples/sec: import sound; s=sound.record(5,32000);"
   
   import numpy;
   global opened;
   global stream;
   CHUNK = 1000 #Blocksize
   WIDTH = 2 #2 bytes per sample
   CHANNELS = 1 #2
   RATE = Fs  #Sampling Rate in Hz
   RECORD_SECONDS = time;

   p = pyaudio.PyAudio()

   a = p.get_device_count()
   print("device count=",a)
   
   #if (opened==0):
   if(1):
     for i in range(0, a):
        print("i = ",i)
        b = p.get_device_info_by_index(i)['maxInputChannels']
        print("max Input Channels=", b)
        b = p.get_device_info_by_index(i)['defaultSampleRate']
        print("default Sample Rate=", b)

     stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=False,
                #input_device_index=3,
                frames_per_buffer=CHUNK)
     opened=1;           

   print("* recording")
   snd=[];
#Loop for the blocks:
   for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
      #Reading from audio input stream into data with block length "CHUNK":
      data = stream.read(CHUNK)
      #Convert from stream of bytes to a list of short integers (2 bytes here) in "samples":
      #shorts = (struct.unpack( "128h", data ))
      shorts = (struct.unpack( 'h' * CHUNK, data ));
      #samples=list(shorts);
      samples=shorts;
      #samples = stream.read(CHUNK).astype(np.int16)
      snd=numpy.append(snd,samples);
   return snd;

