#Demo of a rice endoder
#Gerald Schuller, Aug. 2018

#for installation: sudo pip install audio.coders
from audio.coders import rice
from bitstream import BitStream
import numpy as np

origs=np.arange(-2,6)
print("Original= ", origs)
#b: exponent of 2
ricecode=rice(b=1,signed=True)
riceencoded= BitStream(origs.astype(np.int32), ricecode)
print("rice encoded=", riceencoded)

for index in origs:
  print("Index: ", index, "Rice code: ", BitStream(index, ricecode))
  
ricedecoded=riceencoded.read(ricecode, 8)
print("rice decoded=", ricedecoded)
