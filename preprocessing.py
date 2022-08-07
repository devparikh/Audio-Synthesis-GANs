import matplotlib.pyplot as plt
import librosa
import numpy as np
import os

# Path for the audio data
SC09_test = "/content/sc09/test"
SC09_validation = "/content/sc09/valid"

# Training Parameters:
batch_size = 64
epochs = 30

# Datasets where the audio data and waveforms will be stored
audio_dataset = []
waveform_dataset = []

# A function for reading in the data and displaying their waveforms
def reading_audio(dataset):
  i = 0 
  for audio_file in os.listdir(dataset):
    # Opening the audio file
    audio, sampling_rate = librosa.core.load(os.path.join(dataset, audio_file), sr=16384)
    
    # Checking if the sampling_rate is the same as the length of the audio, and if True then we will delete the file
    if len(audio) < sampling_rate:
      del audio
    else:
    # If its not true then we will continue with the reset of the script
      duration = np.linspace(start=0,
                    stop=len(audio)/sampling_rate,
                    num=len(audio))

      # Title of Waveform
      plt.figure(figsize=(5, 5))
      plt.plot(duration, audio)
      # X and Y axis
      plt.xlabel("Time(Seconds)")
      plt.ylabel("Amplitude")

      i += 1
      print(i)
          
      # Plot the Waveform
      waveform = plt.show()
      
      # Append the waveform to the waveform_dataset
      waveform_dataset.append(waveform)

      # Append to the audio dataset
      audio_dataset.append(audio)  

# Running the functions on the SC09 Test and Validation sets 
reading_audio(SC09_test)
reading_audio(SC09_validation)

# Only taking 4608 samples from the audio training set because that can be batched into 72 batches of 64 samples for training
audio_dataset = audio_dataset[:4608]

audio_dataset = np.asarray(audio_dataset)
audio_dataset = np.resize(audio_dataset, (len(audio_dataset), 16384, 1))
print(len(audio_dataset))
