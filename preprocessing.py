import matplotlib.pyplot as plt
import wave
import numpy as np
import os

# Path for the audio data
SC09_test = "/content/sc09/test"
SC09_validate = "/content/sc09/valid"

# Training Parameters:
batch_size = 64
epochs = 30

audio_dataset = []

def import_audio_dataset(dataset_path):
  for audio_file in os.listdir(dataset_path):
    # Opening the audio file
    audio = wave.open(os.path.join(dataset_path, audio_file), "rb")
        
    # Append to the audio dataset
    audio_dataset.append(audio)

import_audio_dataset(SC09_test)        
import_audio_dataset(SC09_validate)  

print(len(audio_dataset))
waveform_dataset = []

for audio in audio_dataset:
  # Checking configurations for the audio file
  # The framerate of the audio is 48 KHz
  audio_framerate = audio.getframerate()
  # Audio is 1-D also know as Mono
  audio_channels = audio.getnchannels()
  # Read the number of frames in the audio file
  audio_frames = audio.readframes(-1)

  # Convert our audio file to an array of integers in a specific type(int16)
  audio_signal = np.frombuffer(audio_frames, dtype="int8")
            
  # Figure out the duration of length of the audio file
  duration = np.linspace(start=0,
                  stop=len(audio_frames)/audio_framerate,
                  num=len(audio_frames))
  # Close the audio file
  audio.close()

  # Title of Waveform
  plt.figure(figsize=(5, 5))
  plt.plot(duration, audio_signal)
  # X and Y axis
  plt.xlabel("Time(Seconds)")
  plt.ylabel("Amplitude")
    
  # Plot the Waveform
  waveform = plt.show()
  # Append the waveform to the waveform_dataset
  waveform_dataset.append(waveform)

print(len(waveform_dataset))
