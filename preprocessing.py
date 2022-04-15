import matplotlib.pyplot as plt
import wave
import numpy as np
import os

# Path for the audio data
audio_mnist_data = "/content/data"

# Training Parameters:
batch_size = 64
epochs = 30

# Empty dataset for audio after its loaded
audio_dataset = []

# Loading in the data
for folder in os.listdir(audio_mnist_data):
    audio_file_path = os.path.join(audio_mnist_data, folder)
    for audio_file in os.listdir(audio_file_path):
      # Opening the audio file
      audio = wave.open(os.path.join(audio_file_path, audio_file), "rb")
      
      # Append to the audio dataset
      audio_dataset.append(audio)

# Display the length of the audio dataset
print(len(audio_dataset))

batch_10 = int(0.1*len(audio_dataset))

# Get one batch of the total dataset that can be loaded and plotted as a waveform
batch = audio_dataset[batch_10:]

# Empty dataset for the waveforms that are plotted
waveform_dataset = []

for audio in batch:
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
  plt.figure(figsize=(15, 5))
  plt.plot(duration, audio_signal)
  
  # X and Y axis
  plt.xlabel("Time(Seconds)")
  plt.ylabel("Amplitude")
  
  # Plotting the waveform
  waveform = plt.show()

  # Appending the plot to the waveform_dataset
  waveform_dataset.append(waveform)

print(len(waveform_dataset))
