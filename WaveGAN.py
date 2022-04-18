import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Reshape, Conv1DTranspose, Input, Activation, LeakyReLU
from tensorflow.keras.optimizers import Adam
import numpy as np

model_dim = 64
kernel_len = 25
stride = 4
channels = 1

latent_dim = 100

def Generator(input_vector):
  # Initialize the model
  model = Sequential()
  # Input for the Generator
  model.add(Input(input_vector))
  
  # First Layer
  model.add(Dense(256 * model_dim))
  # Reshape Layer
  model.add(Reshape(16, 16 * model_dim))
  model.add(Activation("relu"))

  # Transposed Conv1D + ReLU
  model.add(Conv1DTranspose(16 * model_dim, kernel_len, 4))
  model.add(Activation("relu"))

  # Transposed Conv1D + ReLU
  model.add(Conv1DTranspose(8 * model_dim, kernel_len, 4))
  model.add(Activation("relu"))

  # Transposed Conv1D + ReLU
  model.add(Conv1DTranspose(4 * model_dim, kernel_len, 4))
  model.add(Activation("relu"))

  # Transposed Conv1D + ReLU
  model.add(Conv1DTranspose(2 * model_dim, kernel_len, 4))
  model.add(Activation("relu"))

  # Transposed Conv1D + Tanh
  model.add(Conv1DTranspose(model_dim, kernel_len, 4, channels))
  model.add(Activation("tanh"))

def Discriminator(input_data):
  model = Sequential()
  # Input layer
  model.add(Input=(input_data))
  
  # Conv1D + LeakyReLU
  model.add(Conv1D(model_dim, kernel_len, 4, channels))
  model.add(LeakyReLU(alpha=0.2))
  # Implement Phase Shuffle

  # Conv1D + LeakyReLU
  model.add(Conv1D(2 * model_dim, kernel_len, 4))
  model.add(LeakyReLU(alpha=0.2))
  # Implement Phase Shuffle

  # Conv1D + LeakyReLU
  model.add(Conv1D(4 * model_dim, kernel_len, 4))
  model.add(LeakyReLU(alpha=0.2))
  # Implement Phase Shuffle

  # Conv1D + LeakyReLU
  model.add(Conv1D(8 * model_dim, kernel_len, 4))
  model.add(LeakyReLU(alpha=0.2))
  # Implement Phase Shuffle

  # Conv1D + LeakyReLU
  model.add(Conv1D(16 * model_dim, kernel_len, 4))
  model.add(LeakyReLU(alpha=0.2))
  # Implement Phase Shuffle

  # Reshape layer
  model.add(Reshape(256 * model_dim))

  # Output(Dense) layer
  model.add(Dense(256 * model_dim))
