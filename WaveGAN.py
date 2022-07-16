import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Reshape, Conv1DTranspose, Input, Activation, LeakyReLU
from tensorflow.keras.optimizers import Adam
import numpy as np

model_dim = 64
kernel_len = 25
stride = 4
channels = 1

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

'''Training WaveGAN'''

# Training Parameters
batch_size = 64
latent_dim = 100
epochs = 50

D_updates = 5

# Training Process
for iteration in range(1, epochs+1):

  # Training Discriminator
  for discriminator_updates in range(D_updates):

    # Getting random noise vectors in a array of the batch size from a uniform distribution between -1 and 1
    noise_vector = tf.random.uniform([batch_size, latent_dim], minval=-1, maxval=1, dtype=tf.float32)

    with tf.name_scope("D_G_Output"), tf.variable_scope("D"):
      # Getting output from the Discriminator from the generated data from Generator
      D_G_Output = Discriminator(Generator(noise_vector))

    with tf.name_scope("D_X"), tf.variable_scope("D")
      # Getting output from the Discriminator for real data
      D_X = Discriminator(audio_dataset)
    
    # Getting variables that I want to control when training
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')

  with tf.name_scope("G_O"), tf.variable_scope("G"):
    # Getting output from Generator on noise vector
    G_O = Generator(noise_vector)
    
  # Getting key training variables for Generator model
  Generator_Variables = tf.get_collection([tf.GraphKeys.TRAINABLE_VARIABLES], "G")


  # Passing the data through WGAN-GP to get values for generator and discriminator loss
  Discriminator_Loss, Generator_Loss = WGAN_GP(D_G_Output, D_X, G_O, batch_size)

  # Defining Discriminator Optimizer
  Discriminator_Optimizer = tf.keras.optimizers.Adam(
          learning_rate=1e-4,
          beta1=0.5,
          beta2=0.9)
  
  
  # Defining Generator Optimizer
  Generator_Optimizer = tf.keras.optimizers.Adam(
          learning_rate=1e-4,
          beta1=0.5,
          beta2=0.9)
