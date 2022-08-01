import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Reshape, Conv1DTranspose, Input, Activation, LeakyReLU
from tensorflow.keras.optimizers import Adam
import numpy as np

model_dim = 64
kernel_len = 25
stride = 4
channels = 1

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Reshape, Conv1DTranspose, Input, Activation, LeakyReLU, ReLU
from tensorflow.keras.optimizers import Adam
import numpy as np

model_dim = 64
kernel_len = 25
stride = 4
channels = 1

'''Generator Model'''

# Initialize the model
generator_model = Sequential()
# Input for the Generator
generator_model.add(Input(shape=(100,), batch_size=64, dtype="float32"))
  
# First Layer
generator_model.add(Dense(256 * model_dim))
# Reshape Layer
generator_model.add(Reshape([16, 16 * model_dim]))
generator_model.add(Activation("relu")) 

# Transposed Conv1D + ReLU
generator_model.add(Conv1DTranspose(16 * model_dim, kernel_size=kernel_len, strides=stride, padding="valid"))
generator_model.add(Activation("relu"))

# Transposed Conv1D + ReLU
generator_model.add(Conv1DTranspose(8 * model_dim, kernel_size=kernel_len, strides=stride, padding="valid"))
generator_model.add(Activation("relu"))

# Transposed Conv1D + ReLU
generator_model.add(Conv1DTranspose(4 * model_dim, kernel_size=kernel_len, strides=stride, padding="valid"))
generator_model.add(Activation("relu"))

# Transposed Conv1D + ReLU
generator_model.add(Conv1DTranspose(2 * model_dim, kernel_size=kernel_len, strides=stride, padding="valid"))
generator_model.add(Activation("relu"))

# Transposed Conv1D + Tanh
generator_model.add(Conv1DTranspose(channels, kernel_size=kernel_len, strides=stride, padding="valid"))
generator_model.add(Activation("tanh"))

'''Discriminator Model'''
discriminator_model = Sequential()
# Input layer
discriminator_model.add(Input(shape=(16384, channels), batch_size=64))
  
# Conv1D + LeakyReLU
discriminator_model.add(Conv1D(model_dim, kernel_size=kernel_len, strides=stride, padding="same"))
discriminator_model.add(LeakyReLU(alpha=0.2))

# Conv1D + LeakyReLU
discriminator_model.add(Conv1D(2 * model_dim, kernel_size=kernel_len, strides=stride, padding="same"))
discriminator_model.add(LeakyReLU(alpha=0.2))

# Conv1D + LeakyReLU
discriminator_model.add(Conv1D(4 * model_dim, kernel_size=kernel_len, strides=stride, padding="same"))
discriminator_model.add(LeakyReLU(alpha=0.2))

# Conv1D + LeakyReLU
discriminator_model.add(Conv1D(8 * model_dim, kernel_size=kernel_len, strides=stride, padding="same"))
discriminator_model.add(LeakyReLU(alpha=0.2))

# Conv1D + LeakyReLU
discriminator_model.add(Conv1D(16 * model_dim, kernel_size=kernel_len, strides=stride, padding="same"))
discriminator_model.add(LeakyReLU(alpha=0.2))

# Reshape layer
discriminator_model.add(Reshape([batch_size, -1]))

# Output(Dense) layer
discriminator_model.add(Dense(1))

'''Training WaveGAN'''

# Training Parameters
batch_size = 64
latent_dim = 100
epochs = 50

D_updates = 5

i = 0
batch_set = []
batch_length = 64

# Training Process
for iteration in range(1, epochs+1):
  # Batching the audio dataset
  # Training Discriminator
  for discriminator_updates in range(D_updates):
    # Getting random noise vectors in a array of the batch size from a uniform distribution between -1 and 1
    noise_vector = tf.random.uniform([batch_size, latent_dim], -1., 1., dtype=tf.float32)

    with tf.name_scope("D_G_Output"):
      # Getting output from the Discriminator from the generated data from Generator
      G_O = Generator(noise_vector)
      D_G_Output = Discriminator(G_O)

    with tf.name_scope("D_X"):
      # Getting output from the Discriminator for real data
      D_X = Discriminator(audio_dataset)
    
    # Getting variables that I want to control when training
    D_fake_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D_G_Output")
    D_real_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="D_X")

  # Training Generator
  with tf.name_scope("G_O"):
    # Getting output from Generator on noise vector
    G_O = Generator(noise_vector)
    
  # Getting key training variables for Generator model
  G_vars = tf.get_collection([tf.GraphKeys.TRAINABLE_VARIABLES], "G_O")

  # Passing the data through WGAN-GP to get values for generator and discriminator loss
  WGAN_GP(D_G_Output, D_X, G_O, batch_size)

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
    
  G_Loss_Min = Generator_Optimizer.minimize(G_loss, var_list=[D_fake_vars, D_real_vars])
  D_Loss_Min = Generator_Optimizer.minimize(D_loss, var_list=[G_vars])
