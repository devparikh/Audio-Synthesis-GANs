import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Reshape, Conv1DTranspose, Input, Activation, LeakyReLU, ReLU
from tensorflow.keras.optimizers import Adam
import numpy as np

model_dim = 64
kernel_len = 25
stride = 4
batch_size = 64
channels = 1

'''Generator Model'''

# Initialize the model
generator_model = Sequential()
# Input for the Generator
generator_model.add(Input(shape=(100), dtype="float32"))
  
# First Layer
generator_model.add(Dense(4 * 4 * 16 * model_dim))
# Reshape Layer
generator_model.add(Reshape([16, 16 * model_dim]))
generator_model.add(Activation("relu")) 

# Transposed Conv1D + ReLU
generator_model.add(Conv1DTranspose(8 * model_dim, kernel_size=kernel_len, strides=stride, padding="same"))
generator_model.add(Activation("relu"))

# Transposed Conv1D + ReLU
generator_model.add(Conv1DTranspose(4 * model_dim, kernel_size=kernel_len, strides=stride, padding="same"))
generator_model.add(Activation("relu"))

# Transposed Conv1D + ReLU
generator_model.add(Conv1DTranspose(2 * model_dim, kernel_size=kernel_len, strides=stride, padding="same"))
generator_model.add(Activation("relu"))

# Transposed Conv1D + ReLU
generator_model.add(Conv1DTranspose(model_dim, kernel_size=kernel_len, strides=stride, padding="same"))
generator_model.add(Activation("relu"))

# Transposed Conv1D + Tanh
generator_model.add(Conv1DTranspose(channels, kernel_size=kernel_len, strides=stride, padding="same"))
generator_model.add(Activation("tanh"))


'''Discriminator Model'''
discriminator_model = Sequential()
# Input layer
discriminator_model.add(Input(shape=(16384, channels), dtype="float32"))
  
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
discriminator_model.add(Reshape([-1]))

# Output(Dense) layer
discriminator_model.add(Dense(1))

'''Training WaveGAN'''

# Training Parameters
batch_size = 64
latent_dim = 100
epochs = 72

# Training Parameters
latent_dim = 100
epochs = 72
batch_size = 64

batch_count = 0
start_point = 0
batch_set = []

for batches in range(0, 72):
  batch = audio_dataset[start_point:(start_point+64)]
  batch_set.append(batch)
  start_point += 64

# Defining Discriminator Optimizer
Discriminator_Optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4, 
        beta_1=0.5,
        beta_2=0.9)
  
  
# Defining Generator Optimizer
Generator_Optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4, 
        beta_1=0.5,
        beta_2=0.9)

for epoch in range(1, epochs+1):
  print("\nEpoch # {}".format(epoch))
  print("-------------------------------------")
  i = 5
  batch_count += 1 
  # Training Discriminator
  while i > 0:
    i -= 1 
    # Getting random noise vectors in a array of the batch size from a uniform distribution between -1 and 1
    noise_vector = tf.random.uniform([batch_size, latent_dim], -1., 1., dtype=tf.float32)

    Discriminator_Variables = discriminator_model.trainable_variables
    
    generator_model.trainable = False
    discriminator_model.trainable = True

    # Capturing gradient information from the D_G_Output
    with tf.GradientTape() as D_tape:
      G_O = generator_model(noise_vector)
      D_G_Output = discriminator_model(G_O)
      D_tape.watch(D_G_Output)
      
      # Getting predictions from the Discriminator
      batch = batch_set[batch_count]
      D_X = discriminator_model(batch)
      D_tape.watch(D_X)
      
      D_tape.watch(Discriminator_Variables)

    # Calculating WGAN for Discriminator
    D_loss =  tf.reduce_mean(D_X) - tf.reduce_mean(D_G_Output)

    # Enforcing Gradient Penalty
    alpha = tf.random.normal([1, 1, 1], 0.0, 1.0)
    difference = batch - G_O
    interpolated = batch + (alpha * difference)

    # Capturing the gradient from the interpolation of the real dataset probability distribution and the generated data distribution
    with tf.GradientTape() as gradient_penalty_tape:
      gradient_penalty_tape.watch(interpolated)
      # Getting an output for interpolated data from Discriminator
      penalty_prediction = discriminator_model(interpolated)

    # Calculating the gradient of the interpolated data
    gradients = gradient_penalty_tape.gradient(penalty_prediction, [interpolated])[0]
    gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2]))
    gradient_penalty = tf.reduce_mean((gradient_norm - 1.0) ** 2)

    # A Lambda value added to the gradient penalty to weight it
    LAMBDA = 10

    # Appending Gradient Penalty to Critic/Discriminator
    D_loss += LAMBDA * gradient_penalty

    D_grad = D_tape.gradient([D_G_Output, D_X], Discriminator_Variables)
    D_backprop = Discriminator_Optimizer.apply_gradients(zip(D_grad, Discriminator_Variables))

    print("Discriminator Loss Update #{}: {}".format(i, float(D_loss)))

  # Training the Generator
  if i == 0:
    print("-------------------------------------")
    generator_model.trainable = True
    discriminator_model.trainable = False

    Generator_Variables = generator_model.trainable_variables

    with tf.GradientTape() as G_tape:
      # Passing a batch of noise vectors through the Generator
      G_O = generator_model(noise_vector)
      # The generated audio samples are forwarded passed through the Discriminator network
      D_G_O = discriminator_model(G_O)

      G_tape.watch(G_O)

      G_tape.watch(Generator_Variables)

    G_loss = -tf.reduce_mean(D_G_O)

    g_grad = G_tape.gradient(G_O, Generator_Variables)
    G_backprop = Generator_Optimizer.apply_gradients(zip(g_grad, Generator_Variables))

    print("Generator Loss: {}".format(float(G_loss)))
