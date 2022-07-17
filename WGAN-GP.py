def WGAN_GP(D_G_z, D_x, G_z, batch_size):
  G_loss = -tf.reduce_mean(D_G_z)
  D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

  # Create Gradient Penalty
  alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
  difference = G_z - D_x
  interpolated = D_x + alpha * difference

  with tf.GradientTape() as gradient_penalty_tape:
    gradient_penalty_tape.watch(interpolated)
    # Getting an output for interpolated data from Discriminator
    penalty_prediction = Discriminator(interpolated)

  # Calculating the gradient of the interpolated data
  gradients = tf.gradient(penalty_prediction, [interpolated])[0]
  gradient_norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduce_indicies=[1, 2]))
  gradient_penalty = tf.reduce_mean((gradient_norm - 1.0), ** 2)

  # A Lambda value added to the gradient penalty to weight it
  LAMBDA = 10

  # Appending Gradient Penalty to Critic/Discriminator
  D_loss += LAMBDA * gradient_penalty

  return D_loss
  return G_loss
