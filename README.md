Audio-Synthesis-GANs

# What is WaveGAN?

WaveGAN are a variant of DCGANs that generates multi-second audio waveforms, they are a specific niche that hasn't been touched by other GANs in the space which primarily focus on visual data. It's official paper is Adverserial Audio Synthesis where the authors talk about the applications of GANs in arts more specifically music and speech through this model, explaining the loss function used by WaveGANs WGAN-GP(Wasserstein GAN + Gradient Penalty) although I implemented WGAN which ensures greater training stability, more informative gradients for backpropogating the models during training due to the difference between Jensen-Shannon Divergence and Wasserstein Distance.

# Key architectural differences between DCGAN and WaveGAN:
- 2D Convolutions/Transposed Convolutions are flattened to 1D in the Discriminator and Generator respectively
- Flattening the strides from 2x2 to 4
- Removing batch normalization from both models
- Using WGAN-GP as the loss function

# WaveGAN Architecture:

Generator Model:

Layer 1: Input Layer — Takes in a latent vector in uniform distribution (-1,1)

Layer 2: Dense Layer 

Layer 3: Reshape Layer

Layer 4: ReLU Activation Function

Layer 5: Transposed Convolutional 1D layer(Stride=4, Kernel Size=25 pixels)

Layer 4: ReLU Activation Function

Layer 5: Transposed Convolutional 1D layer(Stride=4, Kernel Size=25 pixels)

Layer 4: ReLU Activation Function

Layer 5: Transposed Convolutional 1D layer(Stride=4, Kernel Size=25 pixels)

Layer 4: ReLU Activation Function

Layer 5: Transposed Convolutional 1D layer(Stride=4, Kernel Size=25 pixels)

Layer 4: ReLU Activation Function

Layer 5: Transposed Convolutional 1D layer(Stride=4, Kernel Size=25 pixels)

Layer 4: Tanh Activation Function(Output Layer)

Discriminator Model:

Layer 1: Input Layer — Takes as input generated data from Generator or real data

Layer 2: Convolutional 1D Layer(Stride=4, Kernel Size=25 pixel)

Layer 3: LeakyReLU Activation Function

Layer 4: Phase Shuffle(n=2)

Layer 5: Convolutional 1D Layer(Stride=4, Kernel Size=25 pixel)

Layer 6: LeakyReLU Activation Function

Layer 7: Phase Shuffle(n=2)

Layer 8: Convolutional 1D Layer(Stride=4, Kernel Size=25 pixel)

Layer 9: LeakyReLU Activation Function

Layer 10: Phase Shuffle(n=2)

Layer 11: Convolutional 1D Layer(Stride=4, Kernel Size=25 pixel)

Layer 12: LeakyReLU Activation Function

Layer 13: Phase Shuffle(n=2)

Layer 14: Convolutional 1D Layer(Stride=4, Kernel Size=25 pixel)

Layer 15: LeakyReLU Activation Function

Layer 16: Reshape Layer

Layer 17: Dense Layer(Output layer)

# WaveGAN’s Optimizer and Loss Functions:
WaveGAN’s optimizer is Adam although when using WGAN loss I used RMSProps. The parameters of RMSProps for the Generator and Discriminator is a learning_rate of 5e-5. 

For my implementation, I used WGAN instead of WGAN-GP recommended by the paper.

Generator’s Loss: Max(D(G(z)))

Discriminator Loss: Max(D(x)) - Min(D(G(z)))

If you want to get a deeper look into the specific configuration of my implementation of WaveGAN, how WGAN works, and the code for this project then check out my article.
