# What is WaveGAN?
![image](https://user-images.githubusercontent.com/47342287/183469786-10f210e3-cdd7-4d8d-ab64-47db73965104.png)

WaveGAN are a variant of DCGANs that generates multi-second audio waveforms, they are a specific niche that hasn't been touched by other GANs in the space which primarily focus on visual data. It's official paper is Adverserial Audio Synthesis where the authors talk about the applications of GANs in arts more specifically music and speech through this model, explaining the loss function used by WaveGANs WGAN-GP(Wasserstein GAN + Gradient Penalty) which ensures greater training stability, more informative gradients for backpropogating the models during training due to the difference between Jensen-Shannon Divergence and Wasserstein Distance.

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
WaveGAN’s optimizer is Adam. The parameters of Adam for the Generator and Discriminator is a learning_rate of 1e-4, beta_1 is 0.5 and beta_2 is 0.9. 

![image](https://user-images.githubusercontent.com/47342287/183469991-458446a0-0d76-4353-be2d-2d706bb10579.png)

Generator’s Loss: Max(D(G(z)))

Discriminator Loss: Max(D(x)) - Min(D(G(z))) + Enforcing the Lipschitz contiunity through Gradient Penalty

If you want to get a deeper look into the specific configuration of my implementation of WaveGAN, how WGAN-GP works, and the code for this project then check out my article https://devparikh21.medium.com/synthesizing-realistic-audio-using-wavegans-49ec42a80340.
