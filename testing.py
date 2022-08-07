'''Testing the Generator Model'''
import soundfile as sf
# Creating new noise vectors to test the Generator
test_size = 5

test_noise_vectors = tf.random.uniform([test_size, latent_dim], -1., 1., dtype=tf.float32)
test_generated_data = generator_model(test_noise_vectors)

i = 0

for generated_audio_sample in test_generated_data:
  i += 1 
  sf.write("test{}.wav".format(i), generated_audio_sample, 16384)
