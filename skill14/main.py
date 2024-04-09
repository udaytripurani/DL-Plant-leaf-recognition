import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Lambda, Reshape, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

# Path to your dataset
dataset_path = "C:\\Users\\venus\\OneDrive\\Desktop\\deep learning\\sample dataset"

# Using ImageDataGenerator to load and preprocess the images
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) # Assuming 20% validation split

# Training dataset
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),  # Assuming images are resized to 150x150
    batch_size=32,
    class_mode='input',  # Use 'input' as class_mode for autoencoder
    subset='training')  # Use subset 'training' for training data

# Validation dataset
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='input',  # Use 'input' as class_mode for autoencoder
    subset='validation')  # Use subset 'validation' for validation data

# Define VAE architecture
latent_dim = 32  # Dimensionality of the latent space

# Encoder
inputs = Input(shape=(150, 150, 3))
x = Conv2D(32, 3, strides=2, activation='relu', padding='same')(inputs)
x = Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)
x = Flatten()(x)
z_mean = Dense(latent_dim)(x)
z_log_var = Dense(latent_dim)(x)

# Reparameterization trick to sample from the learned distribution
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_inputs = Input(shape=(latent_dim,))
x = Dense(18 * 18 * 64, activation='relu')(decoder_inputs)
x = Reshape((18, 18, 64))(x)
x = Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
x = Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
outputs = Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)

# Instantiate VAE model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
decoder = Model(decoder_inputs, outputs, name='decoder')
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae')

# Define VAE loss
reconstruction_loss = tf.keras.losses.mse(K.flatten(inputs), K.flatten(outputs))
reconstruction_loss *= 150 * 150 * 3
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)

# Compile VAE model
vae.compile(optimizer='adam')

# Print model summary
vae.summary()

# Train the model
history = vae.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the model
vae.save(dataset_path + "\\my_vae.h5")

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
