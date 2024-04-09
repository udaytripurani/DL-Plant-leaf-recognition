import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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

# Build the autoencoder model
def build_autoencoder(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    encoded = Conv2D(32, 3, activation='relu', padding='same')(x)
    
    # Decoder
    x = Conv2D(32, 3, activation='relu', padding='same')(encoded)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    decoded = Conv2D(3, 3, activation='sigmoid', padding='same')(x)  # Output image has 3 channels (RGB)
    
    model = Model(inputs, decoded)
    return model

# Instantiate the model
input_shape = (150, 150, 3)  # Assuming input images are 150x150 RGB images
autoencoder = build_autoencoder(input_shape)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mse')

# Print model summary
autoencoder.summary()

# Train the model
history = autoencoder.fit(train_generator, epochs=10, validation_data=validation_generator)

# Save the model
autoencoder.save("my_autoencoder.h5")

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
