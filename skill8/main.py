import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Path to your dataset
dataset_path = "C:\\Users\\venus\\OneDrive\\Desktop\\deep learning\\sample dataset"

# Using ImageDataGenerator to load and preprocess the images
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) # Assuming 20% validation split

# Training dataset
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),  # VGG model input size
    batch_size=32,
    class_mode='categorical',
    subset='training')  # Use subset 'training' for training data

# Validation dataset
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation')  # Use subset 'validation' for validation data

# Load the VGG16 model pretrained on ImageNet without the top (fully connected) layers
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the VGG model (optional)
for layer in vgg_model.layers:
    layer.trainable = False

# Add new fully connected layers for your classification task
model = Sequential([
    vgg_model,
    Flatten(),
    Dense(512, activation='relu'),
    Dense(3, activation='softmax')  # Assuming 3 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32)

# Save the model
model.save("my_vgg_model.h5")

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
