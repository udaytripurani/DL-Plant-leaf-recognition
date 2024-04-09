import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define path for your dataset
dataset_dir = r'C:\Users\venus\OneDrive\Desktop\deep learning\sample dataset'

# Image parameters
img_width, img_height = 150, 150
input_shape = (img_width, img_height, 3)
epochs = 10
batch_size = 32
num_classes = len(os.listdir(dataset_dir))

# Image data generator
datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# CNN model
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten()
])

# LSTM model
lstm_model = Sequential([
    LSTM(128, return_sequences=True),
    LSTM(64),
    Dense(num_classes, activation='softmax')
])

# Combined CNN-LSTM model
combined_model = Sequential([
    cnn_model,
    tf.keras.layers.Reshape((1, -1)),
    lstm_model
])

# Compile the model
combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = combined_model.fit(
    generator,
    steps_per_epoch=generator.samples // batch_size,
    epochs=epochs)

# Save the model
combined_model.save("cnn_lstm_model.h5")

# Plot training history
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
