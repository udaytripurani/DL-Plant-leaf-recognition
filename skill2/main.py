import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Define data paths
data_dir = r'C:\Users\venus\OneDrive\Desktop\deep learning\sample dataset'

# Image dimensions
img_height, img_width = 150, 150

# Batch size
batch_size = 32

# Prepare data generators
datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    os.path.join(data_dir),
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

# Define optimization techniques
optimizers = ['adam', 'sgd', 'rmsprop']

for optimizer_name in optimizers:
    # Define the model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model with the specified optimizer
    if optimizer_name == 'adam':
        optimizer = Adam()
    elif optimizer_name == 'sgd':
        optimizer = SGD()
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop()
    
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=10)
    
    # Save the model
    model.save(f'binary_classification_model_{optimizer_name}.h5')
    print(f"Model with {optimizer_name} optimizer saved successfully.")
