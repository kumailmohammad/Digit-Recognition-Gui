import tensorflow as tf

import tensorflow_datasets as tfds
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dense , Flatten 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


(x_train, y_train) ,(x_test, y_test) = mnist.load_data()

# # Create an image data generator for augmentation
# datagen = ImageDataGenerator(
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.1,
#     shear_range=0.1,
#     horizontal_flip=False
# )
datagen = ImageDataGenerator(
    rotation_range=20,          # Random rotation up to 20 degrees
    width_shift_range=0.2,      # Random horizontal shift
    height_shift_range=0.2,     # Random vertical shift
    zoom_range=0.2,             # Random zoom
    shear_range=10,             # Random shear
    fill_mode="nearest",        # Filling in the gaps after transformations
    brightness_range=[0.8, 1.2] # Random brightness adjustments
)

# Fit the generator to your data

# Train the model with data augmentation

x_train = x_train /255.0
x_test = x_test / 255.0
x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

datagen.fit(x_train)
# creating a model 
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Convolutional layer
    MaxPooling2D((2, 2)),  # Max pooling layer
    Conv2D(64, (3, 3), activation='relu'),  # Another convolutional layer
    MaxPooling2D((2, 2)),  # Max pooling layer
    Flatten(),  # Flattening the output for the dense layer
    Dense(128, activation='relu'),  # Fully connected layer with 128 neurons
    Dense(10, activation='softmax')  # Output layer for 10 classes (digits 0-9)
])


model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=[])

model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=5, validation_data=(x_test, y_test))
# model.fit(x_train,y_train,epochs = 1)

model.save('mnist_model_test.h5')

