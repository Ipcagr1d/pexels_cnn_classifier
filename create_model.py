import csv
import os
import urllib.request
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load CSV file
csv_file = 'dataset.csv'
data = []
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header row
    for row in reader:
        image_id, image_url = row
        data.append((image_id, image_url))

# Download images and save to folder
folder_name = 'images'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

for image_id, image_url in data:
    filename = os.path.join(folder_name, f'{image_id}.jpg')
    urllib.request.urlretrieve(image_url, filename)

# Load and preprocess images
images = []
labels = []
for image_id, _ in data:
    filename = os.path.join(folder_name, f'{image_id}.jpg')
    img = cv2.imread(filename)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0  # normalize pixel values
    images.append(img)
    
    # Get label from image ID
    label = int(image_id.split('_')[-1])  # assume label is the last part of the image ID
    labels.append(label)

# Split data into training, validation, and testing sets
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Define CNN architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Evaluate the model on the testing set
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc:.2f}')

# Save the model
model.save('model.h5')
