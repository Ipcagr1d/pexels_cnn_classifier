import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# set up search criteria
search_terms = ['lemon', 'lychee', 'grape', 'kiwi', 'apple'] # keywords for search

# set up parameters
img_size = (128, 128) # size to resize images
num_classes = len(search_terms) # number of fruit classes

# load images and labels
images = []
labels = []
for i, term in enumerate(search_terms):
    folder_name = f'{term}_images'
    for filename in os.listdir(folder_name):
        image_path = os.path.join(folder_name, filename)
        try:
            # read image and resize
            img = cv2.imread(image_path)
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(i)
        except Exception as e:
            print(f'Error loading {image_path}: {e}')

# convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# convert labels to categorical
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# create model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
