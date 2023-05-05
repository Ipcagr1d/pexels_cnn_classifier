# CNN Creation with Dataset used in Drive

This is the implementation of CNN from the data that we've collected from the followig url `https://drive.google.com/drive/folders/11muQD-KCutA0YQng_oDdF0O2DNJciyw5?usp=sharing `. As a pre-requirement we must mount the Drive with the dataset located so that colab VM has a right to access said images in the drive and ths begin the model creation process

# Image Pre-processing & CNN Model Creation

```python
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
    folder_name = os.path.join('/content/drive/MyDrive/New Data set', f'{term}_images')
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
```

The output

```python
Error loading /content/drive/MyDrive/New Data set/lemon_images/lemon_urls.csv: OpenCV(4.7.0) /io/opencv/modules/imgproc/src/resize.cpp:4062: error: (-215:Assertion failed) !ssize.empty() in function 'resize'

Error loading /content/drive/MyDrive/New Data set/lychee_images/lychee_urls.csv: OpenCV(4.7.0) /io/opencv/modules/imgproc/src/resize.cpp:4062: error: (-215:Assertion failed) !ssize.empty() in function 'resize'

Error loading /content/drive/MyDrive/New Data set/grape_images/grape_urls.csv: OpenCV(4.7.0) /io/opencv/modules/imgproc/src/resize.cpp:4062: error: (-215:Assertion failed) !ssize.empty() in function 'resize'

Epoch 1/10
24/24 [==============================] - 33s 1s/step - loss: 11.2964 - accuracy: 0.4913 - val_loss: 0.6095 - val_accuracy: 0.7181
Epoch 2/10
24/24 [==============================] - 31s 1s/step - loss: 0.5912 - accuracy: 0.7770 - val_loss: 0.5851 - val_accuracy: 0.7979
Epoch 3/10
24/24 [==============================] - 33s 1s/step - loss: 0.4830 - accuracy: 0.8144 - val_loss: 0.5787 - val_accuracy: 0.7819
Epoch 4/10
24/24 [==============================] - 33s 1s/step - loss: 0.4872 - accuracy: 0.8117 - val_loss: 0.5560 - val_accuracy: 0.8191
Epoch 5/10
24/24 [==============================] - 32s 1s/step - loss: 0.3345 - accuracy: 0.8865 - val_loss: 0.4266 - val_accuracy: 0.8564
Epoch 6/10
24/24 [==============================] - 31s 1s/step - loss: 0.2545 - accuracy: 0.9186 - val_loss: 0.5444 - val_accuracy: 0.8085
Epoch 7/10
24/24 [==============================] - 31s 1s/step - loss: 0.3544 - accuracy: 0.8638 - val_loss: 0.4461 - val_accuracy: 0.8351
Epoch 8/10
24/24 [==============================] - 31s 1s/step - loss: 0.1812 - accuracy: 0.9252 - val_loss: 0.5063 - val_accuracy: 0.8298
Epoch 9/10
24/24 [==============================] - 32s 1s/step - loss: 0.1483 - accuracy: 0.9399 - val_loss: 0.4360 - val_accuracy: 0.8830
Epoch 10/10
24/24 [==============================] - 34s 1s/step - loss: 0.1215 - accuracy: 0.9426 - val_loss: 0.4085 - val_accuracy: 0.8936
<keras.callbacks.History at 0x7f74c3733790>
```

Model performance plotting for plotting accuracy loss meant to measure the performance of the generated model itself.

```python
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

pre = Precision()
re = Recall()
acc = BinaryAccuracy()

# iterate over test set
for batch in zip(np.array_split(X_test, 32), np.array_split(y_test, 32)):
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f"Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}") 
```

Evaluation for as the name means, to evaluate the model itself

```python
1/1 [==============================] - 0s 199ms/step
1/1 [==============================] - 0s 71ms/step
1/1 [==============================] - 0s 75ms/step
1/1 [==============================] - 0s 71ms/step
1/1 [==============================] - 0s 77ms/step
1/1 [==============================] - 0s 75ms/step
1/1 [==============================] - 0s 73ms/step
1/1 [==============================] - 0s 85ms/step
1/1 [==============================] - 0s 73ms/step
1/1 [==============================] - 0s 75ms/step
1/1 [==============================] - 0s 78ms/step
1/1 [==============================] - 0s 121ms/step
1/1 [==============================] - 0s 127ms/step
1/1 [==============================] - 0s 134ms/step
1/1 [==============================] - 0s 130ms/step
1/1 [==============================] - 0s 134ms/step
1/1 [==============================] - 0s 145ms/step
1/1 [==============================] - 0s 145ms/step
1/1 [==============================] - 0s 134ms/step
1/1 [==============================] - 0s 128ms/step
1/1 [==============================] - 0s 127ms/step
1/1 [==============================] - 0s 133ms/step
1/1 [==============================] - 0s 134ms/step
1/1 [==============================] - 0s 145ms/step
1/1 [==============================] - 0s 122ms/step
1/1 [==============================] - 0s 138ms/step
1/1 [==============================] - 0s 129ms/step
1/1 [==============================] - 0s 130ms/step
1/1 [==============================] - 0s 120ms/step
1/1 [==============================] - 0s 106ms/step
1/1 [==============================] - 0s 109ms/step
1/1 [==============================] - 0s 82ms/step
Precision: 0.8563829660415649, Recall: 0.8563829660415649, Accuracy: 0.9425531625747681
```
