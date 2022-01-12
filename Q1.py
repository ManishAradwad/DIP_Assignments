"""
Author: Manish Aradwad
Date: 21/11/21

Assignment 5 - Q1

"""

# Importing libraries
import os
import pandas as pd
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Model
import numpy as np
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.layers import Flatten, Input

categories = ["airplanes", "bikes", "cars", "faces"]
data_path = "./Images"

le = preprocessing.LabelEncoder()
scaler = preprocessing.MinMaxScaler()

df_cats = []
df_files = []

# Getting all the images for training set
train_images = []
for cat in categories:
  train_path = os.path.join(data_path, cat+"_train")
  train_images = sorted(os.listdir(train_path))
  for img in train_images:
    df_files.append(train_path + "/" + img)
    df_cats.append(cat)

df_train = pd.DataFrame({
    'filename': df_files,
    'category': df_cats
})

# Encoding the labels of training set
le.fit(df_train['category'])
y = le.transform(df_train['category'])

# Getting the pre-trained VGG16 model without the final dense layer
base_model = VGG16(weights='imagenet', include_top=False)
inputs = Input(shape=(48, 48, 3), name = 'image_input')
x = base_model(inputs)
x = Flatten()(x)
model = Model(inputs = inputs, outputs = x)

x_train = []
y_train = []

# Extracting the features from the training set images and building a dataframe from the features
for f in df_train.filename[:]:
  img_path = f
  img = image.load_img(img_path, target_size=(48 ,48))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  features = model.predict(x)
  features_reduce = features.squeeze()
  x_train.append(features_reduce)
x_train = pd.DataFrame(x_train)
scaler.fit(x_train)
x_train = scaler.transform(x_train)

# Building training and validation sets
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y, test_size=0.2, stratify=y, random_state = 8)

# Building a KNN classifier
knn = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 1)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_valid)
print("\nAccuracy score on Validation Set: %0.2f" % accuracy_score(y_pred, y_valid))

df_cats = []
df_files = []

# Building the test set
test_images = []
for cat in categories:
  test_path = os.path.join(data_path, cat+"_test")
  test_images = sorted(os.listdir(test_path))
  for img in test_images:
    df_files.append(test_path + "/" + img)
    df_cats.append(cat)

df_test = pd.DataFrame({
    'filename': df_files,
    'category': df_cats
})

# Encoding labels of the test set
le.fit(df_test['category'])
y = le.transform(df_test['category'])

x_test = []
y_test = y

# Getting the features from test set images
for f in df_test.filename[:]:
  img_path = f
  img = image.load_img(img_path, target_size=(48 ,48))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)

  features = model.predict(x)
  features_reduce = features.squeeze()
  x_test.append(features_reduce)

# Predicting the labels for test set
x_test = pd.DataFrame(x_test)
scaler.fit(x_test)
x_test = scaler.transform(x_test)
y_pred = knn.predict(x_test)
print("\nAccuracy score on Test Set: ", accuracy_score(y_pred, y_test))