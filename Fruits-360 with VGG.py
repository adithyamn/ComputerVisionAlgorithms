from __future__ import print_function, division
from builtins import range, input
from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt 

from glob import glob

#Preprocessing 
IMAGE_SIZE = [100, 100]

#Training Configuration
epochs = 5
batch_size = 32

#Specify Path
train_path = 'D:/Folders/ML & CV/Advanced Computer Vision/Fruits_360/fruits-360/fruits-360-small/Training'
valid_path = 'D:/Folders/ML & CV/Advanced Computer Vision/Fruits_360/fruits-360/fruits-360-small/Validation'


#Using a Glob function to get the files related to a particular String.
image_files = glob(train_path + '/*/*.jp*g')   
valid_image_files = glob(valid_path + '/*/*.jp*g')

#Getting number of classes by looking at the number of folders(classes)
folders = glob(train_path + '/*')


#Viewing Random Image
plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()

#Implementing the VGG model.
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)	#Getting the input shape as the image size
																					#Using the pretrained weights from imagenet.
																					#include_top is False. Not considering the Last Layer.
																					#Using Transfer Learning to train the current dataset.

for layer in vgg.layers:	#Goes through each layer in the VGG architecture.
  layer.trainable = False 	#Not training the exsisting weights in VGG. 
  							#Setting the trainalbe attribute to False.

#Adding the output layer (transfer Learning) 
x = Flatten()(vgg.output) 									#Using Keras functional API to add the output layer to the VGG
prediction = Dense(len(folders), activation='softmax')(x)	#Using a simple Logistics Regresssion. 
															#Logistic Regression is used for MultiClass Classificaiton.
															#Softmax Activation Funciton.



#Compiling the functional API
model = Model(inputs=vgg.input, outputs=prediction)

# View the Summary
model.summary()

model.compile(						#Compiling the Model.
  loss='categorical_crossentropy',	#Specifying the Loss function, Activation Function and Accuracy Mertrics
  optimizer='rmsprop',
  metrics=['accuracy']
)

# Using an Image Data Generator to AUGMENT the data in random ways for better generalisation
gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input
)

#Get Label mapping for confusion matrix plot later
test_gen = gen.flow_from_directory(valid_path, target_size=IMAGE_SIZE)
print(test_gen.class_indices)    #Gettting the Class indecies (0-(K-1))
labels = [None] * len(test_gen.class_indices) 
for k, v in test_gen.class_indices.items():
  labels[v] = k

#Checking the color space conversion from RGB to BGR
for x, y in test_gen:
  print("min:", x[0].min(), "max:", x[0].max())
  plt.title(labels[np.argmax(y[0])])
  plt.imshow(x[0])
  plt.show()
  break


# Creating Generators
train_generator = gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)
valid_generator = gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)


#Fitting the Model
r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=epochs,
  steps_per_epoch=len(image_files) // batch_size,
  validation_steps=len(valid_image_files) // batch_size,
)



def get_confusion_matrix(data_path, N):
  print("Generating confusion matrix", N)
  predictions = []
  targets = []
  i = 0
  for x, y in gen.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size * 2):
    i += 1
    if i % 50 == 0:
      print(i)
    p = model.predict(x)
    p = np.argmax(p, axis=1)
    y = np.argmax(y, axis=1)
    predictions = np.concatenate((predictions, p))
    targets = np.concatenate((targets, y))
    if len(targets) >= N:
      break

  cm = confusion_matrix(targets, predictions)
  return cm


cm = get_confusion_matrix(train_path, len(image_files))
print(cm)
valid_cm = get_confusion_matrix(valid_path, len(valid_image_files))
print(valid_cm)




# Loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()

# Accuracy
plt.plot(r.history['accuracy'], label='train accuracy')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
