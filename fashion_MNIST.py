import tensorflow as tf
import numpy as np
import keras.layers as layers
import keras.models as models 
import tensorflow.keras.datasets as ds

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#loading the fashion mnist data in traning and test sets
fashion_mnist =  ds.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#normnlise the data so that the weights and pixel values don't vary that much
train_images = train_images / 255.0
test_images = test_images / 255.0


model = models.Sequential([layers.Flatten(input_shape = (28,28)), 
        layers.Dense(128, activation = 'relu')])
model.add(layers.Dense(64, activation = 'relu'))        
model.add(layers.Dense(10, activation = 'softmax'))

model.compile(optimizer= tf.optimizers.Adam(), loss = 'sparse_categorical_crossentropy',
            metrics= ['accuracy'])
model.fit(train_images, train_labels, epochs = 20)

model.evaluate(test_images, test_labels)

predictions = model.predict(test_images)
print(type(predictions))
print('The shape of predictions:', np.shape(predictions))
print('The shape of test labels:', np.shape(test_labels))

predictions_max_from_softmax = []
for i in range(len(test_labels)):
    max = 0.0
    max_index = 0
    for j in range(10):
        if max < predictions[i][j]:
            max = predictions[i][j]
            max_index = j
    predictions_max_from_softmax.append(max_index)

print(np.shape(predictions_max_from_softmax))
sum = 0

for i in range(len(predictions_max_from_softmax)):
    if (predictions_max_from_softmax[i] == test_labels[i]):
         sum += 1
    else: continue


#print(model.weights)
print(np.shape(train_images))
print(sum / len(predictions))