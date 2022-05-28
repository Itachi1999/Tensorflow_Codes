import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential

#The X and Y data mapped as y = 2x - 1

X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0], dtype= float)
Y = np.array([-3.0, -1.0, 1.0, 3.0, 5.0], dtype = float)

#The model with 1 neuron where the optimizer is SGD and loss function is SME
model = Sequential([Dense(units = 1, input_shape = [1])])
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')

#Training the model to fit the data with epochs of 500
model.fit(X, Y, epochs = 500)

#Making a prediction from the model where the input is 100.0 and the ideal output should be 199.0
print(model.predict([100.0]))

#The result came as [[198.89265]], very close to 199.0 but not equal to 199.0