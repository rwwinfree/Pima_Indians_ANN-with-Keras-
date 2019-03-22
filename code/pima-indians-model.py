
# coding: utf-8

# The following code is for the tutorial "Develop Your First Neural Network in Python With Keras Step-By-Step"
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
import numpy
#fix random seed for reproducibility
numpy.random.seed(7)


# In[ ]:


#load pima indians dataset
dataset = numpy.loadtxt("../dataset/pima-indians-diabetes.csv", delimiter=",")


# In[ ]:


#split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]


# In[ ]:


# Create the model itself
model = Sequential()
#The first layer of the model. #12 neurons, 8 input vars. Using the rectifier activation function
model.add(Dense(12, input_dim=8, activation='relu'))
#Layer 2
model.add(Dense(8, activation='relu'))
#Layer 3; 1 neuron to predict yes or no for onset of diabetes
# Sigmoid forces either 0 or 1 with a default threshold of 0.5
model.add(Dense(1,activation='sigmoid'))


# In[ ]:


# Compile the Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[ ]:


# Fit (train) the model on the training data
model.fit(X, Y, epochs=150, batch_size=100)


# In[ ]:


# Evaluate the performance of the model
# Usually we have training and testing data. For this example they only had training data. So will evealuate on the training data. 
# Usually want separate because just evaluating on the training data may just show you how well the model just fits that
# set of data
scores = model.evaluate(X,Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:


# calculate predictions
predictions = model.predict(X)


# In[ ]:


# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)

