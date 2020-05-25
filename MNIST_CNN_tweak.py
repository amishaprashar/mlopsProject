
# coding: utf-8

# In[1]:


from keras.datasets import mnist


# In[2]:


from keras.models import Sequential


# In[3]:


from keras.layers import Dense


# In[4]:


from keras.utils import np_utils


# In[5]:


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


# In[6]:


num_pixels = X_train.shape[1] * X_train.shape[2] 
 


# In[7]:


X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')    


# In[8]:


X_train = X_train / 255
X_test = X_test / 255


# In[9]:


Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test) 
num_classes = Y_test.shape[1] 


# In[28]:


def baseline_model(neuron):
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[30]:


neuron = 5
model = baseline_model(neuron) 
accuracy = 0.0
 
 


# In[31]:


def buildModel():
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=200, verbose=0)
    scores = model.evaluate(X_test, Y_test, verbose=0)
    accuracy = scores[1]*100
    print("Accuracy: %.2f%%" % (scores[1]*100))
    return accuracy


# In[32]:


buildModel()
count = 0
best_acc = accuracy
best_neuron = 0


# In[33]:


def resetWeights():
    print("Reseting weights")
    w = model.get_weights()
    w = [[j*0 for j in i] for i in w]
    model.set_weights(w)
while accuracy < 99 and count < 4:
    print("Updating Model")
    model = baseline_model(neuron*2)
    neuron = neuron * 2
    count = count + 1
    accuracy = buildModel()
    if best_acc < accuracy:
        best_acc = accuracy
        best_neuron = neuron
    print()
    resetWeights()
print("**********")
print(best_neuron)
model = baseline_model(best_neuron)
buildModel()
model.save("mnist_modelUpdated.h5")
print("Model Saved")




# In[35]:


file1= open("result.txt", "w")
file1.write(str(best_acc))
file1.close()


# In[ ]:





# In[ ]:





# In[ ]:




