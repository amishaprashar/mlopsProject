
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


# In[8]:


num_pixels = X_train.shape[1] * X_train.shape[2] 
 


# In[12]:


X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')    


# In[13]:


X_train = X_train / 255
X_test = X_test / 255


# In[15]:


Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test) 
num_classes = Y_test.shape[1] 


# In[22]:


def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# In[23]:


model = baseline_model()
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=200, verbose=0)


# In[25]:


scores = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
 


# In[32]:


model.save("mnist_model.h5") 
 


# In[33]:


file1= open("result.txt", "w")
file1.write(str(scores[1]*100))


# In[ ]:





# In[ ]:





# In[ ]:




