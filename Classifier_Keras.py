
# coding: utf-8

# In[1]:

from keras.models import Sequential
from keras.layers import Dense 
import numpy as np
from keras.utils import np_utils


# In[2]:

np.random.seed(7)


# In[3]:

from keras.datasets import mnist


# In[4]:

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[5]:

print(X_train.shape)


# In[6]:

X_train = X_train.reshape(X_train.shape[0],28,28)
X_test = X_test.reshape(X_test.shape[0],28,28)


# In[7]:

X_train=X_train.reshape((X_train.shape[0],-1),order='F')
X_test = X_test.reshape((X_test.shape[0],-1), order='F')
X_train[0]


# In[8]:

a = np.array([[[ 1.,  5.,  4.],
    [ 1.,  5.,  4.],
    [ 1.,  2.,  4.]],

   [[ 3.,  6.,  4.],
    [ 6.,  6.,  4.],
    [ 6.,  6.,  4.]]])


# In[9]:

a.reshape((a.shape[0],-1), order='F')


# In[10]:

X_train=X_train.astype('float32')
X_train/=255


# In[11]:

X_test = X_test.astype('float32')
X_test/=255


# In[12]:

print(y_train[:10])


# In[13]:

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print(y_train.shape)


# In[14]:

model = Sequential()


# In[15]:

model.add(Dense(784, input_dim=(784), activation = 'sigmoid'))


# In[16]:

model.output_shape


# In[17]:

model.add(Dense(30,activation='sigmoid'))


# In[18]:

model.output_shape


# In[19]:

model.add(Dense(10, activation='sigmoid'))


# In[20]:

model.output_shape


# In[21]:

print(model.summary())


# In[1]:

model.compile(loss='categorical_crossentropy', optimizer='sgd',metrics=['accuracy'])


# In[ ]:

model.fit(X_train, y_train, batch_size=15, nb_epoch=15, verbose=1)


# In[ ]:
scores=model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[ ]:



