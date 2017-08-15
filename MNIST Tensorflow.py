
# coding: utf-8

# In[2]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[3]:


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


# In[5]:


x = tf.placeholder( tf.float32, [None,784])


# In[6]:


W = tf.Variable(tf.zeros([784,10]))


# In[7]:


b = tf.Variable(tf.zeros([10]))


# In[13]:


y = tf.matmul(x,W)+b


# In[14]:


y_ = tf.placeholder(tf.float32, [None,10])


# In[15]:


"""cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
"""

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# In[16]:


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# In[17]:


sess = tf.InteractiveSession()


# In[18]:


tf.global_variables_initializer().run()


# In[19]:


for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_: batch_ys})


# In[ ]:
correct_prediction = tf.equal(tf.argmax(y,1),tf.arg_max(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_ :mnist.test.labels}))

