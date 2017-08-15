
# coding: utf-8

# In[85]:


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[86]:


mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


# In[87]:


x = tf.placeholder( tf.float32, [None,784])
y_ = tf.placeholder(tf.float32, [None,10])


# In[88]:


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# In[89]:


def  conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# In[90]:


W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])


# In[91]:


x_image = tf.reshape(x,[-1,28,28,1])


# In[92]:


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+ b_conv1)


# In[93]:


h_pool1 = max_pool_2x2(h_conv1)


# In[94]:


W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])


# In[95]:


h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2)+ b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


# In[96]:


W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])


# In[97]:


h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)


# In[98]:


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# In[99]:


W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2)+b_fc2


# In[100]:


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[ ]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i %100==0:
            train_accuraccy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob:1.0})
            print('step %d training accuracy %g'%(i, train_accuraccy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        
    print('test accuracy %g'% accuracy.eval(feed_dict={x:mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        

