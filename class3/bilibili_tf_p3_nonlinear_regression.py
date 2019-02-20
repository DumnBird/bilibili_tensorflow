
# coding: utf-8

# In[10]:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# In[24]:

X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32, [None, 1])

W_L1 = tf.Variable(tf.random_normal([1,10]))
b_L1 = tf.Variable(tf.zeros([1,10]))
WX_plus_b_L1 = tf.matmul(X, W_L1) + b_L1
L1 = tf.nn.tanh(WX_plus_b_L1)


W_L2 = tf.Variable(tf.random_normal([10,1]))
b_L2 = tf.Variable(tf.zeros([1,1]))
WX_plus_b_L2 = tf.matmul(L1, W_L2) + b_L2
prediction = tf.nn.tanh(WX_plus_b_L2)

loss = tf.reduce_mean(tf.square(Y - prediction))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(10000):
        sess.run(train_step, feed_dict = {X:X_data, Y:Y_data})
    prediction_value = sess.run(prediction, feed_dict = {X:X_data})
    
    plt.figure()
    plt.scatter(X_data, Y_data)
    plt.plot(X_data, prediction_value, 'r-', lw = 1)
    plt.show()
    





# In[16]:

X_data = np.linspace(-1,1,200).astype(np.float32)[:,np.newaxis]
print(X_data.shape)
noise = np.random.normal(0, 0.02, X_data.shape)
print(Y_data.shape)
Y_data = np.square(X_data) + noise


# In[ ]:



