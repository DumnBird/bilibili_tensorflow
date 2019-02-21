
# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets( 'D:\MNIST',one_hot = True)


# In[4]:

batch_size = 100
num_batch = mnist.train.num_examples // batch_size


num_nodes_L0 = 28 * 28
num_labels  = 10

num_nodes_L1 = 10



#structure start
X = tf.placeholder(tf.float32, [None, num_nodes_L0])
Y = tf.placeholder(tf.float32, [None, num_labels])


# W_L1 = tf.Variable(tf.random_normal([num_nodes_L0, num_nodes_L1]))效果最差，因为初始权值太大---均值为0 ，方差为1
W_L1 = tf.Variable(tf.random_normal([num_nodes_L0, num_nodes_L1],stddev=0.2))#效果不错，和0初始化差不多
# W_L1 = tf.Variable(tf.zeros([num_nodes_L0, num_nodes_L1]))
b_L1 = tf.Variable(tf.zeros([1, num_nodes_L1]))
XW_plus_b = tf.matmul(X, W_L1) + b_L1
L1 = tf.nn.softmax(XW_plus_b)

loss = tf.reduce_mean(tf.square(L1 - Y))
#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


#布尔值
correct_prediction_bool = tf.equal(tf.argmax(L1, 1), tf.argmax(Y, 1))
#布尔转float32
correct_prediction_float = tf.cast(correct_prediction_bool, tf.float32)
#准确率
accuracy = tf.reduce_mean(correct_prediction_float)
#structure end

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(60):
        for batch in range(num_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict = {X:batch_x, Y:batch_y})

        print(epoch, ",", sess.run(accuracy, feed_dict = {X:mnist.test.images, Y:mnist.test.labels}))
    

