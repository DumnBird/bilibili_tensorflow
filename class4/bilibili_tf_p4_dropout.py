
# coding: utf-8

# In[3]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# In[4]:

mnist = input_data.read_data_sets( 'D:\MNIST',one_hot = True)


# In[8]:

batch_size = 100
num_batch = mnist.train.num_examples // batch_size


num_nodes_L0 = 28 * 28
num_labels  = 10

num_nodes_L1 = 2000
num_nodes_L2 = 1000
num_nodes_L3 = 10


#structure start
X = tf.placeholder(tf.float32, [None, num_nodes_L0])
Y = tf.placeholder(tf.float32, [None, num_labels])
keep_prob = tf.placeholder(tf.float32)


# W_L1 = tf.Variable(tf.random_normal([num_nodes_L0, num_nodes_L1]))效果最差，因为初始权值太大---均值为0 ，方差为1
W_L1 = tf.Variable(tf.random_normal([num_nodes_L0, num_nodes_L1],stddev=0.2))#效果不错，和0初始化差不多
# W_L1 = tf.Variable(tf.zeros([num_nodes_L0, num_nodes_L1]))
b_L1 = tf.Variable(tf.zeros([1, num_nodes_L1]))
XW_plus_b_L1 = tf.matmul(X, W_L1) + b_L1
L1 = tf.nn.relu(XW_plus_b_L1)
L1_drop = tf.nn.dropout(L1, keep_prob)

W_L2 = tf.Variable(tf.random_normal([num_nodes_L1, num_nodes_L2],stddev=0.2))
b_L2 = tf.Variable(tf.zeros([1, num_nodes_L2]))
XW_plus_b_L2 = tf.matmul(L1, W_L2) + b_L2
L2 = tf.nn.relu(XW_plus_b_L2)
L2_drop = tf.nn.dropout(L2, keep_prob)


W_L3 = tf.Variable(tf.random_normal([num_nodes_L2, num_nodes_L3],stddev=0.2))
b_L3 = tf.Variable(tf.zeros([1, num_nodes_L3]))
XW_plus_b_L3 = tf.matmul(L2, W_L3) + b_L3
L3 = tf.nn.softmax(XW_plus_b_L3)



#二次代价函数
# loss = tf.reduce_mean(tf.square(L1 - Y))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=XW_plus_b_L3))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#布尔值
correct_prediction_bool = tf.equal(tf.argmax(L3, 1), tf.argmax(Y, 1))
#布尔转float32
correct_prediction_float = tf.cast(correct_prediction_bool, tf.float32)
#准确率
accuracy = tf.reduce_mean(correct_prediction_float)
#structure end

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10):
        for batch in range(num_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict = {X:batch_x, Y:batch_y})
        print(epoch, ":", "test_acc=",sess.run(accuracy, feed_dict = {X:mnist.test.images, Y:mnist.test.labels ,keep_prob:0.8}),",",
              "train_acc=",sess.run(accuracy, feed_dict = {X:mnist.train.images, Y:mnist.train.labels, keep_prob:0.8}))
    


# In[ ]:
"""
2layer
0 : test_acc= 0.9331 , train_acc= 0.94258183
1 : test_acc= 0.9464 , train_acc= 0.96263635
2 : test_acc= 0.9521 , train_acc= 0.9738727
3 : test_acc= 0.958 , train_acc= 0.98
4 : test_acc= 0.961 , train_acc= 0.9840364
5 : test_acc= 0.9638 , train_acc= 0.9883818
6 : test_acc= 0.9652 , train_acc= 0.9893818
7 : test_acc= 0.9658 , train_acc= 0.99241817
8 : test_acc= 0.9678 , train_acc= 0.9948
9 : test_acc= 0.9677 , train_acc= 0.9953455
10 : test_acc= 0.9696 , train_acc= 0.997
11 : test_acc= 0.9704 , train_acc= 0.99743634
12 : test_acc= 0.9696 , train_acc= 0.9982909
13 : test_acc= 0.9704 , train_acc= 0.99912727
14 : test_acc= 0.971 , train_acc= 0.9992727
15 : test_acc= 0.9713 , train_acc= 0.9993455
16 : test_acc= 0.9721 , train_acc= 0.9995273
17 : test_acc= 0.9716 , train_acc= 0.99965453
18 : test_acc= 0.9723 , train_acc= 0.99978185
19 : test_acc= 0.9734 , train_acc= 0.99972725
    

#3layer
0 : test_acc= 0.9372 , train_acc= 0.96007276
1 : test_acc= 0.9497 , train_acc= 0.9834545
2 : test_acc= 0.9531 , train_acc= 0.9930182
3 : test_acc= 0.9574 , train_acc= 0.9971273
4 : test_acc= 0.9597 , train_acc= 0.9992909
5 : test_acc= 0.9608 , train_acc= 0.99970907
6 : test_acc= 0.9606 , train_acc= 0.9999273
7 : test_acc= 0.9619 , train_acc= 0.99994546
8 : test_acc= 0.9622 , train_acc= 1.0
9 : test_acc= 0.962 , train_acc= 1.0

#3layer with dropout=0.8
0 : test_acc= 0.9307 , train_acc= 0.94865453
1 : test_acc= 0.9486 , train_acc= 0.97481817
2 : test_acc= 0.9513 , train_acc= 0.98481816
3 : test_acc= 0.956 , train_acc= 0.99143636
4 : test_acc= 0.9587 , train_acc= 0.99512726
5 : test_acc= 0.9591 , train_acc= 0.9977091
6 : test_acc= 0.96 , train_acc= 0.99874544
7 : test_acc= 0.961 , train_acc= 0.99907273
8 : test_acc= 0.961 , train_acc= 0.9996909
9 : test_acc= 0.9621 , train_acc= 0.9998182

"""