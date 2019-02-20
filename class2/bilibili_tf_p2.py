
# coding: utf-8

# In[79]:

import numpy as np
import tensorflow as tf


# In[20]:

m1 = tf.Variable([1,2])
m2 = tf.constant([3,3])
m3 = tf.subtract(m1,m2)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(m3))


# In[24]:

#定义变量state，初始值=0
state = tf.Variable(0, name = "counter")
#note：定义一个加一op
add_1 = tf.add(state,1)
#定义一个赋值op
update = tf.assign(state, add_1)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(5):
        print(sess.run(update))


# In[30]:

#fetch
add = tf.add(m1,m2)
mul = tf.multiply(m1,m2)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run([add, mul]))


# In[32]:

#feed
p1= tf.placeholder(tf.float32)
p2 = tf.placeholder(tf.float32)
op = tf.add(p1,p2)
init = tf.global_variables_initializer()
# with tf.Session() as sess:
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(op, feed_dict = {p1:1,p2:2}))


# In[95]:

np.random.seed(1)
x = np.random.rand(100)
y = x * 0.1 + 2

k = tf.Variable(0.)#note:一定是0.而不是0
b = tf.Variable(0.)
y1 = x*k+b

loss = tf.reduce_mean(tf.square(y1 - y))
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(201):
        sess.run(train)
        if i%20 == 0:
            print(i,sess.run([k,b]))




# In[96]:

np.random.seed(1)
x_data = np.random.rand(100)
y_data = x_data * 0.1 + 2

b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

loss = tf.reduce_mean(tf.square(y_data - y))
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            print(step, sess.run([k,b]))


# In[ ]:



