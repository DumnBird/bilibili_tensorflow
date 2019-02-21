
# coding: utf-8

# In[2]:

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets( 'D:\MNIST',one_hot = True)


# In[5]:

batch_size = 100
num_batch = mnist.train.num_examples // batch_size


num_nodes_L0 = 28 * 28
num_labels  = 10

num_nodes_L1 = 500
num_nodes_L2 = 100
num_nodes_L3 = 10


#structure start
X = tf.placeholder(tf.float32, [None, num_nodes_L0])
Y = tf.placeholder(tf.float32, [None, num_labels])
keep_prob = tf.placeholder(tf.float32)
lr = tf.placeholder(tf.float32)


# W_L1 = tf.Variable(tf.random_normal([num_nodes_L0, num_nodes_L1]))效果最差，因为初始权值太大---均值为0 ，方差为1
W_L1 = tf.Variable(tf.random_normal([num_nodes_L0, num_nodes_L1],stddev=0.2))#效果不错，和0初始化差不多
# W_L1 = tf.Variable(tf.zeros([num_nodes_L0, num_nodes_L1]))
b_L1 = tf.Variable(tf.zeros([1, num_nodes_L1]))
XW_plus_b_L1 = tf.matmul(X, W_L1) + b_L1
L1 = tf.nn.relu(XW_plus_b_L1)
# L1_drop = tf.nn.dropout(L1, keep_prob)

W_L2 = tf.Variable(tf.random_normal([num_nodes_L1, num_nodes_L2],stddev=0.2))
b_L2 = tf.Variable(tf.zeros([1, num_nodes_L2]))
XW_plus_b_L2 = tf.matmul(L1, W_L2) + b_L2
L2 = tf.nn.relu(XW_plus_b_L2)
# L2_drop = tf.nn.dropout(L2, keep_prob)


W_L3 = tf.Variable(tf.random_normal([num_nodes_L2, num_nodes_L3],stddev=0.2))
b_L3 = tf.Variable(tf.zeros([1, num_nodes_L3]))
XW_plus_b_L3 = tf.matmul(L2, W_L3) + b_L3
L3 = tf.nn.softmax(XW_plus_b_L3)



#二次代价函数
# loss = tf.reduce_mean(tf.square(L1 - Y))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=XW_plus_b_L3))

train_step = tf.train.AdamOptimizer(lr).minimize(loss)

#布尔值
correct_prediction_bool = tf.equal(tf.argmax(L3, 1), tf.argmax(Y, 1))
#布尔转float32
correct_prediction_float = tf.cast(correct_prediction_bool, tf.float32)
#准确率
accuracy = tf.reduce_mean(correct_prediction_float)
#structure end

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(60):
        lr_dynamic = 0.001 * (0.98 ** epoch)
        for batch in range(num_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict = {X:batch_x, Y:batch_y, lr:lr_dynamic})
        print(epoch, ":", "test_acc=",sess.run(accuracy, feed_dict = {X:mnist.test.images, Y:mnist.test.labels }),",",
              "train_acc=",sess.run(accuracy, feed_dict = {X:mnist.train.images, Y:mnist.train.labels}))
    
    


# In[ ]:
"""
GdOptimizer,lr = 0.1
0 : test_acc= 0.9351 , train_acc= 0.93845457
1 : test_acc= 0.9499 , train_acc= 0.95796365
2 : test_acc= 0.9567 , train_acc= 0.9680909
3 : test_acc= 0.9558 , train_acc= 0.9705273
4 : test_acc= 0.961 , train_acc= 0.9796
5 : test_acc= 0.9662 , train_acc= 0.9843091
6 : test_acc= 0.9672 , train_acc= 0.9876
7 : test_acc= 0.9637 , train_acc= 0.98623633
8 : test_acc= 0.9685 , train_acc= 0.99227273
9 : test_acc= 0.9687 , train_acc= 0.9931091
10 : test_acc= 0.9692 , train_acc= 0.9953455
11 : test_acc= 0.9698 , train_acc= 0.9965636
12 : test_acc= 0.9697 , train_acc= 0.99743634
13 : test_acc= 0.9687 , train_acc= 0.99814546
14 : test_acc= 0.969 , train_acc= 0.9986
15 : test_acc= 0.9709 , train_acc= 0.99923635
16 : test_acc= 0.9708 , train_acc= 0.9994364
17 : test_acc= 0.9698 , train_acc= 0.9995091
18 : test_acc= 0.9708 , train_acc= 0.99972725
19 : test_acc= 0.9711 , train_acc= 0.9998
20 : test_acc= 0.972 , train_acc= 0.99987274
21 : test_acc= 0.9707 , train_acc= 0.9999273
22 : test_acc= 0.9714 , train_acc= 0.99994546
23 : test_acc= 0.9715 , train_acc= 0.9998909
24 : test_acc= 0.9716 , train_acc= 0.9999818
25 : test_acc= 0.9711 , train_acc= 0.9999818
26 : test_acc= 0.9718 , train_acc= 1.0
27 : test_acc= 0.9718 , train_acc= 1.0
28 : test_acc= 0.9715 , train_acc= 1.0
29 : test_acc= 0.972 , train_acc= 1.0
30 : test_acc= 0.9719 , train_acc= 1.0
31 : test_acc= 0.972 , train_acc= 1.0
32 : test_acc= 0.9722 , train_acc= 1.0
33 : test_acc= 0.9723 , train_acc= 1.0
34 : test_acc= 0.9725 , train_acc= 1.0
35 : test_acc= 0.972 , train_acc= 1.0
36 : test_acc= 0.9724 , train_acc= 1.0
37 : test_acc= 0.9725 , train_acc= 1.0
38 : test_acc= 0.972 , train_acc= 1.0
39 : test_acc= 0.9723 , train_acc= 1.0
40 : test_acc= 0.9726 , train_acc= 1.0
41 : test_acc= 0.9726 , train_acc= 1.0
42 : test_acc= 0.9724 , train_acc= 1.0
43 : test_acc= 0.9722 , train_acc= 1.0
44 : test_acc= 0.973 , train_acc= 1.0
45 : test_acc= 0.9727 , train_acc= 1.0
46 : test_acc= 0.9724 , train_acc= 1.0
47 : test_acc= 0.9727 , train_acc= 1.0
48 : test_acc= 0.9727 , train_acc= 1.0
49 : test_acc= 0.9727 , train_acc= 1.0
50 : test_acc= 0.9723 , train_acc= 1.0
51 : test_acc= 0.9727 , train_acc= 1.0
52 : test_acc= 0.9728 , train_acc= 1.0
53 : test_acc= 0.9728 , train_acc= 1.0
54 : test_acc= 0.9729 , train_acc= 1.0
55 : test_acc= 0.9726 , train_acc= 1.0
56 : test_acc= 0.973 , train_acc= 1.0
57 : test_acc= 0.9728 , train_acc= 1.0
58 : test_acc= 0.9729 , train_acc= 1.0
59 : test_acc= 0.9728 , train_acc= 1.0
    
    
#AdamOptimizer, lr = 0.001
0 : test_acc= 0.9469 , train_acc= 0.9584727
1 : test_acc= 0.9553 , train_acc= 0.97354543
2 : test_acc= 0.9659 , train_acc= 0.98556364
3 : test_acc= 0.9686 , train_acc= 0.9912
4 : test_acc= 0.9679 , train_acc= 0.9931818
5 : test_acc= 0.9679 , train_acc= 0.9908
6 : test_acc= 0.9702 , train_acc= 0.9944909
7 : test_acc= 0.9746 , train_acc= 0.9977091
8 : test_acc= 0.9709 , train_acc= 0.99587274
9 : test_acc= 0.9709 , train_acc= 0.9945818
10 : test_acc= 0.9751 , train_acc= 0.99767274
11 : test_acc= 0.9736 , train_acc= 0.99603635
12 : test_acc= 0.9714 , train_acc= 0.99667275
13 : test_acc= 0.9751 , train_acc= 0.99792725
14 : test_acc= 0.9755 , train_acc= 0.9982182
15 : test_acc= 0.9767 , train_acc= 0.99734545
16 : test_acc= 0.9735 , train_acc= 0.9968727
17 : test_acc= 0.9775 , train_acc= 0.99774545
18 : test_acc= 0.9742 , train_acc= 0.99634546
19 : test_acc= 0.9743 , train_acc= 0.99790907
20 : test_acc= 0.9777 , train_acc= 0.9985273
21 : test_acc= 0.9767 , train_acc= 0.99790907
22 : test_acc= 0.976 , train_acc= 0.99734545
23 : test_acc= 0.9717 , train_acc= 0.99618185
24 : test_acc= 0.9765 , train_acc= 0.99814546
25 : test_acc= 0.9756 , train_acc= 0.9971455
26 : test_acc= 0.9776 , train_acc= 0.99912727
27 : test_acc= 0.9709 , train_acc= 0.9946909
28 : test_acc= 0.9771 , train_acc= 0.99832726
29 : test_acc= 0.9779 , train_acc= 0.998
30 : test_acc= 0.9755 , train_acc= 0.99863636
31 : test_acc= 0.9794 , train_acc= 0.9978909
32 : test_acc= 0.98 , train_acc= 0.99947274
33 : test_acc= 0.9799 , train_acc= 0.9994
34 : test_acc= 0.9787 , train_acc= 0.9984364
35 : test_acc= 0.9766 , train_acc= 0.99487275
36 : test_acc= 0.9813 , train_acc= 0.9994
37 : test_acc= 0.9789 , train_acc= 0.99765456
38 : test_acc= 0.9764 , train_acc= 0.998
39 : test_acc= 0.9784 , train_acc= 0.99889094
40 : test_acc= 0.981 , train_acc= 0.9997454
41 : test_acc= 0.981 , train_acc= 0.9998364
42 : test_acc= 0.9808 , train_acc= 0.9999818
43 : test_acc= 0.9821 , train_acc= 1.0
44 : test_acc= 0.9824 , train_acc= 1.0
45 : test_acc= 0.9824 , train_acc= 1.0
46 : test_acc= 0.9824 , train_acc= 1.0
47 : test_acc= 0.9823 , train_acc= 1.0
48 : test_acc= 0.9823 , train_acc= 1.0
49 : test_acc= 0.9827 , train_acc= 1.0
50 : test_acc= 0.9826 , train_acc= 1.0
51 : test_acc= 0.9825 , train_acc= 1.0
52 : test_acc= 0.9824 , train_acc= 1.0
53 : test_acc= 0.9823 , train_acc= 1.0
54 : test_acc= 0.9826 , train_acc= 1.0
55 : test_acc= 0.9829 , train_acc= 1.0
56 : test_acc= 0.9826 , train_acc= 1.0
57 : test_acc= 0.9826 , train_acc= 1.0
58 : test_acc= 0.9826 , train_acc= 1.0
59 : test_acc= 0.9828 , train_acc= 1.0
    
#AdamOptimizer, lr = 0.001 * (0.98 ** epoch)
0 : test_acc= 0.9443 , train_acc= 0.95616364
1 : test_acc= 0.9618 , train_acc= 0.97785455
2 : test_acc= 0.9628 , train_acc= 0.9816727
3 : test_acc= 0.9694 , train_acc= 0.99276364
4 : test_acc= 0.9657 , train_acc= 0.99070907
5 : test_acc= 0.9681 , train_acc= 0.9940182
6 : test_acc= 0.9675 , train_acc= 0.99358183
7 : test_acc= 0.9699 , train_acc= 0.9959273
8 : test_acc= 0.971 , train_acc= 0.9941091
9 : test_acc= 0.9667 , train_acc= 0.99307275
10 : test_acc= 0.9739 , train_acc= 0.9984364
11 : test_acc= 0.9761 , train_acc= 0.9997454
12 : test_acc= 0.9708 , train_acc= 0.9936182
13 : test_acc= 0.9701 , train_acc= 0.9942727
14 : test_acc= 0.9769 , train_acc= 0.9990909
15 : test_acc= 0.9766 , train_acc= 0.9988545
16 : test_acc= 0.9784 , train_acc= 0.9994364
17 : test_acc= 0.9736 , train_acc= 0.99834543
18 : test_acc= 0.9745 , train_acc= 0.99798185
19 : test_acc= 0.9768 , train_acc= 0.99881816
20 : test_acc= 0.9787 , train_acc= 0.99978185
21 : test_acc= 0.9809 , train_acc= 1.0
22 : test_acc= 0.974 , train_acc= 0.9962182
23 : test_acc= 0.9779 , train_acc= 0.99887276
24 : test_acc= 0.9787 , train_acc= 0.9998
25 : test_acc= 0.9798 , train_acc= 0.99996364
26 : test_acc= 0.9802 , train_acc= 1.0
27 : test_acc= 0.9808 , train_acc= 1.0
28 : test_acc= 0.9811 , train_acc= 1.0
29 : test_acc= 0.981 , train_acc= 1.0
30 : test_acc= 0.9814 , train_acc= 1.0
31 : test_acc= 0.9809 , train_acc= 1.0
32 : test_acc= 0.9811 , train_acc= 1.0
33 : test_acc= 0.9814 , train_acc= 1.0
34 : test_acc= 0.9815 , train_acc= 1.0
35 : test_acc= 0.9813 , train_acc= 1.0
36 : test_acc= 0.9812 , train_acc= 1.0
37 : test_acc= 0.981 , train_acc= 1.0
38 : test_acc= 0.9815 , train_acc= 1.0
39 : test_acc= 0.981 , train_acc= 1.0
40 : test_acc= 0.9813 , train_acc= 1.0
41 : test_acc= 0.9812 , train_acc= 1.0
42 : test_acc= 0.981 , train_acc= 1.0
43 : test_acc= 0.9816 , train_acc= 1.0
44 : test_acc= 0.9814 , train_acc= 1.0
45 : test_acc= 0.9813 , train_acc= 1.0
46 : test_acc= 0.9816 , train_acc= 1.0
47 : test_acc= 0.9813 , train_acc= 1.0
48 : test_acc= 0.9814 , train_acc= 1.0
49 : test_acc= 0.9817 , train_acc= 1.0
50 : test_acc= 0.9812 , train_acc= 1.0
51 : test_acc= 0.9815 , train_acc= 1.0
52 : test_acc= 0.981 , train_acc= 1.0
53 : test_acc= 0.9817 , train_acc= 1.0
54 : test_acc= 0.9815 , train_acc= 1.0
55 : test_acc= 0.9814 , train_acc= 1.0
56 : test_acc= 0.9811 , train_acc= 1.0
57 : test_acc= 0.9816 , train_acc= 1.0
58 : test_acc= 0.9817 , train_acc= 1.0
59 : test_acc= 0.9817 , train_acc= 1.0

"""