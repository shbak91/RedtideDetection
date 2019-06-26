# coding: utf-8

import tensorflow as tf
import numpy as np
import pandas as pd
import datetime

tf.set_random_seed(608)

Data_Train_raw = pd.read_csv('Dataset_Train_2019_0622_s.csv')
Data_Test_raw = pd.read_csv('Dataset_Test_2019_0622_s.csv')

# Create Train & Test Dataset
name_col = Data_Train_raw.columns[0:8]

Data_Train = Data_Train_raw[name_col]
Data_Train_Class = Data_Train_raw[["Class1", "Class2", "Class3"]]

Data_Test = Data_Test_raw[name_col]
Data_Test_Class = Data_Test_raw[["Class1", "Class2", "Class3"]]

x_data = np.array(Data_Train.values, dtype=np.float32)
y_data = np.array(Data_Train_Class.values, dtype=np.float32)

x_data_test = np.array(Data_Test.values, dtype=np.float32)
y_data_test = np.array(Data_Test_Class.values, dtype=np.float32)

# Create Deep Neural Network Model(Multi-layer Neural Network)
n_node = 500
learning_rate = 1e-5
k_prob = 0.5
epoch = 10000

# Input layer
X = tf.placeholder(tf.float32, shape=[None, 8], name='X')
Y = tf.placeholder(tf.float32, shape=[None, 3], name='Y')

keep_prob = tf.placeholder(tf.float32)

# Hidden Layer1
W1 = tf.Variable(tf.random_normal([8, n_node], stddev=0.01), name='weight1')
b1 = tf.Variable(tf.random_normal([n_node], stddev=0.01), name='bias1')
layer1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1), name='layer1')
layer1 = tf.nn.dropout(layer1, keep_prob)

# Hidden Layer2
W2 = tf.Variable(tf.random_normal([n_node, n_node], stddev=0.01), name='weight2')
b2 = tf.Variable(tf.random_normal([n_node], stddev=0.01), name='bias2')
layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W2), b2), name='layer2')
layer2 = tf.nn.dropout(layer2, keep_prob)

# Hidden Layer3
W3 = tf.Variable(tf.random_normal([n_node, n_node], stddev=0.01), name='weight3')
b3 = tf.Variable(tf.random_normal([n_node], stddev=0.01), name='bias3')
layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, W3), b3), name='layer3')
layer3 = tf.nn.dropout(layer3, keep_prob)

# Hidden Layer4
W4 = tf.Variable(tf.random_normal([n_node, n_node], stddev=0.01), name='weight4')
b4 = tf.Variable(tf.random_normal([n_node], stddev=0.01), name='bias4')
layer4 = tf.nn.relu(tf.add(tf.matmul(layer3, W4), b4), name='layer4')
layer4 = tf.nn.dropout(layer4, keep_prob)

# Output layer
W5 = tf.Variable(tf.random_normal([n_node, 3], stddev=0.01), name='weight5')
b5 = tf.Variable(tf.random_normal([3], stddev=0.01), name='bias5')
hypothesis = tf.add(tf.matmul(layer4, W5), b5)
output = tf.nn.softmax(hypothesis, name='output_layer')

# tf.train.Saver
saver = tf.train.Saver()

# Cost
cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=Y)
cost = tf.reduce_mean(cost_i)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Predict
predict = tf.cast(output > 0.5, dtype=np.float32)
# accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, Y), dtype=np.float32))
correct_prediction = tf.equal(tf.argmax(Y, axis=1), tf.argmax(output, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Start Training & Monitoring
print("Start Training!")

Time_Start = datetime.datetime.now()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    m = np.zeros((epoch, 5))
    # col 1 : epoch
    # col 2 : cost_train
    # col 3 : cost_test
    # col 4 : accuracy_train
    # col 5: accuracy_test
    for step in range(epoch):

        sess.run(train, feed_dict={X: x_data, Y: y_data, keep_prob: k_prob})

        cost_train = sess.run(cost, feed_dict={X: x_data, Y: y_data, keep_prob: k_prob})
        cost_test = sess.run(cost, feed_dict={X: x_data_test, Y: y_data_test, keep_prob: 1.0})

        accuracy_train = sess.run(accuracy, feed_dict={X: x_data, Y: y_data, keep_prob: k_prob})
        accuracy_test = sess.run(accuracy, feed_dict={X: x_data_test, Y: y_data_test, keep_prob: 1.0})

        m[step, 0] = step
        m[step, 1] = cost_train
        m[step, 2] = cost_test
        m[step, 3] = accuracy_train
        m[step, 4] = accuracy_test

        if step % 100 == 0:
            print("Epoch : ", step)
            print("Train Data Cost : ", cost_train)
            print("Test Data Cost : ", cost_test)

            # Accuracy Report
            print("Accuracy(Train) : ", accuracy_train)
            print("Accuracy(Test) : ", accuracy_test)
            print("============================================", "\n")

    print("Stop Epoch : ", step)
    print("Stop Train Data Cost : ", cost_train)
    print("Stop Test Data Cost : ", cost_test)
    print("Stop Train Data Accuracy : ", accuracy_train)
    print("Stop Test Data Accuracy : ", accuracy_test)

    saver.save(sess, "./HAB_Detection_2019_0622")

Time_Finish = datetime.datetime.now()

Duration = Time_Finish - Time_Start

print(Duration.seconds)

np.savetxt('Log_2019_0622.csv', m, delimiter=',', fmt='%10.4f', comments='',
           header='epoch,cost_train,cost_test,accu_train,accu_test')
