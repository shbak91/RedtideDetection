import tensorflow as tf
import pandas as pd
import numpy as np

DetectionDATE = '20140908'
# Load Data
D = np.loadtxt(DetectionDATE + '_nLw.csv',
               delimiter=',',
               skiprows=1)

# Input Data 표준화
Ref = np.loadtxt('MeanStd_2019_0622.csv', delimiter=',', skiprows=1)

location = D[:, 0:2]
D = D[:, 2:10]

for i in range(D.shape[0]):

    for j in range(D.shape[1]):

        D[i, j] = (D[i, j] - Ref[j, 0])/(Ref[j, 1])

D = np.c_[location, D]

np.savetxt(DetectionDATE + '_nLw_s.csv', D,
           delimiter=',',
           fmt='%2.5f',
           header='lat,lon,nlw1,nlw2,nlw3,nlw4,nlw5,nlw6,nlw7,nlw8',
           comments='')

D = pd.read_csv(DetectionDATE + '_nLw_s.csv')

x_data = D[['nlw1', 'nlw2', 'nlw3', 'nlw4', 'nlw5', 'nlw6', 'nlw7', 'nlw8']]

x_data2 = D[['lat', 'lon', 'nlw1', 'nlw2', 'nlw3', 'nlw4', 'nlw5', 'nlw6', 'nlw7', 'nlw8']]


n_node = 500
learning_rate = 1e-5
k_prob = 1.0

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

# Run
with tf.Session() as sess:
    saver.restore(sess, 'Model3 2019_0622/HAB_Detection_2019_0622')

    p = tf.argmax(output, axis=1)
    result = sess.run(p, feed_dict={X: x_data, keep_prob: k_prob})
    result = np.array(result, dtype=np.float32)

    h = sess.run(output, feed_dict={X: x_data, keep_prob: k_prob})

r = np.c_[x_data2, h, result]

r = r[r[:, 10] > 0.5, :]

np.savetxt('Detection_Result_' + DetectionDATE + '.csv', r,
           header='lat,lon,nlw1,nlw2,nlw3,nlw4,nlw5,nlw6,nlw7,nlw8,P_Class1,P_Class2,P_Class3,PredictionResult',
           delimiter=',',
           comments='',
           fmt='%2.5f')


