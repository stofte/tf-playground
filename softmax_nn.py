from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


mnist_path = "/mnt/c/src/tf-playground/mnist"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def main(_):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    X = tf.placeholder(tf.float32, [None, 784], 'X')
    W = tf.Variable(tf.zeros([784, 10]), 'W')
    b = tf.Variable(tf.zeros([10]), 'b')
    Y = tf.nn.softmax(tf.matmul(X, W) + b)

    Y_ = tf.placeholder(tf.float32, [10, None], 'Y_')
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))
    train_step = tf.train.GradientDescentOptimizer(0.003).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer().run()

    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={X: batch_xs, Y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=mnist_path, help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)