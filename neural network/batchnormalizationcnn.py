
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
nodes=512

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


FLAGS = None
############################
from tensorflow.python.framework import ops
import numbers
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import math_ops
############################

	
##############################
##############################
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name='W')

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name='B')

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def bat_norm(layer, axis, output, epsilon):
  fc_mean, fc_var = tf.nn.moments(
    layer,
    axes=axis  # the dimension you wanna normalize, here [0] for batch
    # for image, you wanna do [0,1,2] for [batch, height, with] but not change channel
  )
  scale = tf.Variable(tf.ones([output]))
  shift = tf.Variable(tf.zeros([output]))
  layer = tf.nn.batch_normalization(layer, fc_mean, fc_var, shift, scale, epsilon)
  # similar with this two steps:
  # h_fc1 =(h_fc1-fc_mean)/tf.sqrt(fc_var+0.001)
  # h_fc1=h_fc1*scale+shift
  return layer

#/Users/cw13f/Desktop/neuron_network/GPUtest
def main(_):
  
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784], name='x')
  #W = tf.Variable(tf.zeros([784, 10]))
  #b = tf.Variable(tf.zeros([10]))
  #y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10], name='lables')
  keep_prob = tf.placeholder(tf.float32)
  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.

  sess = tf.InteractiveSession(config=config)

  
  sess.run(tf.global_variables_initializer())
  
  x_image = tf.reshape(x, [-1,28,28,1])
  x_image= bat_norm(x_image, [0, 1, 2], 1, 0.001)
  with tf.name_scope('conv'):
    W_conv1 = weight_variable([4, 4, 1, 16])
    b_conv1 = bias_variable([16])    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_pool1=bat_norm(h_pool1, [0,1,2], 16, 0.001)
    tf.summary.histogram('weights',W_conv1)
    tf.summary.histogram('biases',b_conv1)
    tf.summary.histogram('activation', h_conv1)
	
  with tf.name_scope('conv'):
    W_conv2 = weight_variable([4, 4, 16, 32])
    b_conv2 = bias_variable([32])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2=bat_norm(h_pool2, [0,1,2], 32, 0.001)
    tf.summary.histogram('weights',W_conv2)
    tf.summary.histogram('biases',b_conv2)
    tf.summary.histogram('activation', h_conv2)
	
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*32])

  with tf.name_scope('fc'):
    W_fc1 = weight_variable([7 * 7 * 32, nodes])
    b_fc1 = bias_variable([nodes])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1=bat_norm(h_fc1, [0], nodes, 0.001)
    h_fc1_drop = dropoutr(h_fc1, keep_prob)	
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    tf.summary.histogram('weights',W_fc1)
    tf.summary.histogram('biases',b_fc1)
    tf.summary.histogram('activation', h_fc1)
###
#  W_fc11 = weight_variable([1024, 1024])
#  b_fc11 = bias_variable([1024])
  
#  h_fc11 = tf.nn.tanh(tf.matmul(h_fc1_drop, W_fc11) + b_fc11)
#  h_fc11_drop =tf.nn.dropout(h_fc11, keep_prob)
###
  with tf.name_scope('fc'):
    W_fc2 = weight_variable([nodes, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    y_conv = bat_norm(y_conv, [0], 10, 0.001)
    tf.summary.histogram('weights',W_fc2)
    tf.summary.histogram('biases',b_fc2)
    tf.summary.histogram('activation', y_conv)
	
  with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	  
  with tf.name_scope('training'):  
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	
  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	
  sess.run(tf.global_variables_initializer())

  ###########
  tf.summary.scalar('cross_entropy', cross_entropy)
  tf.summary.scalar('accuracy', accuracy)
  tf.summary.image('input', x_image, 20)
  merged_summary = tf.summary.merge_all()
  writer = tf.summary.FileWriter("/Users/cw13f/Desktop/neuronnetwork/GPUtest")
  writer.add_graph(sess.graph)
  # Train
  
  for i in range(200000):
    batch = mnist.train.next_batch(100)
    if i%100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
      print("step %d, training accuracy %g"%(i, train_accuracy))
      batch2 = mnist.test.next_batch(10000)
      acc=accuracy.eval(feed_dict={x: batch2[0], y_: batch2[1], keep_prob: 1.0})
      print("test accuracy %g"%acc)
    if i%100 ==0:
      s=sess.run(merged_summary, feed_dict={x:batch[0], y_:batch[1], keep_prob: 1.0})
      writer.add_summary(s,i)
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  #test
  acc=0;
  for i in range(100):
    batch = mnist.test.next_batch(100)
    acc=acc+accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
  acc=acc/100.0
  print("test accuracy %g"%acc)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
