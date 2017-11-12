import tensorflow as tf
import numpy as np

W=tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32, name='weight') #everything must be the same
b=tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32, name='biases')

# not need init step

saver= tf.train.Saver()
with tf.Session() as sess:
	saver.restore(sess,'/Users/cw13f/Desktop/nntest/save_net.ckpt')
	print('weights: ', sess.run(W))
	print('biases: ', sess.run(b))