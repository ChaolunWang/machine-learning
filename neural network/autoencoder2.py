import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/Users/cw13f/Desktop/nntest/data', one_hot=False)

#Visualize decoder setting
#Parameters

learning_rate = 0.001
training_epochs = 200
batch_size = 256
display_step = 1
examples_to_show = 10

# Network Parameters
n_input = 784

# tf Graph input(only pictures)
X=tf.placeholder('float', [None, n_input])

# hidden layer settings
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2

weights = {
	'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1])),
	'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
	'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
	'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
	
	'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3])),
	'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2])),	
	'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1])),
	'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input])),
}

biases = {
	'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
	'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
	
	'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
	'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
	'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
	'decoder_b4': tf.Variable(tf.random_normal([n_input])),	
}


# Building the encoder
def encoder(x):
	layer_1 = tf.nn.sigmoid(
		tf.add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
	# Encoder Hidden layer with sigmoid activation #2
	layer_2 = tf.nn.sigmoid(
		tf.add(tf.matmul(layer_1,weights['encoder_h2']),biases['encoder_b2']))	
	layer_3 = tf.nn.sigmoid(
		tf.add(tf.matmul(layer_2,weights['encoder_h3']),biases['encoder_b3']))
	# Encoder Hidden layer with sigmoid activation #2
	layer_4 =tf.add(tf.matmul(layer_3,weights['encoder_h4']),biases['encoder_b4'])
	return layer_4
	
# Building the decoder
def decoder(x):
	layer_1 = tf.nn.sigmoid(
		tf.add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))
	# Decoder Hidden layer with sigmoid activation #2
	layer_2 = tf.nn.sigmoid(
		tf.add(tf.matmul(layer_1,weights['decoder_h2']),biases['decoder_b2']))
	layer_3 = tf.nn.sigmoid(
		tf.add(tf.matmul(layer_2,weights['decoder_h3']),biases['decoder_b3']))
	# Decoder Hidden layer with sigmoid activation #2
	layer_4 = tf.nn.sigmoid(
		tf.add(tf.matmul(layer_3,weights['decoder_h4']),biases['decoder_b4']))
	return layer_4

# Construct model	
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf. reduce_mean(tf.pow(y_true-y_pred,2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

#Launch the graph
with tf.Session() as sess:
	sess.run(init)
	total_batch = int(mnist.train.num_examples/batch_size)
	#Training cycle
	for epoch in range(training_epochs):
		#Loop over all batches
		for i in range(total_batch):
			batch_xs, batch_ys = mnist.train.next_batch(batch_size)
			#Run optimization op and cost op
			_, c=sess.run([optimizer,cost], feed_dict={X:batch_xs})
		#Display logs per epoch step
		if epoch%display_step == 0:
			print('Epoch:','%04d'%(epoch+1),
				'cost=','{:.9f}'.format(c))
				
	print('Optimization Finished')
	
	## Visualize the encoded result
	encoder_result=sess.run(encoder_op, feed_dict={X: mnist.test.images})
	plt.scatter(encoder_result[:,0], encoder_result[:,1], c=mnist.test.labels)
	plt.show()
	