import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

def _weight(shape_, name=''):
	#print(name)
	return tf.get_variable(name, shape=shape_, initializer=tf.random_normal_initializer(0.01))

gen_name = 'generator'
dis_name = 'discriminator'

# -------------------
# | discriminator		|
# -------------------

a = gen_name + "1"
print(a)

num_inputs_d = 784

def log(x):
	return tf.log(tf.maximum(x, 1e-6))

def discriminator(input_):
	layer_1_size_d = int(num_inputs_d/10)
	W_1_d = _weight([num_inputs_d, layer_1_size_d], name=dis_name + "1")
	b_1_d = _weight([layer_1_size_d,], name=dis_name + "2")

	layer_1_d = tf.nn.tanh(tf.matmul(input_, W_1_d) + b_1_d)

	layer_2_size_d = 1
	W_2_d = _weight([layer_1_size_d, layer_2_size_d], name=dis_name + "3")
	b_2_d = _weight([layer_2_size_d,], name=dis_name + "4")

	layer_2_raw_d = tf.matmul(layer_1_d, W_2_d) + b_2_d

	output_d = tf.sigmoid(layer_2_raw_d)

	return output_d


# -------------------
# | generator		|
# -------------------

num_inputs_g = 5
generator_loss = 0
discriminator_loss = 1

def generator(input_):
	# data generation doesn't require knowledge of the discriminator
	layer_1_size_g = num_inputs_g*80
	W_1_g = _weight([num_inputs_g, layer_1_size_g], name=gen_name + "1")
	b_1_g = _weight([layer_1_size_g,], name=gen_name + "2")

	layer_1_g = tf.nn.tanh(tf.matmul(input_, W_1_g) + b_1_g)

	#layer_2_size_g = 3*layer_1_size_g
	layer_2_size_g = num_inputs_d
	W_2_g = _weight([layer_1_size_g, layer_2_size_g], name=gen_name + "3")
	b_2_g = _weight([layer_2_size_g,], name=gen_name + "4")
	'''
	layer_2_g = tf.nn.tanh(tf.matmul(layer_1_g, W_2_g) + b_2_g)

	layer_3_size_g = num_inputs_d
	W_3_g = _weight([layer_2_size_g, layer_3_size_g], name=gen_name + "5")
	b_3_g = _weight([layer_3_size_g,], name=gen_name + "6")
	layer_3_raw_g = tf.matmul(layer_2_g, W_3_g) + b_3_g

	output_g = tf.nn.sigmoid(layer_3_raw_g)
	'''
	layer_2_raw_g = tf.matmul(layer_1_g, W_2_g) + b_2_g
	output_g = tf.nn.sigmoid(layer_2_raw_g)
	return output_g

# label placeholder (loss function is just cross-entropy)
#y_d = tf.placeholder(shape=[None, 2], dtype=tf.float32)
with tf.variable_scope('G'):
	# latent variable placeholder
	X_g = tf.placeholder(shape=[None, None], dtype=tf.float32)
	G = generator(X_g)

with tf.variable_scope('D'):
	# input data placeholder (data from generator and dataset)
	X_d = tf.placeholder(shape=[None, num_inputs_d], dtype=tf.float32)
	D1 = discriminator(X_d)

with tf.variable_scope('D', reuse=True):
	D2 = discriminator(G)

vars_d = [v for v in tf.trainable_variables() if dis_name in v.name]
vars_g = [v for v in tf.trainable_variables() if gen_name in v.name]
print('generator vars')
for v in vars_g:
	print(v.name)
print('discriminator vars')
for v in vars_d:
	print(v.name)

#some other bullshit for calculating a loss for the discriminator
loss_d = tf.reduce_mean(-log(D1) - log(1. - D2))
loss_g = tf.reduce_mean(-log(D2))

opt_d = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_d, var_list=vars_d)
opt_g = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_g, var_list=vars_g)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

batch_size = 128
fd = open('train_errord.dat', 'w+')
fg = open('train_errorg.dat', 'w+')

for i in range(50000):
	# train the discriminator
	'''
	if discriminator_loss > 0.01:
		for j in range(5):
			train_batch_d = mnist.train.next_batch(batch_size)[0]
			train_batch_g = np.random.rand(batch_size, num_inputs_g)*10 - 5
			discriminator_loss, _ = sess.run([loss_d, opt_d], {X_d:train_batch_d, X_g:train_batch_g})
	else:
		train_batch_d = mnist.train.next_batch(batch_size)[0]
		train_batch_g = np.random.rand(batch_size, num_inputs_g)*10 - 5
		discriminator_loss, _ = sess.run([loss_d, opt_d], {X_d:train_batch_d, X_g:train_batch_g})
	'''
	train_batch_d = mnist.train.next_batch(batch_size)[0]
	train_batch_g = np.random.rand(batch_size, num_inputs_g)*10 - 5
	discriminator_loss, _ = sess.run([loss_d, opt_d], {X_d:train_batch_d, X_g:train_batch_g})

	# train the generator
	train_batch_g = np.random.rand(batch_size, num_inputs_g)*10 - 5
	generator_loss, _ = sess.run([loss_g, opt_g], {X_g:train_batch_g})

	if generator_loss > 13.8:
		break
	'''
	if generator_loss > 10 and generator_loss < 13.8 and i > 50000:
		for j in range(3):
			train_batch_g = np.random.rand(batch_size, num_inputs_g)*10 - 5
			generator_loss, _ = sess.run([loss_g, opt_g], {X_g:train_batch_g})
	else:
		train_batch_g = np.random.rand(batch_size, num_inputs_g)*10 - 5
		generator_loss, _ = sess.run([loss_g, opt_g], {X_g:train_batch_g})
	'''
	if i % 100 == 0:
		print('generator loss:\t\t', generator_loss)
		print('discriminator loss:\t', discriminator_loss)
		print('step:', i)
		fd.write(str(i) + "," + str(discriminator_loss) + "\n")
		fg.write(str(i) + "," + str(generator_loss) + "\n")
		
generated_images = sess.run(G, {X_g: train_batch_g})

for j in range(len(generated_images)):
	plt.figure()
	plt.imshow(generated_images[j].reshape([28, 28]), cmap=plt.get_cmap('gray_r'))

plt.show()
