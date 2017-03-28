from tensorflow.examples.tutorials.mnist import input_data



def get_mnist_data():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
	trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
	trX = trX.reshape(-1, 28, 28)
	teX = teX.reshape(-1, 28, 28)
	return (trX, trY, teX, teY)