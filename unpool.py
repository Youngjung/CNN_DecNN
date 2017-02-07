import tensorflow as tf
import numpy as np

def unravel_argmax(argmax, shape):
	output_list = []
	output_list.append(argmax // (shape[2] * shape[3]))
	output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
	return tf.pack(output_list)

def unpool_layer_with_argmax_batch(x, kernel, argmax):
	'''
	original: https://github.com/fabianbormann/Tensorflow-DeconvNet-Segmentation/blob/master/DeconvNet.py
	Args:
		x: 4D tensor of shape [batch_size x height x width x channels]
		argmax: A Tensor of type Targmax. 4-D. The flattened indices of the max
		values chosen for each output.
	Return:
		4D output tensor of shape [batch_size x kernel*height x kernel*width x channels]
	'''
	x_shape = tf.shape(x)
	#x_shape = x.get_shape()
	out_shape = [x_shape[0], x_shape[1]*kernel, x_shape[2]*kernel, x_shape[3]]

	batch_size = out_shape[0]
	height = out_shape[1]
	width = out_shape[2]
	channels = out_shape[3]

	argmax_shape = tf.to_int64([batch_size, height, width, channels])
	argmax = unravel_argmax(argmax, argmax_shape)

	t1 = tf.to_int64(tf.range(channels))
	t1 = tf.tile(t1, [batch_size*(width//kernel)*(height//kernel)])
	t1 = tf.reshape(t1, [-1, channels])
	t1 = tf.transpose(t1, perm=[1, 0])
	t1 = tf.reshape(t1, [channels, batch_size, height//kernel, width//kernel, 1])
	t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

	t2 = tf.to_int64(tf.range(batch_size))
	t2 = tf.tile(t2, [channels*(width//kernel)*(height//kernel)])
	t2 = tf.reshape(t2, [-1, batch_size])
	t2 = tf.transpose(t2, perm=[1, 0])
	t2 = tf.reshape(t2, [batch_size, channels, height//kernel, width//kernel, 1])

	t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

	t = tf.concat(4, [t2, t3, t1])
	indices = tf.reshape(t, [(height//kernel)*(width//kernel)*channels*batch_size, 4])

	x1 = tf.transpose(x, perm=[0, 3, 1, 2])
	values = tf.reshape(x1, [-1])

	delta = tf.SparseTensor(indices, values, tf.to_int64(out_shape))
	reordered = tf.sparse_reorder(delta)
	return tf.sparse_tensor_to_dense( reordered )

def unpool_layer_fixed(x, shape, unpool_mat=None):
	"""
	Unpool the input with a fixed matrix to perform kronecker product with.
	original: https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/models/pool.py
	Args:
		x (tf.Tensor): a NHWC tensor
		shape: int or (h, w) tuple
		unpool_mat: a tf.Tensor or np.ndarray 2D matrix with size=shape.
			If is None, will use a matrix with 1 at top-left corner.
	Returns:
		tf.Tensor: a NHWC tensor.
	"""
	shape = shape2d(shape)

	# a faster implementation for this special case
	if shape[0] == 2 and shape[1] == 2 and unpool_mat is None:
		return UnPooling2x2ZeroFilled(x)

	input_shape = x.get_shape().as_list()
	if unpool_mat is None:
		mat = np.zeros(shape, dtype='float32')
		mat[0][0] = 1
		unpool_mat = tf.constant(mat, name='unpool_mat')
	elif isinstance(unpool_mat, np.ndarray):
		unpool_mat = tf.constant(unpool_mat, name='unpool_mat')
	assert unpool_mat.get_shape().as_list() == list(shape)

	# perform a tensor-matrix kronecker product
	fx = flatten(tf.transpose(x, [0, 3, 1, 2]))
	fx = tf.expand_dims(fx, -1)	   # (bchw)x1
	mat = tf.expand_dims(flatten(unpool_mat), 0)  # 1x(shxsw)
	prod = tf.matmul(fx, mat)  # (bchw) x(shxsw)
	prod = tf.reshape(prod, tf.stack(
		[-1, input_shape[3], input_shape[1], input_shape[2], shape[0], shape[1]]))
	prod = tf.transpose(prod, [0, 2, 4, 3, 5, 1])
	prod = tf.reshape(prod, tf.stack(
		[-1, input_shape[1] * shape[0], input_shape[2] * shape[1], input_shape[3]]))
	return prod

def shape2d(a):
	"""
	Ensure a 2D shape.
	Args:
		a: a int or tuple/list of length 2
	Returns:
		list: of length 2. if ``a`` is a int, return ``[a, a]``.
	"""
	if type(a) == int:
		return [a, a]
	if isinstance(a, (list, tuple)):
		assert len(a) == 2
		return list(a)
	raise RuntimeError("Illegal shape: {}".format(a))

def flatten(x):
	"""
	Flatten the tensor.
	"""
	return tf.reshape(x, [-1])


def UnPooling2x2ZeroFilled(x):
	# https://github.com/tensorflow/tensorflow/issues/2169
	out = tf.concat([x, tf.zeros_like(x)], 3)
	out = tf.concat([out, tf.zeros_like(out)], 2)

	sh = x.get_shape().as_list()
	if None not in sh[1:]:
		out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
		return tf.reshape(out, out_size)
	else:
		shv = tf.shape(x)
		ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]))
		ret.set_shape([None, None, None, sh[3]])
		return ret
