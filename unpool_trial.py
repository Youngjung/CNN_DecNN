# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#		 http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System				| Step Time (sec/batch)	|		 Accuracy
------------------------------------------------------------------
1 Tesla K20m	| 0.35-0.60							| ~86% at 60K steps	(5 hours)
1 Tesla K40m	| 0.25-0.35							| ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf
import numpy as np

import pdb

slim = tf.contrib.slim

def main(argv=None):	# pylint: disable=unused-argument
    x_data = np.arange(36, dtype=np.float32).reshape(1,6,6,1)
    print(x_data)
    pooled = slim.max_pool2d( x_data, [3,3], stride=3 )
    #pooled, argmax = tf.nn.max_pool_with_argmax( x_data, [1,2,2,1], [1,1,1,1], "VALID" )
    argmax = np.ndarray( shape=(1,1,1,1), dtype=np.int64 )
    argmax.fill(0)
    argmax = tf.convert_to_tensor( argmax )
    unpooled = unpool_layer_with_argmax_batch( pooled, 3, argmax )

    sess = tf.Session()
    print( sess.run( pooled ) )
    print( sess.run( argmax ) )
    print( sess.run( unpooled ) )

def unravel_argmax(argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    return tf.pack(output_list)

def unpool_layer_without_argmax_batch(x, kernel):
    x_shape = tf.shape(x)
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
    return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))


def unpool_layer_with_argmax_batch(x, kernel, argmax):
    '''
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

    pdb.set_trace()
    delta = tf.SparseTensor(indices, values, tf.to_int64(out_shape))
    return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))


if __name__ == '__main__':
	tf.app.run()
