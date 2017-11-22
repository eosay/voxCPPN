import sys
import os
import math
import argparse
import tensorflow as tf 
import numpy as np 
import tools

'''Generate, visualize, and save voxel shapes for nn training'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=32,
                        help='Voxel dimensions cubed')
    parser.add_argument('--amount', type=int, default=3, 
                        help='The number of voxel shapes to generate')
    parser.add_argument('--seed', type=int, default=None,
                        help='tensorflow weight init seed')
    args = parser.parse_args()
    
    for file in os.listdir(tools.get_path('shapes')):
        os.remove(tools.get_path('shapes', file))
    
    make_shape(args)

def gaussian(x):
    sigma = 1
    mean = 0
    return (tf.exp((-(x-mean)**2) / (2 * sigma ** 2))) / (sigma * tf.sqrt(2 * math.pi))

def make_shape(args):
    tf.set_random_seed(args.seed)
    tools.gen_coord_datasets()

    # get dataset
    xdata = tools.load_coord_dataset(args.size)
    x = tf.placeholder(tf.float32, [None, 4])

    for i in range(args.amount):
        # cppn network graph
        wrange = 2
        w1 = tf.Variable(tf.random_uniform([4, 30], -wrange, wrange))
        w2 = tf.Variable(tf.random_uniform([30, 30], -wrange, wrange))
        w3 = tf.Variable(tf.random_uniform([30, 30], -wrange, wrange))
        w4 = tf.Variable(tf.random_uniform([30, 30], -wrange, wrange))
        w5 = tf.Variable(tf.random_uniform([30, 1], -wrange, wrange))

        layer1 = gaussian(tf.matmul(x, w1))
        layer2 = tf.sin(tf.matmul(layer1, w2))
        layer3 = tf.nn.sigmoid(tf.matmul(layer2, w3))
        layer4 = tf.tanh(tf.matmul(layer3, w4))
        layer5 = tf.sigmoid(tf.matmul(layer4, w5))

        # run cppn network for number of shapes
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            outputs = sess.run(layer5, feed_dict={x: xdata})

            voxels = np.rint(outputs).reshape(args.size, args.size, args.size)
            voxels[voxels < 0] = 0
            # save to shapes dir
            path = tools.get_path('shapes', 'shape{}'.format(i))
            np.save(path, voxels)
            tools.render_voxels(voxels)

if __name__ == '__main__':
    main()
