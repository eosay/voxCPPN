import os
import sys
import argparse
import time
import tensorflow as tf 
import numpy as np 
import tools

'''run the trained model on a specified latent vector to output
    the associated voxel shape'''

def main(args):

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    shape_amount = len(os.listdir(tools.get_path('shapes')))
    vol = args.size ** 3
    
    # data
    latent_vec = np.random.uniform(size=(shape_amount, 1))    
    coord_vec = tools.load_coord_dataset(args.size)
    xdata = np.append(coord_vec, latent_vec[args.shape] * np.ones((vol, 1)), axis=1)

    print('latent vector input\n', latent_vec)
    
    # load the model
    params = np.load('model.npy')

    # model graph
    x = tf.placeholder(tf.float32, [None, 5])
    W = params[0]
    B = params[1]
    w1 = tf.constant(W[0])
    w2 = tf.constant(W[1])
    w3 = tf.constant(W[2])
    b1 = tf.constant(B[0])
    b2 = tf.constant(B[1])

    hidden1 = tf.tanh(tf.matmul(x, w1) + b1)
    hidden2 = tf.tanh(tf.matmul(hidden1, w2) + b2)    
    output = tf.nn.sigmoid(tf.matmul(hidden2, w3))

    with tf.Session() as sess:
        pred = sess.run(output, feed_dict={x: xdata})

    # render in browser
    voxels = np.rint(pred).reshape(args.size, args.size, args.size)
    tools.render_voxels(voxels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape', type=int, default=0,
                        help='number of the shape to output')
    parser.add_argument('--size', type=int, default=32, 
                        help='Voxel dimensions cubed, can be different size for train vs latent op')
    parser.add_argument('--seed', type=int, default=256,
                        help='latent vector seed')
    start = time.time()
    main(parser.parse_args())
    print('time (min)', (time.time() - start) / 60)