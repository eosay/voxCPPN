import os
import sys
import argparse
import time
import tensorflow as tf 
import numpy as np 
import tools

def main(args):
    '''Simple feed forward neural network for voxel shape encoding and generative 
       modeling with latent vectors. Uses Tensorflow and can easily run on a CPU.

       usage: net.py [-h] [--op OP] [--size SIZE] [--seed SEED]

        optional arguments:
        -h, --help   show this help message and exit
        --op OP      operation to complete: train | latent
        --size SIZE  Voxel dimensions cubed, can be different size for train vs
                    latent op
        --seed SEED  tensorflow weight init seed
       
       The network is very small which allows for fast training (<10 min on CPU) and
       overfitting (something that is actually wanted for this net!) However it should 
       be noted that there is a balance with network size and the net's ability to 
       encode complex shapes. This is purely a simple interesting project for generative modeling
       that can be trained and run on a laptop. Inspired by Compositional Pattern Producing Networks (CPPN).
       
       For training: train at a low resolution using the SIZE CLI argument, 32 works well
       For running the net/latent space visualizations: SIZE 32-128 are good visualization 
       resoultions. Be careful with this becuase you can quickly run out of memory.

       Author: Dominic Cascino
       Date: Oct 2017'''
    # hyper paramters
    batch_size = 1000
    lr = 0.001
    iters = 2000
    samples = 100
    size = args.size
    vol = size ** 3
    seed = args.seed
    np.random.seed(seed)
    tf.set_random_seed(seed)
    shape_amount = len(os.listdir(tools.get_path('shapes')))
    save_path = 'model'
    # dataset
    coord_vec = tools.load_coord_dataset(size)
    # scalar latent vector for each shape, latent space is only 1d
    latent_vec = np.random.uniform(size=(shape_amount, 1))    
    xdata = []
    ydata = []

    for i in range(shape_amount):
        latent = np.expand_dims(latent_vec[i], axis=0).repeat(vol, axis=0)
        shape_path = tools.get_path('shapes', 'shape{}.npy'.format(i))
        xdata.append(np.append(coord_vec, latent, axis=1))
        ydata.append(np.expand_dims(np.load(shape_path).flatten(), axis=1))

    # small feed forward neural network graph
    # a small network seems to work better, the less paramters, 
    # the easier it overfits to the shapes (which is what we want!)
    # but a larger network does allow for more complexity to be encoded
    x = tf.placeholder(tf.float32, [None, 5])
    y = tf.placeholder(tf.float32, [None, 1])

    w1 = tf.Variable(tf.random_uniform([5, 10]))
    w2 = tf.Variable(tf.random_uniform([10, 10]))   
    w3 = tf.Variable(tf.random_uniform([10, 1]))

    b1 = tf.Variable(tf.random_uniform([10]))
    b2 = tf.Variable(tf.random_uniform([10]))

    hidden1 = tf.tanh(tf.matmul(x, w1) + b1)
    hidden2 = tf.tanh(tf.matmul(hidden1, w2) + b2)    
    output = tf.nn.sigmoid(tf.matmul(hidden2, w3))

    # loss and optim
    loss = tf.reduce_sum(tf.square(y - output))
    train = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    init = tf.global_variables_initializer()

    if args.op == 'train':
        # training loop
        with tf.Session() as sess:
            sess.run(init)
            for i in range(iters):
                for j in range(0, vol, batch_size):
                    # train on each shape one after the other
                    for k in range(shape_amount):
                        sess.run(train, feed_dict={x: xdata[k][j:batch_size+j], y: ydata[k][j:batch_size+j]})

                if i % samples == 0:
                    # loss sampling
                    print('epoch', i)
                    for shp in range(shape_amount):
                        e = sess.run(loss, feed_dict={x: xdata[shp], y: ydata[shp]})
                        print('loss{} {:0.3f}    '.format(shp, e), end='', flush=True)
                    print('\n')

            # save model here
            for data in xdata:
                out = sess.run(output, feed_dict={x: data})
                tools.render_voxels(np.rint(out).reshape(size, size, size))
            
            # save just the weights
            W = sess.run([w1, w2, w3])
            B = sess.run([b1, b2])
            np.save(save_path, [W, B])

    elif args.op == 'latent':
        # dataset
        # size can be greater or smaller than what the model has been trained on,
        # changing the size scales the output shape since the difference between
        # the minmax normalized datapoints increases or decreases
        # this means that the model can train on a low resolution and 
        # output an arbitrary resolution shape after training
        coord_vec = tools.load_coord_dataset(args.size)
        vol = args.size ** 3
        print('latent vector inputs\n', latent_vec)

        lmin, lmax = np.amin(latent_vec), np.amax(latent_vec)
        # steps: how many shapes to build and render in animation
        # resolution 32 -> 20ish steps
        # resolution 64 -> 20ish steps
        # resolution 128 -> 5 steps, dont go above 10
        # resolution 256 -> you will probably run out of memory!
        steps = 5
        if args.size < 128:
            steps = 20
        shifts = np.linspace(lmin, lmax, steps)
        data_list = []
        base = np.ones((vol, 1))
        # traverse the latent space between the latent vector inputs
        for i in range(steps):
            latent_shift = shifts[i] * base
            # np.append is a memory issue for large sizes 128-256,
            # since it makes copies of the original array
            data_list.append(np.append(coord_vec, latent_shift, axis=1))

        # restore model, just uses numpy saving and not tf saver
        params = np.load(save_path + '.npy')

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

        results = []
        with tf.Session() as sess:
            for data in data_list:
                results.append(np.rint(sess.run(output, feed_dict={x: data}).reshape(size, size, size)))

        # render the latent space traversal as animation in browser
        tools.render_voxel_ani(results)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', type=str, default='train',
                        help='operation to complete: train | latent')
    parser.add_argument('--size', type=int, default=32, 
                        help='Voxel dimensions cubed, can be different size for train vs latent op')
    parser.add_argument('--seed', type=int, default=256,
                        help='tensorflow weight init seed')
    start = time.time()
    main(parser.parse_args())
    print('time (min)', (time.time() - start) / 60)