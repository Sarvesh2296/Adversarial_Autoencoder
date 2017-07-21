from keras.preprocessing.image import array_to_img
from PIL import Image
import scipy.misc
import utils2 as utils
import collections

batch_size   = 64
img_size     = 128
channels     = 3
input_dim    = img_size*img_size*channels
HIDDEN_DIM   = 1024
latent_dim   = 30
EPOCHS       = 6
ctr = 0

import tensorflow as tf
import sys
import numpy as np
import scipy.misc
import os

def lrelu(x, leak = 0.2, name = 'lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1+leak)
        f2 = 0.5 * (1-leak)
        return f1*x + f2*abs(x)

def linear(input, output_dim, scope=None, stddev=0.01):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b

def encoder(images):
    output = tf.convert_to_tensor(images)
    output = lrelu(linear(output, HIDDEN_DIM, scope='e_layer1', stddev=0.01))
    output = lrelu(linear(output, HIDDEN_DIM, scope='e_layer2', stddev=0.01))
    return linear(output, latent_dim, scope='e_layer3', stddev=0.01)

def decoder(latent_vars,reuse = False):
    if reuse:
            tf.get_variable_scope().reuse_variables()
    output = tf.convert_to_tensor(latent_vars)
    print (output.get_shape())
    output = lrelu(linear(output, HIDDEN_DIM, scope='d_layer1', stddev=0.01))
    output = lrelu(linear(output, HIDDEN_DIM, scope='d_layer2', stddev=0.01))
    return   tf.sigmoid(linear(output, input_dim, scope='d_layer3', stddev=0.01))

def discriminator(inputs,reuse = False):
    if reuse:
            tf.get_variable_scope().reuse_variables()
    output = tf.convert_to_tensor(inputs)
    output = lrelu(linear(output, HIDDEN_DIM, scope='dis_layer1', stddev=0.01))
    output = lrelu(linear(output, HIDDEN_DIM, scope='dis_layer2', stddev=0.01))
    return tf.sigmoid(linear(output, 1, scope='dis_layer3', stddev=0.01))

def noise(n_samples):
    batch_z = np.random.uniform(-1, 1, [n_samples,latent_dim]).astype(np.float32)
    return batch_z

x = tf.placeholder(tf.float32, [None, input_dim])

latents = encoder(x)
reconstructions = decoder(latents)

with tf.variable_scope('latent_score') as scope:
    tf.VariableScope.reuse =None
    latent_score = discriminator(latents,reuse = False)

reg_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(latent_score),logits = latent_score))
true_noise_score = discriminator(noise(batch_size),reuse = True)
reconst_cost = tf.reduce_mean(tf.squared_difference(reconstructions,x))

full_enc_cost = 1000*reconst_cost + reg_cost

dec_cost = reconst_cost

discrim_cost  =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(latent_score),logits = latent_score))
discrim_cost +=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(true_noise_score),logits = true_noise_score))

t_vars = tf.trainable_variables()
enc_params  = [var for var in t_vars if 'e_' in var.name]
dec_params  = [var for var in t_vars if 'd_' in var.name]                             
discrim_params = [var for var in t_vars if 'dis_' in var.name]                              
full_cost = full_enc_cost + dec_cost + discrim_cost

lr = tf.placeholder(tf.float32)
e_optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(full_enc_cost, var_list=enc_params)
d_optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(dec_cost, var_list=dec_params)
dis_optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(discrim_cost, var_list=discrim_params)

tf.summary.scalar('Encoder Loss', full_enc_cost)
tf.summary.scalar('Decoder Loss', dec_cost)
tf.summary.scalar('Discriminator Loss', discrim_cost)
tf.summary.scalar('Total Loss', full_cost)

merge = tf.summary.merge_all()
logdir = "./Tensorboard"

session_conf = tf.ConfigProto(gpu_options =tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
                                  allow_soft_placement=True,
                                  log_device_placement=False)

sess = tf.InteractiveSession(config=session_conf)
# Train

writer = tf.summary.FileWriter(logdir, sess.graph)
#Initilaize the learning rate
l_rate = 1e-2
tf.global_variables_initializer().run()
for step in range(EPOCHS):
    dataset = utils.get_image_paths("/home/kunal/fast_style_transfer_per/Coco")
    length = len(dataset)
    #A queue to maintain the last 7 total_loss
    loss_count = collections.deque(7*[0],7)
    for i in range(length/batch_size):
        batch_xs = utils.next_batch(dataset,i,batch_size, ctr)
        _,_,_ = sess.run([d_optim,e_optim, dis_optim], feed_dict={x: batch_xs, lr :l_rate})
        reconstructions_, latents_ = sess.run([reconstructions,latents], feed_dict={x: batch_xs, lr : l_rate}) 
        enc_loss = sess.run(full_enc_cost, feed_dict={x: batch_xs, lr :l_rate})
        dec_loss = sess.run(dec_cost, feed_dict={x:batch_xs, lr : l_rate})
        discrim_loss = sess.run(discrim_cost, feed_dict={x:batch_xs, lr :l_rate})
        total_loss = enc_loss + dec_loss + discrim_loss
        loss_dict = [total_loss, enc_loss, dec_loss, discrim_loss]
        #The learning rate decay algorithm
        if i%350 == 0:
            sum = 0
            for k in range(7):
                sum = sum + loss_count[k]
    	    avg = sum/7
    	    if step == 0:
                deviation = 5
                decay = 3
    	    elif step == 1:
                deviation = 4
                decay = 1e1
    	    elif step == 2:
                deviation = 3
                decay = 1e1
    	    elif step == 3:
                deviation = 2.5
    	    else:
                deviation = 2
            #Checking if the current total_loss is within the deviation of the average of the last 7 loss counts
            if total_loss > avg-deviation and total_loss < avg+deviation:
                l_rate = l_rate/decay
                print "LEARNING RATE HAS BEEN CHANGED TO {} since loss has plateaued".format(l_rate)
    	    elif total_loss > avg:
                l_rate = l_rate/decay
                print "LEARNING RATE HAS BEEN CHANGED TO {} since loss is overshooting".format(l_rate)
        
        if i%50 == 0:
            loss_count.append(total_loss)	
            print "Epoch = {}, Iteration = {}".format(step, i)
            print "Total Loss = {0}, Encoder loss = {1}, Decoder Loss = {2}, Discriminator Loss = {3}".format(*loss_dict)
            
        if i%100 == 0:
            test_image = utils.next_image(dataset, i,batch_size)
            ctr = ctr+1
            sample_test = sess.run(reconstructions, feed_dict = {x : test_image, lr : 1e-3})
    	    test_image.astype(np.float32)
    	    test_image = np.reshape(test_image, (128, 128, 3))
    	    sample_test = np.reshape(sample_test, ( 128, 128, 3))
    	    scipy.misc.imsave("./output/imgtest_epoch_{}_iteration_{}.bmp".format(step, i),sample_test)
    	    scipy.misc.imsave("./output/imgorg1_epoch_{}_iteration_{}.bmp".format(step, i),test_image )


