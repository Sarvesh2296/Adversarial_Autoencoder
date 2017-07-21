from keras.preprocessing.image import array_to_img
from PIL import Image
import scipy.misc
import utils2 as utils

#Hyperparameters
batch_size   = 32
img_size     = 128
channels     = 3
input_dim    = img_size*img_size*channels
hidden_layer = 1024
latent_dim   = 30
epochs       = 5
ctr = 0

import tensorflow as tf

#Directory from where to import the whole data
dataset = utils.get_image_paths("/home/kunal/fast_style_transfer_per/Coco")

import sys
import numpy as np
import scipy.misc
import os

#Defining a lrelu function to use as an Activation function 
def lrelu(x, leak = 0.2, name = 'lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1+leak)
        f2 = 0.5 * (1-leak)
        return f1*x + f2*abs(x)

#Helper function to calculate and update the layers: They are a helper function to construct the layers
def linear(input, output_dim, scope=None, stddev=0.01):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b

#A two hidden layer neural encoder which encodes the images to latent dimension specified 
def encoder(images):
    output = tf.convert_to_tensor(images)
    output = lrelu(linear(output, hidden_layer, scope='e_layer1', stddev=0.01))
    output = lrelu(linear(output, hidden_layer, scope='e_layer2', stddev=0.01))
    return linear(output, latent_dim, scope='e_layer3', stddev=0.01)

#A two  hidden layer neural decoder which decodes the image from the latent dimension specified
def decoder(latent_vars,reuse = False):
    if reuse:
            tf.get_variable_scope().reuse_variables()
    output = tf.convert_to_tensor(latent_vars)
    print (output.get_shape())
    output = lrelu(linear(output, hidden_layer, scope='d_layer1', stddev=0.01))
    output = lrelu(linear(output, hidden_layer, scope='d_layer2', stddev=0.01))
    return   tf.sigmoid(linear(output, input_dim, scope='d_layer3', stddev=0.01))

#A two layer neural discriminator which takes the latent representation as the input and outputs a single node representing a true/false:
def discriminator(inputs,reuse = False):
    if reuse:
            tf.get_variable_scope().reuse_variables()
    output = tf.convert_to_tensor(inputs)
    output = lrelu(linear(output, hidden_layer, scope='dis_layer1', stddev=0.01))
    output = lrelu(linear(output, hidden_layer, scope='dis_layer2', stddev=0.01))
    return tf.sigmoid(linear(output, 1, scope='dis_layer3', stddev=0.01))

#To generate the prior distribution following which the latent distribution will be shaped
def noise(n_samples):
    batch_z = np.random.uniform(-1, 1, [n_samples,latent_dim]).astype(np.float32)
    return batch_z
# Placeholder for the images with any batch size
x = tf.placeholder(tf.float32, [None, input_dim])

#The network is constructed
latents = encoder(x)
reconstructions = decoder(latents)
#The encoder generated latent distribution sample is taken to calculate the latent score given by the decoder
with tf.variable_scope('latent_score') as scope:
    tf.VariableScope.reuse =None
    latent_score = discriminator(latents,reuse = False)

#Loss function is defined between latent score and 1s. Encoder wants this.
reg_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(latent_score),logits = latent_score))
true_noise_score = discriminator(noise(batch_size),reuse = True)

#Reconstruction loss between input to encoder and output of decoder
reconst_cost = tf.reduce_mean(tf.squared_difference(reconstructions,x))

#This loss is defined to initiate the reconstruction loss first and once the representaion is learnt, then shape it through the adversarial network
full_enc_cost = 1000*reconst_cost + reg_cost

dec_cost = reconst_cost

# Discriminator loss which tries to make scores 0s when latent distribution is the input and 1s if prior distribution is the input
discrim_cost  =tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.zeros_like(latent_score),logits = latent_score))
discrim_cost +=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = tf.ones_like(true_noise_score),logits = true_noise_score))

#Updating all thr trainable variables
t_vars = tf.trainable_variables()
enc_params  = [var for var in t_vars if 'e_' in var.name]
dec_params  = [var for var in t_vars if 'd_' in var.name]                             
discrim_params = [var for var in t_vars if 'dis_' in var.name]                              
full_cost = full_enc_cost + dec_cost + discrim_cost

lr = tf.placeholder(tf.float32)

#Defining the optimizers with variable learning rate which will be passed on during the training
e_optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(full_enc_cost, var_list=enc_params)
d_optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(dec_cost, var_list=dec_params)
dis_optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(discrim_cost, var_list=discrim_params)

#The summary pointers to get the summary of each of the variables in Tensorboard
# If you want the Tensorboard summary, kindly uncomment the commands written within the training code
tf.summary.scalar('Encoder Loss', full_enc_cost)
tf.summary.scalar('Decoder Loss', dec_cost)
tf.summary.scalar('Discriminator Loss', discrim_cost)
tf.summary.scalar('Total Loss', full_cost)

#Merge all the summaries
merge = tf.summary.merge_all()
logdir = "./Tensorboard"

session_conf = tf.ConfigProto(gpu_options =tf.GPUOptions(per_process_gpu_memory_fraction=0.9),
                                  allow_soft_placement=True,
                                  log_device_placement=False)

sess = tf.InteractiveSession(config=session_conf)

writer = tf.summary.FileWriter(logdir, sess.graph)

tf.global_variables_initializer().run()
for step in range(epochs):
    length = len(dataset)
    #This loop is to ensure that the whole dataset is iterated once
    for i in range(length/batch_size):
        if step == 0:
            #Loads the next batch. next_batch is defined in the Utils2.py library
            batch_xs = utils.next_batch(dataset,i,batch_size, ctr)
            _,_,_ = sess.run([d_optim,e_optim, dis_optim], feed_dict={x: batch_xs, lr : 1e-4})
            reconstructions_, latents_ = sess.run([reconstructions,latents], feed_dict={x: batch_xs, lr : 1e-4}) 
            enc_loss = sess.run(full_enc_cost, feed_dict={x: batch_xs, lr : 1e-4})
            dec_loss = sess.run(dec_cost, feed_dict={x:batch_xs, lr : 1e-4})
            discrim_loss = sess.run(discrim_cost, feed_dict={x:batch_xs, lr : 1e-4})
            total_loss = enc_loss + dec_loss + discrim_loss
            loss_dict = [total_loss, enc_loss, dec_loss, discrim_loss]
        elif step == 1 or step == 2:
	       batch_xs = utils.next_batch(dataset,i,batch_size, ctr)
            _,_,_ = sess.run([d_optim,e_optim, dis_optim], feed_dict={x: batch_xs, lr : 5e-5})
            reconstructions_, latents_ = sess.run([reconstructions,latents], feed_dict={x: batch_xs,lr : 5e-5})
            enc_loss = sess.run(full_enc_cost, feed_dict={x: batch_xs,lr : 5e-5})
            dec_loss = sess.run(dec_cost, feed_dict={x:batch_xs, lr : 5e-5})
            discrim_loss = sess.run(discrim_cost, feed_dict={x:batch_xs, lr : 5e-5})
            total_loss = enc_loss + dec_loss + discrim_loss
            loss_dict = [total_loss, enc_loss, dec_loss, discrim_loss]
        elif step>2:
            batch_xs = utils.next_batch(dataset,i,batch_size, ctr)
            _,_,_ = sess.run([d_optim,e_optim, dis_optim], feed_dict={x: batch_xs, lr : 1e-5})
            reconstructions_, latents_ = sess.run([reconstructions,latents], feed_dict={x: batch_xs,lr : 1e-5})
            enc_loss = sess.run(full_enc_cost, feed_dict={x: batch_xs,lr : 1e-5})
            dec_loss = sess.run(dec_cost, feed_dict={x:batch_xs, lr : 1e-5})
            discrim_loss = sess.run(discrim_cost, feed_dict={x:batch_xs, lr : 1e-5})
            total_loss = enc_loss + dec_loss + discrim_loss
            loss_dict = [total_loss, enc_loss, dec_loss, discrim_loss]
        if i%100 == 0:
            if step==0:
	           print "Learning rate is {}".format(1e-4)
    	    elif step ==1:
    		   print "Learning rate is {}".format(5e-5)
    	    elif step>2:
    		   print "Learning rate is {}".format(1e-5)
    	    print "Total Loss = {0}, Encoder loss = {1}, Decoder Loss = {2}, Discriminator Loss = {3}".format(*loss_dict)
          #   summary = sess.run(merge, feed_dict={x:batch_x})
     	    # writer.add_summary(summary, i)
        if i%200 == 0:
            #For testing the model and saving the image to see the results
            test_image = utils.next_image(dataset, i,batch_size)
	        ctr = ctr+1
            sample_test = sess.run(reconstructions, feed_dict = {x : test_image, lr : 1e-3})
    	    test_image.astype(np.float32)
    	    test_image = np.reshape(test_image, (128, 128, 3))
    	    sample_test = np.reshape(sample_test, ( 128, 128, 3))
    	    scipy.misc.imsave("./output/imgtest_epoch_{}_iteration_{}.bmp".format(step, i),sample_test)
    	    scipy.misc.imsave("./output/imgorg1_epoch_{}_iteration_{}.bmp".format(step, i),test_image )


